r"""
Gravitational-wave transient surveys provide a biased sample of the astrophysical population.
The likelihood function used for population inference is given be

.. math::

    {\cal L}(\{d_i\} | \Lambda) &= \prod_i {\cal L}(d_i | \Lambda, {\rm det})

    &= \prod_i \frac{{\cal L}(d_i | \Lambda)}{P_{\rm det}(\Lambda)}

    &= \frac{1}{P_{\rm det}^{N}(\Lambda)} \prod_i \int d\theta_i p(d_i | \theta_i) \pi(\theta_i | \Lambda).

The quantity :math:`P_{\rm det}(\Lambda)` is the detection probability for a single source (see `<selection.html>`_).

The integrals over the per-event parameters :math:`\theta_i` are typically performed using Monte Carlo integration

.. math::

    \hat{{\cal L}}(d_i | \Lambda) = \frac{1}{K} \sum_{k=1}^K \frac{\pi(\theta_k | \Lambda)}{\pi(\theta_k | \varnothing)}.

The full approximate log-likelihood is then given by

.. math::

    \ln \hat{\cal L}(\{d_i\} | \Lambda) = \sum_i \ln \hat{{\cal L}}(d_i | \Lambda) - N \ln \hat{P}_{\rm det}(\Lambda).

This approximation is implemented in :class:`gwpopulation.hyperpe.HyperparameterLikelihood`.

There is another related expression for the likelihood as the result of an inhomoegeneous Poisson process.
In this case the likelihood is given by

.. math::

    \ln {\cal L}(\{d_i\} | \Lambda) = N \ln R - N_{\rm exp}(\Lambda)
    + \sum_i \ln \hat{{\cal L}}(d_i | \Lambda)

Here :math:`R` is the total merger rate and :math:`T` is the total observation time
and :math:`N_{\rm exp}(\Lambda) = RT\hat{P}_{\rm det}(\Lambda)`.
This is implemented in :class:`gwpopulation.hyperpe.RateLikelihood`.

Each of these Monte Carlo integrals have associated uncertainties which are propagated through the likelihood calculation
and can be calculated using :func:`gwpopulation.hyperpe.HyperparameterLikelihood.ln_likelihood_and_variance`.
"""

import types

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.utils import logger, infer_args_from_function_except_n_args
from bilby.hyper.model import Model

from .utils import get_name, to_number, to_numpy
from .models.redshift import _Redshift
from .models.mass import BaseSmoothedMassDistribution
from .experimental.sgwb_utils import dEdf, _RhoC

xp = np

__all__ = ["HyperparameterLikelihood", "RateLikelihood", "LocalMergerRateLikelihood", "StochasticLikelihood", "JointCBCStochasticLikelihood", "xp"]


def _inv_efunc_flat_wcdm(z, Om0, w0=-1.0):
    r"""
    Standalone inverse E(z) for a flat :math:`w\mathrm{CDM}` cosmology:
 
    .. math::
 
        E^{-1}(z) = \left[\Omega_{m,0}(1+z)^3
                     + (1 - \Omega_{m,0})(1+z)^{3(1+w_0)}\right]^{-1/2}
 
    When :math:`w_0 = -1` this reduces to flat :math:`\Lambda\mathrm{CDM}`.
 
    This is a pure-arithmetic function with no data-dependent branches,
    making it safe for use inside :func:`jax.jit`-traced code.  It
    replaces the call to ``wcosmo``'s :meth:`inv_efunc`, which triggers
    ``TracerArrayConversionError`` because of internal ``if`` branches on
    the value of ``Om0``.
    """
    zp1 = 1.0 + z
    return (Om0 * zp1 ** 3 + (1.0 - Om0) * zp1 ** (3.0 * (1.0 + w0))) ** (-0.5)



class HyperparameterLikelihood(Likelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions with
    including selection effects.

    See Eq. (34) of `Thrane and Talbot <https://arxiv.org/abs/1809.02293>`_
    for a definition.

    For the uncertainty calculation see the Appendix of
    `Golomb and Talbot <https://arxiv.org/abs/2106.15745>`_ and
    `Farr <https://arxiv.org/abs/1904.10879>`_.
    """

    def __init__(
        self,
        posteriors,
        hyper_prior,
        ln_evidences=None,
        max_samples=1e100,
        selection_function=lambda args: 1,
        conversion_function=lambda args: (args, None),
        maximum_uncertainty=xp.inf,
    ):
        """
        Parameters
        ----------
        posteriors: list
            An list of pandas data frames of samples sets of samples.
            Each set may have a different size.
            These can contain a `prior` column containing the original prior
            values.
        hyper_prior: `bilby.hyper.model.Model`
            The population model, this can alternatively be a function.
        ln_evidences: list, optional
            Log evidences for single runs to ensure proper normalisation
            of the hyperparameter likelihood. If not provided, the original
            evidences will be set to 0. This produces a Bayes factor between
            the sampling power_prior and the hyperparameterised model.
        selection_function: func
            Function which evaluates your population selection function.
        conversion_function: func
            Function which converts a dictionary of sampled parameter to a
            dictionary of parameters of the population model.
        max_samples: int, optional
            Maximum number of samples to use from each set.
        maximum_uncertainty: float
            The maximum allowed uncertainty in the natural log likelihood.
            If the uncertainty is larger than this value a log likelihood of
            -inf will be returned. Default = inf
        """

        self.samples_per_posterior = max_samples
        self.data = self.resample_posteriors(posteriors, max_samples=max_samples)

        if isinstance(hyper_prior, types.FunctionType):
            hyper_prior = Model([hyper_prior])
        elif not (
            hasattr(hyper_prior, "parameters")
            and callable(getattr(hyper_prior, "prob"))
        ):
            raise AttributeError(
                "hyper_prior must either be a function, "
                "or a class with attribute 'parameters' and method 'prob'"
            )
        self.hyper_prior = hyper_prior
        super().__init__()

        if "prior" in self.data:
            self.sampling_prior = self.data.pop("prior")
        else:
            logger.info("No prior values provided, defaulting to 1.")
            self.sampling_prior = 1

        if ln_evidences is not None:
            self.total_noise_evidence = np.sum(ln_evidences)
        else:
            self.total_noise_evidence = np.nan

        self.conversion_function = conversion_function
        self.selection_function = selection_function

        self.n_posteriors = len(posteriors)
        self.maximum_uncertainty = maximum_uncertainty
        self._inf = np.nan_to_num(np.inf)

    __doc__ += __init__.__doc__

    @property
    def maximum_uncertainty(self):
        """
        The maximum allowed uncertainty in the estimate of the log-likelihood.
        If the uncertainty is larger than this value a log likelihood of -inf
        is returned.
        """
        return self._maximum_uncertainty

    @maximum_uncertainty.setter
    def maximum_uncertainty(self, value):
        self._maximum_uncertainty = value
        if value in [xp.inf, np.inf]:
            self._max_variance = value
        else:
            self._max_variance = value**2

    def ln_likelihood_and_variance(self, parameters):
        """
        Compute the ln likelihood estimator and its variance.
        """
        parameters, added_keys = self.conversion_function(parameters)
        ln_bayes_factors, variances = self._compute_per_event_ln_bayes_factors(
            parameters
        )
        ln_l = xp.sum(ln_bayes_factors)
        variance = xp.sum(variances)
        selection, selection_variance = self._get_selection_factor(
            parameters=parameters
        )
        variance += selection_variance
        ln_l += selection
        return ln_l, to_number(variance, float)

    def log_likelihood_ratio(self, parameters):
        ln_l, variance = self.ln_likelihood_and_variance(parameters=parameters)
        ln_l = xp.nan_to_num(ln_l, nan=-xp.inf)
        ln_l -= xp.nan_to_num(xp.inf * (self.maximum_uncertainty < variance), nan=0)
        return to_number(xp.nan_to_num(ln_l), float)

    def noise_log_likelihood(self):
        return self.total_noise_evidence

    def log_likelihood(self, parameters):
        return self.noise_log_likelihood() + self.log_likelihood_ratio(
            parameters=parameters
        )

    def _compute_per_event_ln_bayes_factors(
        self, parameters, *, return_uncertainty=True
    ):
        weights = self.hyper_prior.prob(self.data, **parameters) / self.sampling_prior
        expectation = xp.mean(weights, axis=-1)
        if return_uncertainty:
            square_expectation = xp.mean(weights**2, axis=-1)
            variance = (square_expectation - expectation**2) / (
                self.samples_per_posterior * expectation**2
            )
            return xp.log(expectation), variance
        else:
            return xp.log(expectation)

    def _get_selection_factor(self, parameters, *, return_uncertainty=True):
        selection, variance = self._selection_function_with_uncertainty(
            parameters=parameters
        )
        total_selection = -self.n_posteriors * xp.log(selection)
        if return_uncertainty:
            total_variance = self.n_posteriors**2 * xp.divide(variance, selection**2)
            return total_selection, total_variance
        else:
            return total_selection

    def _selection_function_with_uncertainty(self, parameters):
        result = self.selection_function(parameters)
        if isinstance(result, tuple):
            selection, variance = result
        else:
            selection = result
            variance = 0.0
        return selection, variance

    def generate_extra_statistics(self, sample):
        r"""
        Given an input sample, add extra statistics

        Adds:

        - :code:`ln_bf_idx`: :math:`\frac{\ln {\cal L}(d_{i} | \Lambda)}
          {\ln {\cal L}(d_{i} | \varnothing)}`
          for each of the events in the data
        - :code:`selection`: :math:`P_{\rm det}`
        - :code:`var_idx`, :code:`selection_variance`: the uncertainty in
          each Monte Carlo integral
        - :code:`total_variance`: the total variance in the likelihood

        .. note::

            The quantity :code:`selection_variance` is the variance in
            :code:`P_{\rm det}` and not the total variance from the contribution
            of the selection function to the likelihood.

        Parameters
        ----------
        sample: dict
            Input sample to compute the extra things for.

        Returns
        -------
        sample: dict
            The input dict, modified in place.
        """
        parameters, added_keys = self.conversion_function(sample.copy())
        ln_ls, variances = self._compute_per_event_ln_bayes_factors(
            parameters, return_uncertainty=True
        )
        total_variance = sum(variances)
        for ii in range(self.n_posteriors):
            sample[f"ln_bf_{ii}"] = to_number(ln_ls[ii], float)
            sample[f"var_{ii}"] = to_number(variances[ii], float)
        selection, variance = self._selection_function_with_uncertainty(
            parameters=parameters
        )
        variance /= selection**2
        selection_variance = variance * self.n_posteriors**2
        sample["selection"] = selection
        sample["selection_variance"] = variance
        total_variance += selection_variance
        sample["variance"] = to_number(total_variance, float)
        return sample

    def generate_rate_posterior_sample(self, parameters):
        r"""
        Generate a sample from the posterior distribution for rate assuming a
        :math:`1 / R` prior.

        The likelihood evaluated is analytically marginalized over rate.
        However the rate dependent likelihood can be trivially written.

        .. math::
            p(R) = \Gamma(n=N, \text{scale}=\mathcal{V})

        Here :math:`\Gamma` is the Gamma distribution, :math:`N` is the number
        of events being analyzed and :math:`\mathcal{V}` is the total observed 4-volume.

        .. note::

            This function only uses the :code:`numpy` backend. It can be used
            with the other backends as it returns a float, but does not support
            e.g., autodifferentiation with :code:`jax`.

        Returns
        -------
        rate: float
            A sample from the posterior distribution for rate.
        """
        from scipy.stats import gamma

        if hasattr(self.selection_function, "detection_efficiency") and hasattr(
            self.selection_function, "surveyed_hypervolume"
        ):
            efficiency, _ = self.selection_function.detection_efficiency(parameters)
            vt = efficiency * self.selection_function.surveyed_hypervolume(parameters)
        else:
            vt = self.selection_function(parameters)
        rate = gamma(a=self.n_posteriors).rvs() / vt
        return rate

    def resample_posteriors(self, posteriors, max_samples=1e300):
        """
        Convert list of pandas DataFrame object to dict of arrays.

        Parameters
        ----------
        posteriors: list
            List of pandas DataFrame objects.
        max_samples: int, opt
            Maximum number of samples to take from each posterior,
            default is length of shortest posterior chain.

        Returns
        -------
        data: dict
            Dictionary containing arrays of size (n_posteriors, max_samples)
            There is a key for each shared key in posteriors.
        """
        for posterior in posteriors:
            max_samples = min(len(posterior), max_samples)
        data = {key: [] for key in posteriors[0]}
        logger.debug(f"Downsampling to {max_samples} samples per posterior.")
        self.samples_per_posterior = max_samples
        for posterior in posteriors:
            temp = posterior.sample(self.samples_per_posterior)
            for key in data:
                data[key].append(temp[key])
        for key in data:
            data[key] = xp.array(data[key])
        return data

    def posterior_predictive_resample(self, samples, return_weights=False):
        """
        Resample the original single event posteriors to use the PPD from each
        of the other events as the prior.

        Parameters
        ----------
        samples: pd.DataFrame, dict, list
            The samples to do the weighting over, typically the posterior from
            some run.
        return_weights: bool, optional
            Whether to return the per-sample weights, default = :code:`False`

        Returns
        -------
        new_samples: dict
            Dictionary containing the weighted posterior samples for each of
            the events.
        weights: array-like
            Weights to apply to the samples, only if :code:`return_weights == True`.
        """
        import pandas as pd
        from tqdm.auto import tqdm

        if isinstance(samples, pd.DataFrame):
            samples = [dict(samples.iloc[ii]) for ii in range(len(samples))]
        elif isinstance(samples, dict):
            samples = [samples]
        weights = xp.zeros((self.n_posteriors, self.samples_per_posterior))
        event_weights = xp.zeros(self.n_posteriors)
        for sample in tqdm(samples):
            parameters, added_keys = self.conversion_function(sample.copy())
            new_weights = (
                self.hyper_prior.prob(self.data, **parameters) / self.sampling_prior
            )
            event_weights += xp.mean(new_weights, axis=-1)
            new_weights = (new_weights.T / xp.sum(new_weights, axis=-1)).T
            weights += new_weights
        weights = (weights.T / xp.sum(weights, axis=-1)).T
        new_idxs = xp.empty_like(weights, dtype=int)
        for ii in range(self.n_posteriors):
            if "jax" in xp.__name__:
                from jax import random

                rng_key = random.PRNGKey(np.random.randint(10000000))
                new_idxs = new_idxs.at[ii].set(
                    random.choice(
                        rng_key,
                        xp.arange(self.samples_per_posterior),
                        shape=(self.samples_per_posterior,),
                        replace=True,
                        p=weights[ii],
                    )
                )
            else:
                new_idxs[ii] = xp.asarray(
                    np.random.choice(
                        range(self.samples_per_posterior),
                        size=self.samples_per_posterior,
                        replace=True,
                        p=to_numpy(weights[ii]),
                    )
                )
        new_samples = {
            key: xp.vstack(
                [self.data[key][ii, new_idxs[ii]] for ii in range(self.n_posteriors)]
            )
            for key in self.data
        }
        event_weights = list(event_weights)
        weight_string = " ".join([f"{float(weight):.1f}" for weight in event_weights])
        logger.info(f"Resampling done, sum of weights for events are {weight_string}")
        if return_weights:
            return new_samples, weights
        else:
            return new_samples

    @property
    def meta_data(self):
        return dict(
            model=[get_name(model) for model in self.hyper_prior.models],
            data={key: to_numpy(self.data[key]) for key in self.data},
            n_events=self.n_posteriors,
            sampling_prior=to_numpy(self.sampling_prior),
            samples_per_posterior=self.samples_per_posterior,
        )


class RateLikelihood(HyperparameterLikelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions
    and estimating rates with including selection effects.

    See Eq. (34) of `Thrane and Talbot <https://arxiv.org/abs/1809.02293>`_
    for a definition.

    """

    __doc__ += HyperparameterLikelihood.__init__.__doc__

    def _get_selection_factor(self, parameters, *, return_uncertainty=True):
        r"""
        The selection factor for the rate likelihood is

        .. math::

            \ln P_{\rm det} = N \ln R - N_{\rm exp}(\Lambda)

        The uncertainty is given by

        .. math::

            \sigma^2 = \frac{N_{\rm exp}(\Lambda) \sigma^2_{\rm det}}{P_{\rm det}^2}

        Parameters
        ----------
        return_uncertainty: bool
            Whether to return the uncertainty in the selection factor.
        """
        selection, variance = self._selection_function_with_uncertainty(
            parameters=parameters
        )
        n_expected = selection * parameters["rate"]
        total_selection = -n_expected + self.n_posteriors * xp.log(parameters["rate"])
        if return_uncertainty:
            total_variance = n_expected * variance / selection**2
            return total_selection, total_variance
        else:
            return total_selection

    def generate_rate_posterior_sample(self, parameters):
        """
        Since the rate is a sampled parameter,
        this simply returns the current value of the rate parameter.
        """
        return parameters["rate"]


class LocalMergerRateLikelihood(HyperparameterLikelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions
    and estimating local merger rates with including selection effects.

    see the prepared doc.

    """

    __doc__ += HyperparameterLikelihood.__init__.__doc__

    def _get_selection_factor(self, parameters, *, return_uncertainty=True):
        r"""
        The selection factor for the rate likelihood is

        .. math::

            \ln P_{\rm det} = -N_exp + N * \ln (R0 * V)

        The uncertainty is given by

        .. math::

            \sigma^2 = \frac{N_{\rm exp}(\Lambda) \sigma^2_{\rm det}}{P_{\rm det}^2}

        Parameters
        ----------
        return_uncertainty: bool
            Whether to return the uncertainty in the selection factor.
        """
        efficiency, variance = self._selection_function_with_uncertainty(
            parameters=parameters
        )
        vt = efficiency * self.selection_function.surveyed_hypervolume(parameters)
        N_exp = vt * parameters["rate"]
        total_selection = -N_exp + self.n_posteriors * xp.log(parameters["rate"] * self.selection_function.surveyed_hypervolume(parameters))
        if return_uncertainty:
            total_variance = N_exp * variance / efficiency**2
            return total_selection, total_variance
        else:
            return total_selection


class StochasticLikelihood(Likelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions
    and estimating the stochastic gravitational-wave background (SGWB) with existing GWPopulation models.

    see the prepared doc.

    """
    def __init__(
        self,
        stochastic_data,
        hyper_prior, # notice that we have to clearly specify the zmax of Redshift model in the hyper_prior.
        dEdf_freqs, draw_samples_kwargs,
        conversion_function=lambda args: (args, None),
    ):
        """
        Parameters
        ----------
        stochastic_data: dict
            Measured SGWB data.  Must contain:
 
            - ``CIJ``: the cross-correlation estimator :math:`\\hat{C}_{IJ}(f)`
            - ``sigma``: the associated :math:`1\\sigma` uncertainty per frequency bin
            - ``frequencies``: the frequency array

        hyper_prior: `bilby.hyper.model.Model` or callable
            The population model (mass, spin, redshift, …).
        dEdf_freqs: 
        drawn_samples_kwargs: 
        conversion_function: callable, optional
            Function that converts a dictionary of sampled parameters
            to a dictionary of population-model parameters.
        """
        if isinstance(hyper_prior, types.FunctionType):
            hyper_prior = Model([hyper_prior])
        elif not (
            hasattr(hyper_prior, "parameters")
            and callable(getattr(hyper_prior, "prob"))
        ):
            raise AttributeError(
                "hyper_prior must either be a function, "
                "or a class with attribute 'parameters' and method 'prob'"
            )
        self.hyper_prior = hyper_prior
        self.conversion_function = conversion_function

        self.dEdf_freqs = dEdf_freqs
        self.draw_samples_kwargs = draw_samples_kwargs
        #todo create another attribute to save redshift from samples input.

        # after accesing luminosity information, that column should be removed from samples.

        # real measurements of the stochastic search.
        self.CIJ = stochastic_data["CIJ"]
        self.sigma = stochastic_data["sigma"]
        self.frequencies = stochastic_data["frequencies"]

        
        super().__init__()

        # used for cosmological inference.

        self._drawn_samples()
        self._calculate_dEdf()

        self.cosmology_model = self._find_cosmo_model()
        self.redshift_model = self._find_redshift_model()
        self.mass_function_model = self._find_mass_function_model()

        # Cache default Om0 and w0 for fixed-cosmology runs so that
        # _compute_weights can fall back when these are not sampled parameters.
        if self.cosmology_model is not None and hasattr(self.cosmology_model, '_cosmo'):
            cosmo = self.cosmology_model._cosmo
            if hasattr(cosmo, 'Om0'):
                self._default_Om0 = float(cosmo.Om0) if not callable(cosmo.Om0) else 0.3065
            else:
                self._default_Om0 = 0.3065
            if hasattr(cosmo, 'w0'):
                self._default_w0 = float(cosmo.w0) if not callable(cosmo.w0) else -1.0
            else:
                self._default_w0 = -1.0
            if hasattr(cosmo, 'H0'):
                self._default_H0 = float(cosmo.H0) if not callable(cosmo.H0) else 67.74
            else:
                self._default_H0 = 67.74
        else:
            self._default_Om0 = 0.3065
            self._default_w0 = -1.0
            self._default_H0 = 67.74

        self._noise_log_likelihood = None
    
    def _drawn_samples(self):
        # initialize a dictionary of mass_1, mass_ratio, redshift for p(m1, q, z)
        num_samples = int(self.draw_samples_kwargs['num_samples'])

        self.m1s_drawn = np.random.uniform(low=self.draw_samples_kwargs['ref_mMin'], high=self.draw_samples_kwargs['ref_mMax'], size=num_samples)
        self.qs_drawn = np.random.uniform(low=0.01, high=1.0, size=int(num_samples))
        self.m2s_drawn = self.m1s_drawn*self.qs_drawn
        self.zs_drawn = np.random.uniform(low=0,high=self.draw_samples_kwargs['ref_zmax'],size=num_samples)

        self.mass_q_samples =  {'mass_1': self.m1s_drawn, 'mass_ratio': self.qs_drawn}
        self.p_m1_q_z_drawn = np.ones(num_samples) * (1./ (self.draw_samples_kwargs['ref_mMax'] - self.draw_samples_kwargs['ref_mMin'])) * (1./ (1. - 0.01)) * (1./ self.draw_samples_kwargs['ref_zmax'])
    

    def _calculate_dEdf(self):
        num_samples = int(self.draw_samples_kwargs['num_samples'])
        self.dEdfs = xp.asarray([dEdf(self.m1s_drawn[ii]+self.m2s_drawn[ii], self.dEdf_freqs*(1+self.zs_drawn[ii]),
                       eta=self.m2s_drawn[ii]/self.m1s_drawn[ii]/(1+self.m2s_drawn[ii]/self.m1s_drawn[ii])**2) for ii in range(num_samples)])


    def _find_cosmo_model(self):
        """Find model in hyper_prior that has cosmology method."""
        for model in self.hyper_prior.models:
            if hasattr(model, "cosmology"):
                logger.info("cosmology model found in hyper_prior.")
                return model
        logger.info("No cosmology model found in hyper_prior, using default cosmology.")
        return None

    def _find_redshift_model(self):
        """Find the redshift model in hyper_prior."""
        for model in self.hyper_prior.models:
            if isinstance(model, _Redshift):
                logger.info("Redshift model found in hyper_prior.")
                return model
        logger.info("No Redshift model found in hyper_prior, probably needs further check.")
        return None
    
    def _find_mass_function_model(self):
        """Find the mass function model in hyper_prior"""
        for model in self.hyper_prior.models:
            if isinstance(model, BaseSmoothedMassDistribution):
                logger.info("mass function model found in hyper_prior")
                return model
        logger.info("No mass function model found in hyper_prior, probably needs further check.")
        return None

    def _get_function_parameters(self, func, **kwargs):
        """
        If the function is a class method we need to remove more arguments or
        have the variable names provided in the class.
        """
        if hasattr(func, "variable_names"):
            param_keys = func.variable_names
        else:
            param_keys = infer_args_from_function_except_n_args(func, n=0)
            ignore = ["dataset", "data", "self", "cls"]
            for key in ignore:
                if key in param_keys:
                    del param_keys[param_keys.index(key)]
        parameters = dict()
        for key in param_keys:
            if key in kwargs:
                parameters[key] = kwargs[key]
            else:
                raise KeyError(f"Missing parameter {key} for hyper model")
        return parameters

    
    # -----------------------------------------------------------------
    # Compute population weights
    # -----------------------------------------------------------------
    def _compute_weights(self, parameters):
        r"""
        Compute the importance-sampling weights that re-weight the
        proposal samples to the target population :math:`\Lambda`:
 
        .. math::
 
            w_k = \frac{\pi(\theta_k | \Lambda)}{\pi_{\rm draw}(\theta_k)}
                  \left(\frac{d_{L,\rm fid}}{d_{L}(\Lambda)}\right)^2
 
        The :math:`d_L` ratio accounts for the fact that the wave energy
        scales as :math:`|h|^2 \propto d_L^{-2}`, so when the cosmology
        changes the pre-computed energies must be corrected.
 
        Parameters
        ----------
        parameters: dict
            The current hyper-parameter values.
 
        Returns
        -------
        weights: array-like, shape ``(N_samples,)``
        """
        parameters, _ = self.conversion_function(parameters)

        mass_function_parameters = self._get_function_parameters(self.mass_function_model, **parameters)
        pop_m1_q_prob = self.mass_function_model(self.mass_q_samples, **mass_function_parameters)

        redshift_parameters = self._get_function_parameters(self.redshift_model, **parameters)
        Om0 = parameters["Om0"] if "Om0" in parameters else self._default_Om0
        w0 = parameters["w0"] if "w0" in parameters else self._default_w0
        pop_z_prob = (
            self.redshift_model.psi_of_z(self.zs_drawn, **redshift_parameters)
            / (1.0 + self.zs_drawn)
            * _inv_efunc_flat_wcdm(self.zs_drawn, Om0, w0)
        )
        
        weights = pop_m1_q_prob * pop_z_prob / self.p_m1_q_z_drawn
 
        return weights
 
    # -----------------------------------------------------------------
    # Compute Omega_GW
    # -----------------------------------------------------------------
    def _compute_omega_gw(self, parameters):
        r"""
        Compute the predicted :math:`\Omega_{\rm GW}(f | \Lambda)`.
 
        Parameters
        ----------
        parameters: dict
            Current hyper-parameter values (must include ``rate``).
 
        Returns
        -------
        omega: array-like, shape ``(N_freq,)``
        """

        weights = self._compute_weights(parameters)
 
        Rate_norm = parameters["rate"]

        # Extract H0 for the Omega_GW prefactor.
        # When doing cosmological inference H0 is a sampled parameter
        # and must propagate into f^3 / H0^2.
        H0 = parameters["H0"] if "H0" in parameters else self._default_H0

        Omega_spectrum = _RhoC * Rate_norm * self.dEdf_freqs * xp.mean(self.dEdfs * weights[:, None], axis=0) / H0**3 / (365 * 24 * 3600) / 1e9
 
        return xp.interp(self.frequencies, self.dEdf_freqs, Omega_spectrum)

    # -----------------------------------------------------------------
    # Gaussian log-likelihood
    # -----------------------------------------------------------------
    def log_likelihood(self, parameters):
        r"""
        Evaluate the Gaussian log-likelihood:
 
        .. math::
 
            \ln \mathcal{L} = -\frac{1}{2}\sum_k
            \frac{\bigl(\hat{C}_{IJ}(f_k) -
            \Omega_{\rm GW}(f_k|\Lambda)\bigr)^2}{\sigma_k^2}
 
        Parameters
        ----------
        parameters: dict
 
        Returns
        -------
        float
        """
        omega = self._compute_omega_gw(parameters)
        residual = self.CIJ - omega
        return to_number(
            -0.5 * xp.sum(residual ** 2 / self.sigma ** 2), float
        )
 
    # -----------------------------------------------------------------
    # Noise (null) log-likelihood
    # -----------------------------------------------------------------
    def noise_log_likelihood(self):
        r"""
        Log-likelihood under the null hypothesis
        :math:`\Omega_{\rm GW} = 0`:
 
        .. math::
 
            \ln \mathcal{L}_0 = -\frac{1}{2}\sum_k
            \frac{\hat{C}_{IJ}(f_k)^2}{\sigma_k^2}
 
        Returns
        -------
        float
        """
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = to_number(
                -0.5 * xp.sum(self.CIJ ** 2 / self.sigma ** 2), float
            )
        return self._noise_log_likelihood
 
    # -----------------------------------------------------------------
    # Log-likelihood ratio
    # -----------------------------------------------------------------
    def log_likelihood_ratio(self, parameters):
        """
        Returns ``log_likelihood - noise_log_likelihood``.
        """
        return self.log_likelihood(parameters) - self.noise_log_likelihood()


class JointCBCStochasticLikelihood(Likelihood):
    r"""
    Joint likelihood combining resolved CBC events and the stochastic
    gravitational-wave background.
 
    Both components share the same population hyper-parameters
    :math:`\Lambda` (mass, spin, redshift models) and merger rate
    :math:`R_0`.  The joint log-likelihood is
 
    .. math::
 
        \ln \mathcal{L}_{\rm joint}
        = \ln \mathcal{L}_{\rm CBC}(\{d_i\}|\Lambda)
        + \ln \mathcal{L}_{\rm SGWB}(\hat{C}_{IJ}|\Lambda)
 
    where the CBC term is any of the existing
    :class:`HyperparameterLikelihood`, :class:`RateLikelihood`, or
    :class:`LocalMergerRateLikelihood` likelihoods, and the SGWB term
    is a :class:`Stochastic_Likelihood`.
    """
 
    def __init__(
        self,
        cbc_likelihood,
        sgwb_likelihood,
    ):
        """
        Parameters
        ----------
        cbc_likelihood: HyperparameterLikelihood (or subclass)
            The resolved-event population likelihood.  This is
            typically a :class:`RateLikelihood` when the rate is
            sampled directly.
        sgwb_likelihood: Stochastic_Likelihood
            The SGWB likelihood.
        """
        self.cbc_likelihood = cbc_likelihood
        self.sgwb_likelihood = sgwb_likelihood
        super().__init__()
 
    def log_likelihood_ratio(self, parameters):
        r"""
        Sum of the CBC and SGWB log-likelihood ratios:
 
        .. math::
 
            \ln \hat{\mathcal{L}}_{\rm joint}
            = \ln \hat{\mathcal{L}}_{\rm CBC}
            + \ln \hat{\mathcal{L}}_{\rm SGWB}
 
        Parameters
        ----------
        parameters: dict
 
        Returns
        -------
        float
        """
        ln_l_cbc = self.cbc_likelihood.log_likelihood_ratio(parameters)
        ln_l_sgwb = self.sgwb_likelihood.log_likelihood_ratio(parameters)
        return ln_l_cbc + ln_l_sgwb
 
    def noise_log_likelihood(self):
        """
        Sum of the CBC and SGWB noise log-likelihoods.
        """
        return (
            self.cbc_likelihood.noise_log_likelihood()
            + self.sgwb_likelihood.noise_log_likelihood()
        )
 
    def log_likelihood(self, parameters):
        """
        Full joint log-likelihood.
        """
        return self.noise_log_likelihood() + self.log_likelihood_ratio(parameters)



class NullHyperparameterLikelihood(HyperparameterLikelihood):
    """
    A likelihood that can be used to sample the prior space subject to the
    maximum_uncertainty constraint imposed on the variance.

    For the uncertainty calculation see the Appendix of
    `Golomb and Talbot <https://arxiv.org/abs/2106.15745>`_ and
    `Farr <https://arxiv.org/abs/1904.10879>`_.
    """

    def ln_likelihood_and_variance(self, parameters=None):
        """
        Compute the ln likelihood estimator and its variance.
        """
        _, variance = super().ln_likelihood_and_variance(parameters)
        return 0.0, variance
