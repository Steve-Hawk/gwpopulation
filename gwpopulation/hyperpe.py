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
from bilby.core.utils import logger
from bilby.hyper.model import Model

from .utils import get_name, to_number, to_numpy
from .models.redshift import _Redshift
from .experimental.sgwb_utils import wave_energy, omega_gw

xp = np

__all__ = ["HyperparameterLikelihood", "RateLikelihood", "LocalMergerRateLikelihood", "Stochastic_Likelihood", "JointCBCSGWBLikelihood", "xp"]


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


def _compute_single_wave_energy(
    inj_sample, waveform_generator, target_frequencies):
    """
    Module-level helper that computes the wave energy spectrum for a
    single injection sample.  Kept at module level so that it is
    pickle-able for :mod:`multiprocessing`.
 
    Missing waveform parameters (``phase``, ``theta_jn``, spins) are
    filled with isotropic/zero defaults.
 
    Parameters
    ----------
    inj_sample: dict
        Single-event parameter dictionary.
    waveform_generator: bilby.gw.WaveformGenerator
        The waveform generator instance.
    target_frequencies: np.ndarray
        The frequency grid to interpolate onto.
    use_approxed_waveform: bool
        Whether to use the piecewise-closed-form amplitude approximation.
    inspiral_only: bool
        If ``use_approxed_waveform`` is True, whether to truncate at ISCO.
 
    Returns
    -------
    np.ndarray
        The wave energy spectrum interpolated onto *target_frequencies*.
    """
    # Fill defaults for orientation / spin parameters that may be absent.
    if "phase" not in inj_sample:
        inj_sample["phase"] = 2 * np.pi * np.random.rand()
    if "theta_jn" not in inj_sample:
        inj_sample["theta_jn"] = np.arccos(np.random.rand() * 2.0 - 1.0)
    # we don't expect the spins to have a significant effect on the SGWB, so we can just set them to zero in most cases.
    for key in ("a_1", "a_2", "tilt_1", "tilt_2"):
        if key not in inj_sample:
            inj_sample[key] = 0
 
    waveform_frequencies = waveform_generator.frequency_array
    wave_en = wave_energy(
        waveform_generator,
        inj_sample
    )
    return np.interp(target_frequencies, waveform_frequencies, wave_en)


def _mp_worker(args):
    """Thin wrapper so that :func:`multiprocessing.Pool.map` can call
    :func:`_compute_single_wave_energy` with a single tuple argument."""
    return _compute_single_wave_energy(*args)


class Stochastic_Likelihood(Likelihood):
    """
    A likelihood for inferring hyperparameter posterior distributions
    and estimating the stochastic gravitational-wave background (SGWB) with existing GWPopulation models.

    see the prepared doc.

    """
    def __init__(
        self,
        samples, stochastic_data,
        hyper_prior, # notice that we have to clearly specify the zmax of Redshift model in the hyper_prior.
        wave_energies=None,
        waveform_approximant="IMRPhenomD",
        waveform_duration=10,
        sampling_frequency=4096,
        waveform_reference_frequency=25,
        waveform_minimum_frequency=10,
        multiprocess=True,
        conversion_function=lambda args: (args, None),
    ):
        """
        Parameters
        ----------
        samples: dict
            Must contain at least ``mass_1``, ``mass_ratio``,``redshift`` and
            ``luminosity_distance`` (removed later).  May contain a ``prior`` column
            with the original proposal-prior values per sample.
        stochastic_data: dict
            Measured SGWB data.  Must contain:
 
            - ``CIJ``: the cross-correlation estimator :math:`\\hat{C}_{IJ}(f)`
            - ``sigma``: the associated :math:`1\\sigma` uncertainty per frequency bin
            - ``frequencies``: the frequency array

        hyper_prior: `bilby.hyper.model.Model` or callable
            The population model (mass, spin, redshift, …).
        wave_energies: array-like or None, optional
            Pre-computed gravitational-wave energy spectra of shape
            ``(N_samples, N_frequencies)``.  If *None* (the default) the
            energies are computed automatically from ``samples`` using
            the waveform configuration given below.
        waveform_approximant: str, optional
            LAL waveform approximant string,
            Default ``"IMRPhenomD"``.
        waveform_duration: float, optional
            Duration in seconds for the waveform generator.  Default 10.
        sampling_frequency: float, optional
            Sampling frequency in Hz.  Default 4096.
        waveform_reference_frequency: float, optional
            Reference frequency in Hz.  Default 25.
        waveform_minimum_frequency: float, optional
            Minimum frequency in Hz.  Default 10.
        multiprocess: bool, optional
            Whether to use :mod:`multiprocessing` when computing wave
            energies.  Default ``True``.
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
        self.samples = samples
        #todo create another attribute to save redshift from samples input.
        #! check it later.
        self.samples_redshift = self.samples["redshift"]
        # after accesing luminosity information, that column should be removed from samples.
        self.samples_default_luminosity_distance = self.samples.pop("luminosity_distance")

        # real measurements of the stochastic search.
        self.CIJ = stochastic_data["CIJ"]
        self.sigma = stochastic_data["sigma"]
        self.frequencies = stochastic_data["frequencies"]

        if "prior" in self.samples:
            self.sampling_prior = self.samples.pop("prior")
        else:
            logger.info("No prior values provided, defaulting to 1.")
            self.sampling_prior = 1
        
        if wave_energies is not None:
            self.wave_energies = xp.asarray(wave_energies)
        else:
            logger.info(
                f"Computing wave energies for {self.n_samples} samples "
                f"using waveform_approximant={waveform_approximant!r} "
                f"(multiprocess={multiprocess})…"
            )
            self.wave_energies = self._calculate_wave_energies(
                waveform_approximant=waveform_approximant,
                waveform_duration=waveform_duration,
                sampling_frequency=sampling_frequency,
                waveform_reference_frequency=waveform_reference_frequency,
                waveform_minimum_frequency=waveform_minimum_frequency,
                multiprocess=multiprocess,
            )
        
        super().__init__()

        # used for cosmological inference.
        self.cosmology_model = self._find_cosmo_model()
        self.redshift_model = self._find_redshift_model()
        self._noise_log_likelihood = None
    
    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------
    @property
    def n_samples(self):                                               
        """Number of proposal samples."""
        key = next(iter(self.samples))
        return len(self.samples[key])

    # -----------------------------------------------------------------
    # Wave-energy computation
    # -----------------------------------------------------------------
    def _calculate_wave_energies(
        self,
        waveform_approximant,
        waveform_duration,
        sampling_frequency,
        waveform_reference_frequency,
        waveform_minimum_frequency,
        multiprocess,
    ):
        """
        Compute :math:`|\\tilde{h}(f)|^2` for every proposal sample and
        interpolate onto :attr:`self.frequencies`.
 
        Parameters
        ----------
        waveform_approximant: str
        waveform_duration: float
        sampling_frequency: float
        waveform_reference_frequency: float
        waveform_minimum_frequency: float
        multiprocess: bool
 
        Returns
        -------
        wave_energies: np.ndarray, shape ``(N_samples, N_freq)``
        """
        import multiprocessing as mp
 
        import bilby
 
        lal_approximant = waveform_approximant
 
        # Build the bilby waveform generator (only needed for non-PC waveforms,
        # we have redshift in our parameters!
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=waveform_duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant=lal_approximant,
                reference_frequency=waveform_reference_frequency,
                minimum_frequency=waveform_minimum_frequency,
            ),
        )
 
        target_frequencies = np.asarray(self.frequencies)
 
        # Build a list of single-event dictionaries and fill in required key-value pairs
        inj_samples = self._build_injection_list()
 
        if multiprocess:
            logger.info(
                "Using multiprocessing to compute wave energies "
                "(no progress bar)…"
            )
            args_list = [
                (s, waveform_generator, target_frequencies)
                for s in inj_samples
            ]
            with mp.Pool() as pool:
                result = pool.map(_mp_worker, args_list)
        else:
            from tqdm.auto import tqdm
 
            result = [
                _compute_single_wave_energy(
                    s, waveform_generator, target_frequencies)
                for s in tqdm(inj_samples, desc="Computing wave energies")
            ]
        return xp.asarray(result)

    def _build_injection_list(self):
        """
        Convert the columnar ``self.samples`` dict into a list of
        single-event dictionaries, one per proposal sample.
 
        Returns
        -------
        list[dict]
        """
        build_samples = self.samples.copy()
        build_samples['mass_1_source'] = build_samples.pop('mass_1')
        build_samples['mass_1_detector'] = build_samples['mass_1_source'] * (1. + build_samples['redshift'])
        build_samples['mass_2_detector'] = build_samples['mass_1_detector'] * build_samples['mass_ratio']

        # build_samples is a dict contains `mass_1_source`, `mass_ratio`, `mass_1_detector`, `mass_2_detector`, `redshift`

        keys = list(build_samples.keys())
        n = self.n_samples
        inj_samples = []
        for i in range(n):
            inj_samples.append(
                {key: float(build_samples[key][i]) for key in keys}
            )
        
        return inj_samples


    def _find_cosmo_model(self):
        """Find model in hyper_prior that has cosmology method."""
        for model in self.hyper_prior.models:
            if hasattr(model, "cosmology"):
                return model
            else:
                logger.info("No cosmology model found in hyper_prior, using default cosmology.")
        return None

    def _find_redshift_model(self):
        """Find the redshift model in hyper_prior."""
        for model in self.hyper_prior.models:
            if isinstance(model, _Redshift):
                return model
            else:
                logger.info("No Redshift model found in hyper_prior, probably needs further check.")
        return None
    
    def get_luminosity_distance_from_cosmology_model(self, parameters):
        """Get cosmology model from hyper_prior."""
        if self.cosmology_model is not None:
            return self.cosmology_model.cosmology(parameters).luminosity_distance(self.samples_redshift)
        else:
            return self.samples_default_luminosity_distance
    
    # -----------------------------------------------------------------
    # Extract H0 for the Omega_GW prefactor
    # -----------------------------------------------------------------
    def _get_H0(self, parameters):
        r"""
        Return the current Hubble constant in km/s/Mpc, or *None* if the
        cosmology is fixed (in which case :func:`omega_gw` will use its
        built-in Planck15 default).
 
        For variable-cosmology runs (``FlatLambdaCDM``, ``FlatwCDM``)
        the sampled ``H0`` is read directly from *parameters*.
        For fixed-cosmology runs with a cosmology model present the
        value is read from the model's cosmology object.
        If no cosmology model exists at all, *None* is returned.
 
        Parameters
        ----------
        parameters: dict
 
        Returns
        -------
        H0: float or None
            Hubble constant in km/s/Mpc, or None for the default.
        """
        # 1) Variable cosmology: H0 is a sampled parameter.
        if "H0" in parameters:
            return float(parameters["H0"])
 
        # 2) Fixed cosmology model present: read H0 from the model.
        if self.cosmology_model is not None:
            cosmo = self.cosmology_model.cosmology(parameters)
            # wcosmo with units disabled returns a bare float (km/s/Mpc);
            # with units enabled it returns an astropy Quantity.
            H0_val = cosmo.H0
            return float(H0_val.value) if hasattr(H0_val, "value") else float(H0_val)
 
        # 3) No cosmology information at all — use the default.
        return None


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
        pop_prob = self.hyper_prior.prob(self.samples, **parameters)
        weights = pop_prob / self.sampling_prior
 
        # Rescale for variable cosmology (dL_fid / dL_new)^2.
        dL_new = self.get_luminosity_distance_from_cosmology_model(parameters)
        dL_ratio = self.samples_default_luminosity_distance / dL_new
        weights = weights * dL_ratio ** 2
 
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
 
        # Rate normalisation:
        #   R0 * int dz (dV/dz) (1+z)^{-1} psi(z) / 1e9
        # The factor 1e9 converts the volume from Mpc^3 to Gpc^3.
        if self.redshift_model is not None:
            volume_norm = self.redshift_model.normalisation(parameters)
        else:
            volume_norm = 1.0
        Rate_norm = parameters["rate"] * volume_norm / 1e9  # R0 - Gpc^{-3} yr^{-1}

        # Extract H0 for the Omega_GW prefactor.
        # When doing cosmological inference H0 is a sampled parameter
        # and must propagate into f^3 / H0^2.
        H0 = self._get_H0(parameters)
 
        return omega_gw(self.frequencies, self.wave_energies, weights, Rate_norm, H0=H0) / (365 * 24 * 3600)  # convert from per year to per second

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


class JointCBCSGWBLikelihood(Likelihood):
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
