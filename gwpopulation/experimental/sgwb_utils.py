# -*- coding: utf-8 -*-
# Copyright (C) Arianna I. Renzini 2024 (original author of popstock)
# Xiao-Xiao Kou made changes to make it work with GWPopulation for a joint inference of the population and the SGWB.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# This file is part of the popstock package.

import bilby
from bilby.core.utils import logger
import numpy as np

try:
    from wcosmo.astropy import Planck15 as _default_cosmo
    _DEFAULT_H0_SI = _default_cosmo.H0.to("1/s").value  # H0 in 1/s
except Exception:
    # fallback: H0 = 67.74 km/s/Mpc in SI (1/s)
    _DEFAULT_H0_SI = 67.74 * 1e3 / (3.0856775814913673e22)  # km/s/Mpc -> 1/s
 
# 1 Mpc in metres — used to convert H0 [km/s/Mpc] → H0 [1/s]
_MPC_IN_METRES = 3.0856775814913673e22
 
 
from astropy.constants import c
light_speed = c.value
m_sun = 1.99e30 #kg
G = 6.67e-11 #N*m^2/kg^2
mass_to_seconds_conv = G/light_speed**3

def omega_gw(frequencies, wave_energies, weights, Rate_norm, H0=None):
    """
    Compute Omega GW spectrum given a set of wave energy spectra and associated weights.
 
    Parameters
    =======
    frequencies: np.array
        Frequency array associated to the wave energy.
    wave_energies: np.array
        Array of wave energy spectra.
    weights: np.array
        Array of weights per sample: sampling weights * rescale of luminosity distance.
    Rate_norm: float
        It should be R0 * \\int dz (dV/dz) * (1+z)^{-1} * p(z) where R0 is the local merger rate and p(z) is the redshift distribution of mergers.
    H0: float or None, optional
        Hubble constant in km/s/Mpc.  If *None* (default) the Planck15
        value is used.  Pass this when performing cosmological inference
        so that the :math:`f^3 / H_0^2` prefactor tracks the sampled
        cosmology.
 
    Returns
    =======
    The Omega_GW spectrum in a np.array.
    """
    if H0 is not None:
        # Convert H0 from km/s/Mpc to 1/s
        H0_si = H0 * 1e3 / _MPC_IN_METRES
    else:
        H0_si = _DEFAULT_H0_SI
    
    conv = frequencies**3 * 4. * np.pi**2 / (3 * H0_si**2)
    # might consider adding some checks for re-weighting, like:
    # highvals = np.sort(weights)[-10:]
    # weights[weights==highvals]=0
    N_samples = len(weights)
    weighted_energy = np.nansum(weights[:, None] * wave_energies, axis=0) / N_samples
 
    return Rate_norm * conv * weighted_energy
