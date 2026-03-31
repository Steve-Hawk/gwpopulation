import numpy as np
import astropy.constants


_MPC_IN_METRES = 3.0856775814913673e22
km_Mpc = 1e3 / _MPC_IN_METRES
c = 2.998e8
G = 6.67e-11
_RhoC = 8.*np.pi*G/(3*c**2)/(km_Mpc**3)


# adopt from icarogw package
def v(Mtot,f):
    '''
    This function computes the v factor (specify what it is)
    
    Parameters
    ----------
    Mtot: float
        Total mass of the binay in solar masses
    f: array
        Frequency array
    '''
    MsunToSec = astropy.constants.M_sun.value*astropy.constants.G.value/np.power(astropy.constants.c.value,3.)
    return np.array([(np.pi*MsunToSec*f*Mtot)**(1./3.), (np.pi*MsunToSec*f*Mtot)**(2./3.), (np.pi*MsunToSec*f*Mtot)])

def dEdf(Mtot,freqs,eta=0.25,inspiralOnly=False,PN=True,chi=None):

    """
    Function to compute the energy spectrum radiated by a CBC. Taken from (https://ui.adsabs.harvard.edu/abs/2023arXiv231017625T/abstract)
    
    INPUTS
    Mtot: Total mass in units of Msun
    freqs: Array of frequencies at which we want to evaluate dEdf
    eta: Reduced mass ratio. Defaults to 0.25 (equal mass)
    inspiralOnly: If True, will return only energy radiated through inspiral
    """

    Msun = astropy.constants.M_sun.value
    c= astropy.constants.c.value
    G= astropy.constants.G.value
    MsunToSec = astropy.constants.M_sun.value*astropy.constants.G.value/np.power(astropy.constants.c.value,3.)


    if chi is None:
        chi = 0.

    # Initialize energy density
    dEdf_spectrum = np.zeros_like(freqs)

    if inspiralOnly:

        # If inspiral only (used for BNS), cut off at the ISCO
        fMerge = 2.*c**3./(6.*np.sqrt(6.)*2.*np.pi*G*Mtot*Msun)
        inspiral = freqs<fMerge
        dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)

    else:

        if PN:

            # Waveform model from Ajith+ 2011 (10.1103/PhysRevLett.106.241101)

            # PN corrections to break frequencies bounding different waveform regimes
            # See Eq. 2 and Table 1
            eta_arr = np.array([eta,eta*eta,eta*eta*eta])
            chi_arr = np.array([1,chi,chi*chi]).T
            fM_corrections = np.array([[0.6437,0.827,-0.2706],[-0.05822,-3.935,0.],[-7.092,0.,0.]])
            fR_corrections = np.array([[0.1469,-0.1228,-0.02609],[-0.0249,0.1701,0.],[2.325,0.,0.]])
            fC_corrections = np.array([[-0.1331,-0.08172,0.1451],[-0.2714,0.1279,0.],[4.922,0.,0.]])
            sig_corrections = np.array([[-0.4098,-0.03523,0.1008],[1.829,-0.02017,0.],[-2.87,0.,0.]])

            # Define frequencies
            # See Eq. 2 and Table 1
            fMerge = (1. - 4.455*(1.-chi)**0.217 + 3.521*(1.-chi)**0.26 + eta_arr.dot(fM_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            fRing = (0.5 - 0.315*(1.-chi)**0.3 + eta_arr.dot(fR_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            fCut = (0.3236 + 0.04894*chi + 0.01346*chi*chi + eta_arr.dot(fC_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            sigma = (0.25*(1.-chi)**0.45 - 0.1575*(1.-chi)**0.75 + eta_arr.dot(sig_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)

            # Identify piecewise components
            inspiral = freqs<fMerge
            merger = (freqs>=fMerge)*(freqs<fRing)
            ringdown = (freqs>=fRing)*(freqs<fCut)

            # Define PN amplitude corrections
            # See Eq. 1 and following text
            alpha = np.array([0., -323./224. + 451.*eta/168., (27./8.-11.*eta/6.)*chi])
            eps = np.array([1.4547*chi-1.8897, -1.8153*chi+1.6557, 0.])
            vs = v(Mtot,freqs)

            # Compute multiplicative scale factors to enforce continuity of dEdf across boundaries
            # Note that w_m and w_r are the ratios (inspiral/merger) and (merger/ringdown), as defined below
            v_m = v(Mtot,fMerge)
            v_r = v(Mtot,fRing)
            w_m = np.power(fMerge,-1./3.)*np.power(1.+alpha.dot(v_m),2.)/(np.power(fMerge,2./3.)*np.power(1.+eps.dot(v_m),2.)/fMerge)
            w_r = (w_m*np.power(fRing,2./3.)*np.power(1.+eps.dot(v_r),2.)/fMerge)/(np.square(fRing)/(fMerge*fRing**(4./3.)))

            # Energy spectrum --> https://arxiv.org/abs/2306.09861
            dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)*np.power(1.+alpha.dot(vs[:,inspiral]),2.)
            dEdf_spectrum[merger] = w_m*np.power(freqs[merger],2./3.)*np.power(1.+eps.dot(vs[:,merger]),2.)/fMerge
            dEdf_spectrum[ringdown] = w_r*np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))

        else:

            # Waveform model from Ajith+ 2008 (10.1103/PhysRevD.77.104017)
            # Define IMR parameters
            # See Eq. 4.19 and Table 1
            fMerge = (0.29740*eta**2. + 0.044810*eta + 0.095560)/(np.pi*Mtot*MsunToSec)
            fRing = (0.59411*eta**2. + 0.089794*eta + 0.19111)/(np.pi*Mtot*MsunToSec)
            fCut = (0.84845*eta**2. + 0.12828*eta + 0.27299)/(np.pi*Mtot*MsunToSec)
            sigma = (0.50801*eta**2. + 0.077515*eta + 0.022369)/(np.pi*Mtot*MsunToSec)

            # Identify piecewise components
            inspiral = freqs<fMerge
            merger = (freqs>=fMerge)*(freqs<fRing)
            ringdown = (freqs>=fRing)*(freqs<fCut)

            # Energy spectrum
            dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)
            dEdf_spectrum[merger] = np.power(freqs[merger],2./3.)/fMerge
            dEdf_spectrum[ringdown] = np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))


    # Normalization
    Mc = np.power(eta,3./5.)*Mtot*Msun
    amp = np.power(G*np.pi,2./3.)*np.power(Mc,5./3.)/3.
    return amp*dEdf_spectrum