import functools
import numpy as np

from scipy.special import erf, erfc, voigt_profile
from scipy.optimize import fsolve

from astropy import constants, units

from dust_extinction.parameter_averages import CCM89, G03_SMCBar

fwhm2sig  = np.sqrt(np.log(256.))


def _g03_(wave, Av):
    """Return attenuation (linear) at `wave` for a CCM89 law with `Av` and optional `Rv`"""
    #wave = np.clip(wave, 0.1, 3.3333)
    return np.exp(-G03_SMCBar()(1./wave/units.um) * Av / (2.5*np.log10(np.e)))

def gauss_int2(x, mu=0, sig=1, flux=1.):
    xsi = (x-mu)/np.sqrt(2.)/sig # Normalised units!
    dx = np.diff(np.atleast_1d(xsi))
    xsi = np.hstack((xsi, xsi[-1]))
    dx = np.hstack((dx[0], dx, -dx[-1]))/2.
    xsi -= dx
    dx[-1] *= -1.
    dx = dx[1:]+dx[:-1]
    integr = np.diff(erf(xsi)/2.)/dx/np.sqrt(2.)/sig
    return flux * integr

def log_erfc(x):
    return np.where(x<20,
        np.log(erfc(x)),
        -x**2 - 0.5*np.log(np.pi) + np.log(1./x - 0.5/x**3 + 0.75/x**5)
        )

def log_erfc_prior(x, mean=0., scale=1.):
    erfc_arg = (x-mean)/(np.sqrt(2.0) * scale)
    erfc_arg0 = mean/(np.sqrt(2.0) * scale)
    log_normalization = np.log(                                             
        mean*erfc(-erfc_arg0)                                               
        + np.sqrt(2./np.pi)*scale*np.exp(-erfc_arg0**2))                       
    return log_erfc(erfc_arg) - log_normalization                       



def target_func(x, fwhm100b1, fwhm100b2, frac1b):
    """Find fwhm of a the sum of two Gaussian functions."""
    fact = (1-frac1b)/frac1b
    sigrat = fwhm100b1/fwhm100b2
    return (
        np.exp(-0.5*(x*fwhm2sig/fwhm100b1)**2) +
        fact * sigrat * np.exp(-0.5*(x*fwhm2sig/fwhm100b2)**2)
        - 0.5 - 0.5 * fact * sigrat)

def get_fwhm(fwhm100b1, fwhm100b2, frac1b, guess=10):
    """Find fwhm of a the sum of two Gaussian functions."""
    fwhm = 2.*fsolve(target_func, args=(fwhm100b1, fwhm100b2, frac1b), x0=guess)[0]
    return np.abs(fwhm)

def mask_line(wave, wave_c, delta, z, avoid=True):
    if wave_c.unit==units.AA:
        wave_c = (wave_c*(1+z)).to('um')
    wave_c = wave_c.value

    if delta.unit==units.Unit('km/s'):
        delta = (delta/constants.c*wave_c)

    if avoid:
        return np.abs(wave - wave_c)>delta
    else:
        return np.abs(wave - wave_c)>delta

def get_sigma_l(w, f, cont, mask):
     line = (f-cont)[mask] 
     wave = w[mask]
     dw = np.gradient(wave)
     p = np.nansum(line*dw)
     w0 = np.nansum(wave*line*dw)/p
     sig_l = np.nansum((wave-w0)**2 * line * dw)/p
     return np.sqrt(sig_l)




@functools.cache
def _g03_Av1_(wave):
    """Return attenuation (linear) at `wave` for a CCM89 law with `Av` and optional `Rv    `"""
    return np.exp(-G03_SMCBar()(1./wave/units.um) / (2.5*np.log10(np.e)))
