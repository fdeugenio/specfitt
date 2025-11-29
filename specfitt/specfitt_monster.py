import functools
import os
import pickle
import re
import traceback
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, least_squares
from scipy.signal import convolve

from astropy import constants, cosmology, table, units
cosmop = cosmology.Planck18

import emcee
import corner

from dust_extinction.parameter_averages import CCM89, G03_SMCBar

from . import spectrum 
from .specfitt_utils import gauss_int2, _g03_, voigt_profile, log_erfc_prior, get_fwhm
from . import specfitt

__all__ = ['jwst_spec_models', 'jwst_spec_fitter']

class jwst_spec_models(spectrum.jwst_spec):

    # +---+-------------------------------------------------------------------+
    # | 4.| H\\alpha and [NII] models.                                        |
    # +---+-------------------------------------------------------------------+
    def model_double_blr_ha_only(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha narrow and double BLR; absorber."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHe16678': (r'$F_\mathrm{n}(\mathrm{He\,I\lambda 6678})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_f$', 1., '[---]'),
                'tauHa': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'bk00': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float)
                }
        (z_n, sig_n_100, fHa, fHe16678,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHa_b, frac1b,
         v_abs_100, sig_abs_100, C_f, tau0ha,
         a0, b0) = pars
        w_mum = np.array((self.Halpha, self.HeI6678,
            self.Halpha, self.Halpha, self.Halpha))
        w_mum = (w_mum * (1.+z_n)
            * np.array((1., 1.,)
                  + (np.exp(v_blr_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_100/self.c_100_kms),))
            )
        if print_waves: return w_mum

        fHa_b1 = fHa_b * frac1b
        fHa_b2 = fHa_b * (1.-frac1b)
        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100, sig_n_100)
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)
            + (sig_abs_100,))
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHa)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fHe16678)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fHa_b1)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fHa_b2)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[4]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=tau0_norm)
        absrha = 1. - C_f + C_f * np.exp(-tau_ha)
        bk0 = a0 + (self.wave-w_mum[0]) * b0 #+ (self.wave-w_mum[0])**2 * c0

        if print_blobs:
            mask_ha = self.fit_mask[0] & (np.abs(self.wave-w_mum[4])/sig_mum[4]<10)
            dwha = np.gradient(self.wave[mask_ha])
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewha /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame

            return ewha, fHa, fHa_b1+fHa_b2
        return f0, f1, f2*absrha, f3*absrha, bk0

    def model_double_blr_ha_only_fit_init(self):
        """Halpha narrow and double BLR; absorber."""
        lwave = np.array((.6565,))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.125,)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],]

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        guess = np.array((
            self.redshift_guess, 0.5, 0.01, 0.01,
            0., 15., 30., 1.5, .5,
            0., 1.20, 0.9, 3.,
            bkg00, bkg01, 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0.,
             -5., 10., 20., 0.001, 0.,
             -5., 0., 0.0, 0.1,
             bkg00-10*bkg01, -100., -100.),
            (self.redshift_guess+self.dz, 5., 2., 2.,
              5., 20., 70., 10., 1.,
              5., 5., 1.0, 30.,
             bkg00+10*bkg01, 100., 100.)))
        #mask = masks[0] | masks[1] | masks[2]
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, fHa, fHe16678,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHa_b, frac1b,
             v_abs_100, sig_abs_100, C_f, tau0ha,
             a0, b0) = pars

            lnprior = 0.
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.1)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            return lnprior

        return masks, guess, bounds, lnprior



    def model_double_blr_ha_linfit(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha narrow and double BLR; absorber."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fO16300':  (r'$F_\mathrm{n}(\mathrm{[O\,I]\lambda 6300})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHe16678': (r'$F_\mathrm{n}(\mathrm{He\,I\lambda 6678})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_f$', 1., '[---]'),
                'tauHa': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'bk00': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float)
                }
        (z_n, sig_n_100, fO16300, fHa, fHe16678,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHa_b, frac1b,
         v_abs_100, sig_abs_100, C_f, tau0ha,
         a0, b0) = pars
        w_mum = np.array((self.OI6300, self.OI6363, self.Halpha, self.HeI6678,
            self.Halpha, self.Halpha, self.Halpha))
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*4
                  + (np.exp(v_blr_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_100/self.c_100_kms),))
            )
        if print_waves: return w_mum

        fO16363 = fO16300 / 3.131
        fHa_b1 = fHa_b * frac1b
        fHa_b2 = fHa_b * (1.-frac1b)
        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*4
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)
            + (sig_abs_100,))
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fO16300)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO16363)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fHa)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fHe16678)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa_b1)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fHa_b2)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[6]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=tau0_norm)
        absrha = 1. - C_f + C_f * np.exp(-tau_ha)
        bk0 = a0 + (self.wave-w_mum[2]) * b0 #+ (self.wave-w_mum[2])**2 * c0

        if print_blobs:
            mask_ha = self.fit_mask[0] & (np.abs(self.wave-w_mum[6])/sig_mum[6]<10)
            dwha = np.gradient(self.wave[mask_ha])
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewha /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame

            return ewha, fHa, fHa_b1+fHa_b2
        return f0, f1, f2, f3, f4*absrha, f5*absrha, bk0

    def model_double_blr_ha_linfit_fit_init(self):
        """Halpha narrow and double BLR; absorber."""
        lwave = np.array((.6500,))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.240,)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],]

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        guess = np.array((
            self.redshift_guess, 0.5, 0.01, 0.01, 0.01,
            0., 15., 30., 1.5, .5,
            0., 1.20, 0.9, 3.,
            bkg00, bkg01,))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0., 0.,
             -5., 10., 20., 0.001, 0.,
             -5., 0., 0.0, 0.1,
             bkg00-10*bkg01, -100.),
            (self.redshift_guess+self.dz, 5., 2., 2., 2.,
              5., 20., 70., 10., 1.,
              5., 5., 1.0, 30.,
             bkg00+10*bkg01, 100.,)))
        #mask = masks[0] | masks[1] | masks[2]
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, fO16300, fHa, fHe16678,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHa_b, frac1b,
             v_abs_100, sig_abs_100, C_f, tau0ha,
             a0, b0) = pars

            lnprior = 0.
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.1)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2
            lnprior += - 0.5*((b0+(0.5))/0.1)**2

            return lnprior

        return masks, guess, bounds, lnprior



class jwst_spec_fitter(jwst_spec_models, specfitt.jwst_spec_fitter):
    pass
