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
from .specfitt_utils import gauss_int2, _g03_, voigt_profile, log_erfc_prior, get_fwhm, get_sigma_l
from . import specfitt

__all__ = ['jwst_spec_models', 'jwst_spec_fitter']

class jwst_spec_models(specfitt.jwst_spec_models):

    def model_double_blr_abs_untiev(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO35007_intr': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_intr': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583_intr': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_1_100': (r'$v_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'v_blr_2_100': (r'$v_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_Hb_100': (r'$v_\mathrm{abs,H\beta}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'v_abs_Ha_100': (r'$v_\mathrm{abs,H\alpha}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_{f}$', 1., '[---]'),
                'tauHb': (r'$\tau_\mathrm{H\beta}$', 1., '[---]'),
                'tauHa': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'bk00': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk10': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk11': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
            }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewhb'  : (r'$\mathrm{EW(H\beta)}$', 1., r'[\AA]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fN26583_obs': (r'$F(\mathrm{[O\,II]\lambda 6583})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'sig_l_100': (r'$\sigma_\mathrm{l,BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'log_L_Ha_b_2'   : (r'$\log\,L_\mathrm{b,2}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                'logMBH2'      : (r'$\log\,(M_{\bullet,2})$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH2'     : (r'$\lambda_\mathrm{Edd,2}$', 1.,
                            '[---]', float),
                'logMBHsigl'      : (r'$\log\,(M_\bullet,sig)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBHsigl'     : (r'$\lambda_\mathrm{Edd,sig}$', 1.,
                            '[---]', float),
                }

        (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
         v_blr_1_100, v_blr_2_100,
         fwhm_blr_1_100, fwhm_blr_2_100, fHbeta_blr, fHalpha_blr, frac1b,
         v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, C_f, tau0hb, tau0ha,
         a0, b0, a1, b1) = pars

        w_mum = np.array((
            self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hbeta, self.Hbeta, self.Halpha, self.Halpha,
            self.Hbeta, self.Halpha
            ))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*6
                  + (np.exp(v_blr_1_100/self.c_100_kms),)*2
                  + (np.exp(v_blr_2_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_Hb_100/self.c_100_kms),)
                  + (np.exp(v_abs_Ha_100/self.c_100_kms),)
                )
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 2.98
        fN26548 = fN26583 / 3.05
        fHb_b1 = fHbeta_blr * frac1b
        fHb_b2 = fHbeta_blr * (1.-frac1b)
        fHa_b1 = fHalpha_blr * frac1b
        fHa_b2 = fHalpha_blr * (1.-frac1b)
        (fHb_obs, fO34959_obs, fO35007_obs, fN26548_obs, fHa_obs, fN26583_obs,
         fHb_b1_obs, fHb_b2_obs, fHa_b1_obs, fHa_b2_obs, _, _) = np.array((
             fHb, fO34959, fO35007, fN26548, fHa, fN26583,
             fHb_b1, fHb_b2, fHa_b1, fHa_b2, 0., 0.))*atten * self.mu_lens

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*6
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)*2
            + (sig_abs_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHb_obs)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO34959_obs)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO35007_obs)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fN26548_obs)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa_obs)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fN26583_obs)

        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHb_b1_obs)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHb_b2_obs)
        f8 = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fHa_b1_obs)
        f9 = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fHa_b2_obs)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[10]
        tau_hb = tau0hb * gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[11]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=tau0_norm)
        absr = 1. - C_f + C_f * np.exp(-tau_hb - tau_ha)

        bk0 = a0 + (self.wave-w_mum[1]) * b0
        bk1 = a1 + (self.wave-w_mum[4]) * b1
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)

        if print_blobs:
            mask_hb = self.fit_mask[0] & (np.abs(self.wave-w_mum[10])/sig_mum[10]<10)
            mask_ha = self.fit_mask[1] & (np.abs(self.wave-w_mum[11])/sig_mum[11]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp(v_abs_Hb_100/self.c_100_kms) # Rest frame
            ewha /= (1+z_n)*np.exp(v_abs_Ha_100/self.c_100_kms) # Rest frame

            fwhm_blr_100 = get_fwhm(
                fwhm_blr_1_100, fwhm_blr_2_100, frac1b, guess=0.15)
            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)

            sig_l_um = get_sigma_l(self.wave, self.flux, bk1, self.fit_mask[1])
            sig_l_100 = sig_l_um / w_mum[4] * constants.c.to('1e2 km/s').value
    
            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Halpha_blr = fHalpha_blr * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            log_L_Ha_b = np.log10((L_Halpha_blr * lum_fact).to('1e42 erg/s').value)
            logMBH          = 6.6 + 0.47*log_L_Ha_b + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            log_L_Ha_b_2 = log_L_Ha_b + np.log10(1.-frac1b)
            logMBH2         = 6.6 + 0.47*log_L_Ha_b_2 + 2.06*np.log10(fwhm_blr_2_100/10.)
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH2*units.Msun
            lEddMBH2 = (edrat/lEdd).to(1).value

            logMBHsigl      = (0.683  + 7.413 + 0.554*log_L_Ha_b
                + 2.61 * (np.log10(sig_l_100)-1.5) )

            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBHsigl*units.Msun
            lEddMBHsigl = (edrat/lEdd).to(1).value

            return (
                ewhb, ewha,
                fHb_obs, fO35007_obs, fHa_obs, fN26583_obs,
                fHb_b1_obs+fHb_b2_obs, fHa_b1_obs+fHa_b2_obs,
                fwhm_blr_100, sig_l_100,
                logSFR_Ha, log_L_Ha_b, log_L_Ha_b_2,
                logMBH, lEddMBH,
                logMBH2, lEddMBH2,
                logMBHsigl, lEddMBHsigl
                )

        return (
            f0, f1, f2, f3, f4, f5, f6*absr, f7*absr, f8*absr, f9*absr,
            bk0*absr, bk1*absr,
            )

    def model_double_blr_abs_untiev_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.15, 0.20)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1]]

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])

        guess = np.array((
            self.redshift_guess, .2, 0.0, 0.01, 0.01, 0.01,
            0., 0., 15., 30., 0.15, 1.5, .5,
            0., 0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0.0, 0., 0., 0.,
             -5., -5., 10., 15., 0., 0.001, 0.001,
             -5., -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             ),
            (self.redshift_guess+self.dz, .7, 10., 1., 1., 1.,
             5., 5., 20., 70., 10., 30., 1.,
             5., 5., 5., 1.0, 30., 30.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             )))
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
             v_blr_1_100, v_blr_2_100,
             fwhm_blr_1_100, fwhm_blr_2_100, fHbeta_blr, fHalpha_blr, frac1b,
             v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, C_f, tau0hb, tau0ha,
             a0, b0, a1, b1) = pars

            lnprior = 0.
            #lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/.5)**2
            #lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/.5)**2
            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHbeta_blr/fHalpha_blr, mean=.4, scale=0.05)

            return lnprior

        return masks, guess, bounds, lnprior


    def model_double_blr_abs_tiev(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO35007_intr': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_intr': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583_intr': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_1_100': (r'$v_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'v_blr_2_100': (r'$v_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_Hb_100': (r'$v_\mathrm{abs,H\beta}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'v_abs_Ha_100': (r'$v_\mathrm{abs,H\alpha}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_{f}$', 1., '[---]'),
                'tauHb': (r'$\tau_\mathrm{H\beta}$', 1., '[---]'),
                'tauHa': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'bk00': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk10': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk11': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
            }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewhb'  : (r'$\mathrm{EW(H\beta)}$', 1., r'[\AA]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fN26583_obs': (r'$F(\mathrm{[O\,II]\lambda 6583})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'sig_l_100': (r'$\sigma_\mathrm{l,BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'log_L_Ha_b_2'   : (r'$\log\,L_\mathrm{b,2}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                'logMBH2'      : (r'$\log\,(M_{\bullet,2})$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH2'     : (r'$\lambda_\mathrm{Edd,2}$', 1.,
                            '[---]', float),
                'logMBHsigl'      : (r'$\log\,(M_\bullet,sig)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBHsigl'     : (r'$\lambda_\mathrm{Edd,sig}$', 1.,
                            '[---]', float),
                }

        (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
         v_blr_1_100, v_blr_2_100,
         fwhm_blr_1_100, fwhm_blr_2_100, fHbeta_blr, fHalpha_blr, frac1b,
         v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, C_f, tau0hb, tau0ha,
         a0, b0, a1, b1) = pars

        w_mum = np.array((
            self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hbeta, self.Hbeta, self.Halpha, self.Halpha,
            self.Hbeta, self.Halpha
            ))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*6
                  + (np.exp(v_blr_1_100/self.c_100_kms),)*2
                  + (np.exp(v_blr_2_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_Hb_100/self.c_100_kms),)
                  + (np.exp(v_abs_Ha_100/self.c_100_kms),)
                )
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 2.98
        fN26548 = fN26583 / 3.05
        fHb_b1 = fHbeta_blr * frac1b
        fHb_b2 = fHbeta_blr * (1.-frac1b)
        fHa_b1 = fHalpha_blr * frac1b
        fHa_b2 = fHalpha_blr * (1.-frac1b)
        (fHb_obs, fO34959_obs, fO35007_obs, fN26548_obs, fHa_obs, fN26583_obs,
         fHb_b1_obs, fHb_b2_obs, fHa_b1_obs, fHa_b2_obs, _, _) = np.array((
             fHb, fO34959, fO35007, fN26548, fHa, fN26583,
             fHb_b1, fHb_b2, fHa_b1, fHa_b2, 0., 0.))*atten * self.mu_lens

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*6
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)*2
            + (sig_abs_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHb_obs)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO34959_obs)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO35007_obs)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fN26548_obs)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa_obs)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fN26583_obs)

        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHb_b1_obs)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHb_b2_obs)
        f8 = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fHa_b1_obs)
        f9 = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fHa_b2_obs)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[10]
        tau_hb = tau0hb * gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[11]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=tau0_norm)
        absr = 1. - C_f + C_f * np.exp(-tau_hb - tau_ha)

        bk0 = a0 + (self.wave-w_mum[1]) * b0
        bk1 = a1 + (self.wave-w_mum[4]) * b1
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)

        if print_blobs:
            mask_hb = self.fit_mask[0] & (np.abs(self.wave-w_mum[10])/sig_mum[10]<10)
            mask_ha = self.fit_mask[1] & (np.abs(self.wave-w_mum[11])/sig_mum[11]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp(v_abs_Hb_100/self.c_100_kms) # Rest frame
            ewha /= (1+z_n)*np.exp(v_abs_Ha_100/self.c_100_kms) # Rest frame

            fwhm_blr_100 = get_fwhm(
                fwhm_blr_1_100, fwhm_blr_2_100, frac1b, guess=0.15)
            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)

            sig_l_um = get_sigma_l(self.wave, self.flux, bk1, self.fit_mask[1])
            sig_l_100 = sig_l_um / w_mum[4] * constants.c.to('1e2 km/s').value
    
            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Halpha_blr = fHalpha_blr * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            log_L_Ha_b = np.log10((L_Halpha_blr * lum_fact).to('1e42 erg/s').value)
            logMBH          = 6.6 + 0.47*log_L_Ha_b + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            log_L_Ha_b_2 = log_L_Ha_b + np.log10(1.-frac1b)
            logMBH2         = 6.6 + 0.47*log_L_Ha_b_2 + 2.06*np.log10(fwhm_blr_2_100/10.)
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH2*units.Msun
            lEddMBH2 = (edrat/lEdd).to(1).value

            logMBHsigl      = (0.683  + 7.413 + 0.554*log_L_Ha_b
                + 2.61 * (np.log10(sig_l_100)-1.5) )

            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBHsigl*units.Msun
            lEddMBHsigl = (edrat/lEdd).to(1).value

            return (
                ewhb, ewha,
                fHb_obs, fO35007_obs, fHa_obs, fN26583_obs,
                fHb_b1_obs+fHb_b2_obs, fHa_b1_obs+fHa_b2_obs,
                fwhm_blr_100, sig_l_100,
                logSFR_Ha, log_L_Ha_b, log_L_Ha_b_2,
                logMBH, lEddMBH,
                logMBH2, lEddMBH2,
                logMBHsigl, lEddMBHsigl
                )

        return (
            f0, f1, f2, f3, f4, f5, f6*absr, f7*absr, f8*absr, f9*absr,
            bk0*absr, bk1*absr,
            )

    def model_double_blr_abs_tiev_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.15, 0.20)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1]]

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])

        guess = np.array((
            self.redshift_guess, .25, 0.0, 0.01, 0.01, 0.01,
            0., 0., 15., 30., 0.15, 1.5, .5,
            0., 0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0.0, 0., 0., 0.,
             -5., -5., 10., 15., 0., 0.001, 0.001,
             -5., -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             ),
            (self.redshift_guess+self.dz, .75, 10., 1., 1., 1.,
             5., 5., 20., 70., 10., 30., 1.,
             5., 5., 5., 1.0, 30., 30.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             )))
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
             v_blr_1_100, v_blr_2_100,
             fwhm_blr_1_100, fwhm_blr_2_100, fHbeta_blr, fHalpha_blr, frac1b,
             v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, C_f, tau0hb, tau0ha,
             a0, b0, a1, b1) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_2_100-v_blr_1_100)/.25)**2
            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHbeta_blr/fHalpha_blr, mean=.4, scale=0.05)

            return lnprior

        return masks, guess, bounds, lnprior


    def model_exponential_blr_abs(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO35007_intr': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_intr': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583_intr': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'tau_thom': (r'$\tau_\mathrm{BLR}$', 1., '[---]'),
                'T_thom': (r'$T$', 1., r'$[10^4\,\mathrm{K}]$'),
                'v_abs_Hb_100': (r'$v_\mathrm{abs,H\beta}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'v_abs_Ha_100': (r'$v_\mathrm{abs,H\alpha}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_{f}$', 1., '[---]'),
                'tauHb': (r'$\tau_\mathrm{H\beta}$', 1., '[---]'),
                'tauHa': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'bk00': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk10': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk11': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
            }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewhb'  : (r'$\mathrm{EW(H\beta)}$', 1., r'[\AA]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fN26583_obs': (r'$F(\mathrm{[O\,II]\lambda 6583})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'sig_l_100': (r'$\sigma_\mathrm{l,BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'W'     : (r'$W$', 1., r'$\mathrm{[km\,s^{-1}]}$', float),
                'f_scatt' : (r'$f_\mathrm{scatt}$', 1., '[---]', float),
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                'logMBHsigl'      : (r'$\log\,(M_\bullet,sig)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBHsigl'     : (r'$\lambda_\mathrm{Edd,sig}$', 1.,
                            '[---]', float),
                }

        (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
         v_blr_100,
         fwhm_blr_100, fHbeta_blr, fHalpha_blr, tau_thom, T_thom,
         v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, C_f, tau0hb, tau0ha,
         a0, b0, a1, b1) = pars

        w_mum = np.array((
            self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hbeta, self.Halpha,
            self.Hbeta, self.Halpha
            ))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*6
                  + (np.exp(v_blr_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_Hb_100/self.c_100_kms),)
                  + (np.exp(v_abs_Ha_100/self.c_100_kms),)
                )
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 2.98
        fN26548 = fN26583 / 3.05
        (fHb_obs, fO34959_obs, fO35007_obs, fN26548_obs, fHa_obs, fN26583_obs,
         fHb_b_obs, fHa_b_obs, _, _) = np.array((
             fHb, fO34959, fO35007, fN26548, fHa, fN26583,
             fHbeta_blr, fHalpha_blr, 0., 0.))*atten * self.mu_lens

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*6
            + (fwhm_blr_100/self.fwhm2sig,)*2
            + (sig_abs_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHb_obs)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO34959_obs)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO35007_obs)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fN26548_obs)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa_obs)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fN26583_obs)

        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHb_b_obs)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHa_b_obs)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[8]
        tau_hb = tau0hb * gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[9]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=tau0_norm)
        absr = 1. - C_f + C_f * np.exp(-tau_hb - tau_ha)

        W_kms = (428. * tau_thom + 370.) * np.sqrt(T_thom)

        # Scatter Hb.
        W_mum = W_kms/self.c_kms * w_mum[6]
        dw = np.argmin(np.abs(self.wave-w_mum[6]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f8 = convolve(f6, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f6 *= (1. - f_scatt)
        f8 *= f_scatt

        # Scatter Ha.
        W_mum = W_kms/self.c_kms * w_mum[7]
        dw = np.argmin(np.abs(self.wave-w_mum[7]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f9 = convolve(f7, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f7 *= (1. - f_scatt)
        f9 *= f_scatt

        bk0 = a0 + (self.wave-w_mum[1]) * b0
        bk1 = a1 + (self.wave-w_mum[4]) * b1
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)

        if print_blobs:
            mask_hb = self.fit_mask[0] & (np.abs(self.wave-w_mum[8])/sig_mum[8]<10)
            mask_ha = self.fit_mask[1] & (np.abs(self.wave-w_mum[9])/sig_mum[9]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp(v_abs_Hb_100/self.c_100_kms) # Rest frame
            ewha /= (1+z_n)*np.exp(v_abs_Ha_100/self.c_100_kms) # Rest frame

            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)

            sig_l_um = get_sigma_l(self.wave, self.flux, bk1, self.fit_mask[1])
            sig_l_100 = sig_l_um / w_mum[4] * constants.c.to('1e2 km/s').value
    
            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Halpha_blr = fHalpha_blr * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            log_L_Ha_b = np.log10((L_Halpha_blr * lum_fact).to('1e42 erg/s').value)
            logMBH          = 6.6 + 0.47*log_L_Ha_b + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            logMBHsigl      = (0.683  + 7.413 + 0.554*log_L_Ha_b
                + 2.61 * (np.log10(sig_l_100)-1.5) )

            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBHsigl*units.Msun
            lEddMBHsigl = (edrat/lEdd).to(1).value

            return (
                ewhb, ewha,
                fHb_obs, fO35007_obs, fHa_obs, fN26583_obs,
                fHb_b_obs, fHa_b_obs,
                sig_l_100, W_kms, f_scatt,
                logSFR_Ha, log_L_Ha_b,
                logMBH, lEddMBH,
                logMBHsigl, lEddMBHsigl
                )

        return (
            f0, f1, f2, f3, f4, f5, f6*absr, f7*absr, f8*absr, f9*absr,
            bk0*absr, bk1*absr,
            )

    def model_exponential_blr_abs_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.15, 0.20)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1]]

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])

        guess = np.array((
            self.redshift_guess, 1., 0.0, 0.01, 0.01, 0.01,
            0., 15., 0.15, 1.5, 0.5, 1.0,
            0., 0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0.0, 0., 0., 0.,
             -5., 0.1, 0., 0.001, 0., 0.1,
             -5., -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             ),
            (self.redshift_guess+self.dz, 5., 10., 1., 1., 1.,
             5., 70., 10., 30., 30., 10.,
             5., 5., 5., 1.0, 30., 30.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             )))
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
             v_blr_100,
             fwhm_blr_100, fHbeta_blr, fHalpha_blr, tau_thom, T_thom,
             v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, C_f, tau0hb, tau0ha,
             a0, b0, a1, b1) = pars

            lnprior = 0.
            #lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/.5)**2
            #lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/.5)**2
            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHbeta_blr/fHalpha_blr, mean=.4, scale=0.05)

            return lnprior

        return masks, guess, bounds, lnprior



    def model_double_blr_voigt_abs_tiev(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO35007_intr': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_intr': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583_intr': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_1_100': (r'$v_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'v_blr_2_100': (r'$v_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr_intr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_Hb_100': (r'$v_\mathrm{abs,H\beta}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'v_abs_Ha_100': (r'$v_\mathrm{abs,H\alpha}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_abs_100': (r'$FWHM_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_{f}$', 1., '[---]'),
                'tauHb': (r'$\tau_\mathrm{H\beta}$', 1., '[---]'),
                'tauHa': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'bk00': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk10': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk11': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
            }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewhb'  : (r'$\mathrm{EW(H\beta)}$', 1., r'[\AA]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fN26583_obs': (r'$F(\mathrm{[O\,II]\lambda 6583})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'sig_l_100': (r'$\sigma_\mathrm{l,BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'log_L_Ha_b_2'   : (r'$\log\,L_\mathrm{b,2}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                'logMBH2'      : (r'$\log\,(M_{\bullet,2})$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH2'     : (r'$\lambda_\mathrm{Edd,2}$', 1.,
                            '[---]', float),
                'logMBHsigl'      : (r'$\log\,(M_\bullet,sig)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBHsigl'     : (r'$\lambda_\mathrm{Edd,sig}$', 1.,
                            '[---]', float),
                }

        (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
         v_blr_1_100, v_blr_2_100,
         fwhm_blr_1_100, fwhm_blr_2_100, fHbeta_blr, fHalpha_blr, frac1b,
         v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, fwhm_abs_100, C_f, tau0hb, tau0ha,
         a0, b0, a1, b1) = pars

        w_mum = np.array((
            self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hbeta, self.Hbeta, self.Halpha, self.Halpha,
            self.Hbeta, self.Halpha
            ))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*6
                  + (np.exp(v_blr_1_100/self.c_100_kms),)*2
                  + (np.exp(v_blr_2_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_Hb_100/self.c_100_kms),)
                  + (np.exp(v_abs_Ha_100/self.c_100_kms),)
                )
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 2.98
        fN26548 = fN26583 / 3.05
        fHb_b1 = fHbeta_blr * frac1b
        fHb_b2 = fHbeta_blr * (1.-frac1b)
        fHa_b1 = fHalpha_blr * frac1b
        fHa_b2 = fHalpha_blr * (1.-frac1b)
        (fHb_obs, fO34959_obs, fO35007_obs, fN26548_obs, fHa_obs, fN26583_obs,
         fHb_b1_obs, fHb_b2_obs, fHa_b1_obs, fHa_b2_obs, _, _) = np.array((
             fHb, fO34959, fO35007, fN26548, fHa, fN26583,
             fHb_b1, fHb_b2, fHa_b1, fHa_b2, 0., 0.))*atten * self.mu_lens

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*6
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)*2
            + (sig_abs_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHb_obs)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO34959_obs)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO35007_obs)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fN26548_obs)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa_obs)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fN26583_obs)

        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHb_b1_obs)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHb_b2_obs)
        f8 = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fHa_b1_obs)
        f9 = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fHa_b2_obs)

        hwhm_abs_mum = np.array((fwhm_abs_100,)*2) / constants.c.to('1e2 km/s').value
        hwhm_abs_mum *= (w_mum[10:12] / 2.)

        tau_hb = voigt_profile(self.wave-w_mum[10], sig_mum[10], hwhm_abs_mum[0])
        tau_hb = tau_hb/np.max(tau_hb)*tau0hb

        tau_ha = voigt_profile(self.wave-w_mum[11], sig_mum[11], hwhm_abs_mum[1])
        tau_ha = tau_ha/np.max(tau_ha)*tau0ha

        absr = 1. - C_f + C_f * np.exp(-tau_hb - tau_ha)

        bk0 = a0 + (self.wave-w_mum[1]) * b0
        bk1 = a1 + (self.wave-w_mum[4]) * b1
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)

        if print_blobs:
            mask_hb = self.fit_mask[0] & (np.abs(self.wave-w_mum[10])/sig_mum[10]<10)
            mask_ha = self.fit_mask[1] & (np.abs(self.wave-w_mum[11])/sig_mum[11]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp(v_abs_Hb_100/self.c_100_kms) # Rest frame
            ewha /= (1+z_n)*np.exp(v_abs_Ha_100/self.c_100_kms) # Rest frame

            fwhm_blr_100 = get_fwhm(
                fwhm_blr_1_100, fwhm_blr_2_100, frac1b, guess=0.15)
            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)

            sig_l_um = get_sigma_l(self.wave, self.flux, bk1, self.fit_mask[1])
            sig_l_100 = sig_l_um / w_mum[4] * constants.c.to('1e2 km/s').value
    
            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Halpha_blr = fHalpha_blr * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            log_L_Ha_b = np.log10((L_Halpha_blr * lum_fact).to('1e42 erg/s').value)
            logMBH          = 6.6 + 0.47*log_L_Ha_b + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            log_L_Ha_b_2 = log_L_Ha_b + np.log10(1.-frac1b)
            logMBH2         = 6.6 + 0.47*log_L_Ha_b_2 + 2.06*np.log10(fwhm_blr_2_100/10.)
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH2*units.Msun
            lEddMBH2 = (edrat/lEdd).to(1).value

            logMBHsigl      = (0.683  + 7.413 + 0.554*log_L_Ha_b
                + 2.61 * (np.log10(sig_l_100)-1.5) )

            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBHsigl*units.Msun
            lEddMBHsigl = (edrat/lEdd).to(1).value

            return (
                ewhb, ewha,
                fHb_obs, fO35007_obs, fHa_obs, fN26583_obs,
                fHb_b1_obs+fHb_b2_obs, fHa_b1_obs+fHa_b2_obs,
                fwhm_blr_100, sig_l_100,
                logSFR_Ha, log_L_Ha_b, log_L_Ha_b_2,
                logMBH, lEddMBH,
                logMBH2, lEddMBH2,
                logMBHsigl, lEddMBHsigl
                )

        return (
            f0, f1, f2, f3, f4, f5, f6*absr, f7*absr, f8*absr, f9*absr,
            bk0*absr, bk1*absr,
            )

    def model_double_blr_voigt_abs_tiev_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.15, 0.20)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1]]

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])

        guess = np.array((
            self.redshift_guess, 1., 0.0, 0.01, 0.01, 0.01,
            0., 0., 15., 30., 0.15, 1.5, .5,
            0., 0., 1.20, 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0.0, 0., 0., 0.,
             -5., -5., 10., 15., 0., 0.001, 0.001,
             -5., -5., 0., 0.0, 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             ),
            (self.redshift_guess+self.dz, 5., 10., 1., 1., 1.,
             5., 5., 20., 70., 10., 30., 1.,
             5., 5., 5., 10., 1.0, 30., 30.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             )))
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, Av, fO35007, fHa, fN26583,
             v_blr_1_100, v_blr_2_100,
             fwhm_blr_1_100, fwhm_blr_2_100, fHbeta_blr, fHalpha_blr, frac1b,
             v_abs_Hb_100, v_abs_Ha_100, sig_abs_100, fwhm_abs_100, C_f, tau0hb, tau0ha,
             a0, b0, a1, b1) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_2_100-v_blr_1_100)/.25)**2
            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHbeta_blr/fHalpha_blr, mean=.4, scale=0.05)

            return lnprior

        return masks, guess, bounds, lnprior






# UNTIL HERE


    def model_o2_voigt_2blr_double_abs(
        self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_m_100': (r'$\sigma_\mathrm{m}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fFe24359': (r'$F(\mathrm{[Fe\,II]\lambda 4359})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHgamma_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\gamma})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_abs_1_100': (r'$v_\mathrm{abs,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_1_100': (r'$\sigma_\mathrm{abs,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf_1': (r'$C_{f,1}$', 1., '[---]'),
                'tauHg_1': (r'$\tau_\mathrm{H\gamma,1}$', 1., '[---]'),
                'tauHb_1': (r'$\tau_\mathrm{H\beta,1}$', 1., '[---]'),
                'tauHa_1': (r'$\tau_\mathrm{H\alpha,1}$', 1., '[---]'),
                'v_abs_2_100': (r'$v_\mathrm{abs,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_2_100': (r'$\sigma_\mathrm{abs,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf_2': (r'$C_{f,2}$', 1., '[---]'),
                'tauHg_2': (r'$\tau_\mathrm{H\gamma,2}$', 1., '[---]'),
                'tauHb_2': (r'$\tau_\mathrm{H\beta,2}$', 1., '[---]'),
                'tauHa_2': (r'$\tau_\mathrm{H\alpha,2}$', 1., '[---]'),
                'fO35007_out': (r'$F_\mathrm{out}(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_out_100': (r'$v_\mathrm{out}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_out_100': (r'$\sigma_\mathrm{out}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'bk00': (r'$bk_0$', 1.,
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1.,
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk10': (r'$bk_0$', 1.,
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk11': (r'$bk_1$', 1.,
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk20': (r'$bk_0$', 1.,
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk21': (r'$bk_1$', 1.,
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewhg_1'  : (r'$\mathrm{EW(H\gamma,1)}$', 1., r'[\AA]', float),
                'ewhb_1'  : (r'$\mathrm{EW(H\beta,1)}$', 1., r'[\AA]', float),
                'ewha_1'  : (r'$\mathrm{EW(H\alpha,1)}$', 1., r'[\AA]', float),
                'ewhg_2'  : (r'$\mathrm{EW(H\gamma,2)}$', 1., r'[\AA]', float),
                'ewhb_2'  : (r'$\mathrm{EW(H\beta,2)}$', 1., r'[\AA]', float),
                'ewha_2'  : (r'$\mathrm{EW(H\alpha,2)}$', 1., r'[\AA]', float),
                'fHgamma_obs' : (r'$F_\mathrm{n}(\mathrm{H\gamma})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO34363_obs': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fN26583_obs': (r'$F(\mathrm{[O\,II]\lambda 6583})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHgamma_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\gamma})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$', float),
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b_ism'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                }
        (z_n, sig_n_100, sig_m_100, Av, fFe24359, fO34363, fO35007, fHa, fN26583,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHg_b, fHb_b, fHa_b,
         v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
         v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
         fO35007_out, v_out_100, sig_out_100,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.Hgamma, self.FeII4359, self.OIII4363, self.FeII4414,
            self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hgamma, self.Hbeta, self.Halpha,
            self.Hgamma, self.Hbeta, self.Halpha,
            self.Hgamma, self.Hbeta, self.Halpha,
            self.OIII4959, self.OIII5007,
            ))

        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*10
                  + (np.exp(v_blr_100/self.c_100_kms),)*3
                  + (np.exp(v_abs_1_100/self.c_100_kms),)*3
                  + (np.exp(v_abs_2_100/self.c_100_kms),)*3
                  + (np.exp(v_out_100/self.c_100_kms),)*2)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fHg = fHb*self.Hg2Hb
        fFe24414 = fFe24359 / 1.436 # From pyneb. 1.35 from A_ki NIST
        fO34959 = fO35007 / 2.98
        fO34959_out = fO35007_out / 2.98
        fN26548 = fN26583 / 3.05
        (fHg, fFe24359, fO34363, fFe24414, fHb, fO34959, fO35007,
         fN26548, fHa, fN26583,
         fHg_b, fHb_b, fHa_b, _, _, _, _, _, _, _, _) = np.array((
             fHg, fFe24359, fO34363, fFe24414, fHb, fO34959, fO35007,
             fN26548, fHa, fN26583,
             fHg_b, fHb_b, fHa_b,
             0., 0., 0., 0., 0., 0, .0, 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100, sig_m_100, sig_n_100, sig_m_100)
            + (sig_n_100,)*6
            + (fwhm_blr_1_100/self.fwhm2sig,)*3
            + (sig_abs_1_100,)*3 + (sig_abs_2_100,)*3
            + (sig_out_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        hwhm_blr_mum = np.array((fwhm_blr_2_100,)*3) / constants.c.to('1e2 km/s').value
        hwhm_blr_mum *= (w_mum[8:11] / 2.)

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHg)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fFe24359)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO34363)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fFe24414)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHb)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fO34959)
        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fO35007)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fN26548)
        f8 = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fHa)
        f9 = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fN26583)

        f10= voigt_profile(self.wave-w_mum[10], sig_mum[10], hwhm_blr_mum[0])*fHg_b
        f11= voigt_profile(self.wave-w_mum[11], sig_mum[11], hwhm_blr_mum[1])*fHb_b
        f12= voigt_profile(self.wave-w_mum[12], sig_mum[12], hwhm_blr_mum[2])*fHa_b

        f19 = gauss_int2(self.wave, mu=w_mum[19], sig=sig_mum[19], flux=fO34959_out)
        f20 = gauss_int2(self.wave, mu=w_mum[20], sig=sig_mum[20], flux=fO35007_out)

        ###f13= gauss_int2(self.wave, mu=w_mum[13], sig=sig_mum[13], flux=fHa_b2)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[13]
        tau_hg_1 = tau0hg_1 * gauss_int2(self.wave, mu=w_mum[13], sig=sig_mum[13], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[14]
        tau_hb_1 = tau0hb_1 * gauss_int2(self.wave, mu=w_mum[14], sig=sig_mum[14], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[15]
        tau_ha_1 = tau0ha_1 * gauss_int2(self.wave, mu=w_mum[15], sig=sig_mum[15], flux=tau0_norm)
        absrhg_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hg_1)
        absrhb_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hb_1)
        absrha_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_ha_1)

        # Second absorber.
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[16]
        tau_hg_2 = tau0hg_2 * gauss_int2(self.wave, mu=w_mum[16], sig=sig_mum[16], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[17]
        tau_hb_2 = tau0hb_2 * gauss_int2(self.wave, mu=w_mum[17], sig=sig_mum[17], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[18]
        tau_ha_2 = tau0ha_2 * gauss_int2(self.wave, mu=w_mum[18], sig=sig_mum[18], flux=tau0_norm)
        absrhg_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hg_2)
        absrhb_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hb_2)
        absrha_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_ha_2)

        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[5]) * b1
        bk2 = a2 + (self.wave-w_mum[8]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hg = self.fit_mask[0] & (np.abs(self.wave-w_mum[10])/hwhm_blr_mum[0]<10)
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[11])/hwhm_blr_mum[1]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[12])/hwhm_blr_mum[2]<10)
            dwhg = np.gradient(self.wave[mask_hg])
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhg_1 = np.sum((1. - np.exp(-tau_hg_1[mask_hg]))*dwhg)*1e4 # To [AA]
            ewhb_1 = np.sum((1. - np.exp(-tau_hb_1[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha_1 = np.sum((1. - np.exp(-tau_ha_1[mask_ha]))*dwha)*1e4 # To [AA]
            ewhg_1 /= (1+z_n)*np.exp(v_abs_1_100/self.c_100_kms) # Rest frame
            ewhb_1 /= (1+z_n)*np.exp(v_abs_1_100/self.c_100_kms) # Rest frame
            ewha_1 /= (1+z_n)*np.exp(v_abs_1_100/self.c_100_kms) # Rest frame

            ewhg_2 = np.sum((1. - np.exp(-tau_hg_2[mask_hg]))*dwhg)*1e4 # To [AA]
            ewhb_2 = np.sum((1. - np.exp(-tau_hb_2[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha_2 = np.sum((1. - np.exp(-tau_ha_2[mask_ha]))*dwha)*1e4 # To [AA]
            ewhg_2 /= (1+z_n)*np.exp(v_abs_2_100/self.c_100_kms) # Rest frame
            ewhb_2 /= (1+z_n)*np.exp(v_abs_2_100/self.c_100_kms) # Rest frame
            ewha_2 /= (1+z_n)*np.exp(v_abs_2_100/self.c_100_kms) # Rest frame

            ewhg_1 = (1. - absrhg_1)
            ewhg_1 = np.nansum(ewhg_1[mask_hg]*dwhg)*1e4        # To [AA]
            ewhg_1 /= (1+z_n)*np.exp(v_abs_1_100/self.c_100_kms) # Rest frame
            ewhb_1 = (1. - absrhb_1)
            ewhb_1 = np.nansum(ewhb_1[mask_hb]*dwhb)*1e4        # To [AA]
            ewhb_1 /= (1+z_n)*np.exp(v_abs_1_100/self.c_100_kms) # Rest frame
            ewha_1 = (1. - absrha_1)
            ewha_1 = np.nansum(ewha_1[mask_ha]*dwha)*1e4        # To [AA]
            ewha_1 /= (1+z_n)*np.exp(v_abs_1_100/self.c_100_kms) # Rest frame

            ewhg_2 = (1. - absrhg_2)
            ewhg_2 = np.nansum(ewhg_2[mask_hg]*dwhg)*1e4        # To [AA]
            ewhg_2 /= (1+z_n)*np.exp(v_abs_2_100/self.c_100_kms) # Rest frame
            ewhb_2 = (1. - absrhb_2)
            ewhb_2 = np.nansum(ewhb_2[mask_hb]*dwhb)*1e4        # To [AA]
            ewhb_2 /= (1+z_n)*np.exp(v_abs_2_100/self.c_100_kms) # Rest frame
            ewha_2 = (1. - absrha_2)
            ewha_2 = np.nansum(ewha_2[mask_ha]*dwha)*1e4        # To [AA]
            ewha_2 /= (1+z_n)*np.exp(v_abs_2_100/self.c_100_kms) # Rest frame

            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)

            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Ha_b_ism    = fHa_b * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            log_L_Ha_b_ism = np.log10((L_Ha_b_ism * lum_fact).to('1e42 erg/s').value)
            fwhm_blr_100 = fwhm_blr_2_100
            logMBH          = 6.6 + 0.47*log_L_Ha_b_ism + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b_ism * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            return (
                ewhg_1, ewhb_1, ewha_1, ewhg_2, ewhb_2, ewha_2,
                fHg, fO34363, fHb, fO35007, fHa, fN26583,
                fHg_b, fHb_b, fHa_b, fwhm_blr_100,
                logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH
                )

        return (
            f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
            f10*absrhg_1*absrhg_2, f11*absrhb_1*absrhb_2, f12*absrha_1*absrha_2,
            f19, f20,
            bk0*absrhg_1*absrhg_2, bk1*absrhb_1*absrhb_2,
            bk2*absrha_1*absrha_2,
            )

    def model_o2_voigt_2blr_double_abs_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4340, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.09, 0.15, 0.20)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2]
            ]

        # Mask prominent iron lines near Hgamma
        iron_waves = np.array((.4244, .42647, .4279, .4288,))
        _dw_ = np.gradient(self.wave)
        masks[0] = masks[0] & np.all([
            np.abs(self.wave-iw*(1+self.redshift_guess))/_dw_>3
            for iw in iron_waves], axis=0)
        masks[2] = masks[2] & np.all([ # Mask regions with bad emission lines within 3 pixels
             np.abs(self.wave-_line_wave_*(1+self.redshift_guess))/_dw_>3
             for _line_wave_ in (self.OI6300, self.OI6363, self.HeI6678)], axis=0)

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        guess = np.array((
            self.redshift_guess, 1., 1., 0.5, 0.01, 0.01, 0.01, 0.01, 0.01,
            0., 10., 25., 0.05, 0.15, 1.5,
            0., 1.20, 0.9, 3., 3., 3.,
            0., 1.20, 0.9, 3., 3., 3.,
            0.1, 0., 2.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 1.e-5, 0., 0., 0., 0., 0., 0.,
             -5., 1., 10., 0., 0., 0.001,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             0., -5., 1.,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5., 5., 1., 1., 1., 1., 1.,
             5., 70., 70., 1., 1., 5.,
             5., 5., 1.0, 30., 30., 30.,
             5., 5., 1.0, 30., 30., 30.,
             2., 5., 10.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             bkg20+10*bkg21, 100.)))
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, sig_m_100, Av, fFe24359, fO34363, fO35007, fHa, fN26583,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHg_b, fHb_b, fHa_b,
             v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
             v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
             fO35007_out, v_out_100, sig_out_100,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/.5)**2
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/.5)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.4, scale=0.05)
            lnprior += log_erfc_prior(fHg_b/fHa_b, mean=.2, scale=0.05)

            lnprior += log_erfc_prior(sig_n_100/sig_out_100, mean=1., scale=0.05)

            lnprior += -0.5*((sig_m_100-143)/12)**2.

            return lnprior

        return masks, guess, bounds, lnprior


class jwst_spec_fitter(jwst_spec_models, specfitt.jwst_spec_fitter):
    mu_lens = 5.8
    pass

