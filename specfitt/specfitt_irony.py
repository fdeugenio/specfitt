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

class jwst_spec_models(specfitt.jwst_spec_models):

    def model_o2_voigt_blr_double_abs(
        self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_m_100': (r'$\sigma_\mathrm{m}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
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
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b_ism'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                }
        (z_n, sig_n_100, sig_m_100, Av, fO34363, fO35007, fHa, fN26583,
         v_blr_100, fwhm_blr_100, fHg_b, fHb_b, fHa_b,
         v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
         v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.Hgamma, self.OIII4363, self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hgamma, self.Hbeta, self.Halpha,
            self.Hgamma, self.Hbeta, self.Halpha, self.Hgamma, self.Hbeta, self.Halpha))

        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*8
                  + (np.exp(v_blr_100/self.c_100_kms),)*3
                  + (np.exp(v_abs_1_100/self.c_100_kms),)*3
                  + (np.exp(v_abs_2_100/self.c_100_kms),)*3)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fHg = fHb*self.Hg2Hb
        fO34959 = fO35007 / 2.98
        fN26548 = fN26583 / 3.05
        (fHg, fO34363, fHb, fO34959, fO35007, fN26548, fHa, fN26583,
         fHg_b, fHb_b, fHa_b, _, _, _, _, _, _) = np.array((
             fHg, fO34363, fHb, fO34959, fO35007, fN26548, fHa, fN26583,
             fHg_b, fHb_b, fHa_b, 0., 0., 0., 0, .0, 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100, sig_m_100)
            + (sig_n_100,)*6
            + (0.,)*3 # FWHM for Voigt applied separately.
            + (sig_abs_1_100,)*3 + (sig_abs_2_100,)*3)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        hwhm_blr_mum = np.array((fwhm_blr_100,)*3) / constants.c.to('1e2 km/s').value
        hwhm_blr_mum *= (w_mum[8:11] / 2.)


        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHg)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO34363)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fHb)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO34959)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fO35007)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fN26548)
        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHa)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fN26583)

        f8 = voigt_profile(self.wave-w_mum[8], sig_mum[8], hwhm_blr_mum[0])*fHg_b
        f9 = voigt_profile(self.wave-w_mum[9], sig_mum[9], hwhm_blr_mum[1])*fHb_b
        f10= voigt_profile(self.wave-w_mum[10], sig_mum[10], hwhm_blr_mum[2])*fHa_b

        ###f13= gauss_int2(self.wave, mu=w_mum[13], sig=sig_mum[13], flux=fHa_b2)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[11]
        tau_hg_1 = tau0hg_1 * gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[12]
        tau_hb_1 = tau0hb_1 * gauss_int2(self.wave, mu=w_mum[12], sig=sig_mum[12], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[13]
        tau_ha_1 = tau0ha_1 * gauss_int2(self.wave, mu=w_mum[13], sig=sig_mum[13], flux=tau0_norm)
        absrhg_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hg_1)
        absrhb_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hb_1)
        absrha_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_ha_1)

        # Second absorber.
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[14]
        tau_hg_2 = tau0hg_2 * gauss_int2(self.wave, mu=w_mum[14], sig=sig_mum[14], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[15]
        tau_hb_2 = tau0hb_2 * gauss_int2(self.wave, mu=w_mum[15], sig=sig_mum[15], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[16]
        tau_ha_2 = tau0ha_2 * gauss_int2(self.wave, mu=w_mum[16], sig=sig_mum[16], flux=tau0_norm)
        absrhg_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hg_2)
        absrhb_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hb_2)
        absrha_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_ha_2)

        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[3]) * b1
        bk2 = a2 + (self.wave-w_mum[6]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hg = self.fit_mask[0] & (np.abs(self.wave-w_mum[8])/hwhm_blr_mum[0]<10)
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[9])/hwhm_blr_mum[1]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[10])/hwhm_blr_mum[2]<10)
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

            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)

            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Ha_b_ism    = fHa_b * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            log_L_Ha_b_ism = np.log10((L_Ha_b_ism * lum_fact).to('1e42 erg/s').value)
            logMBH          = 6.6 + 0.47*log_L_Ha_b_ism + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b_ism * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            return (
                ewhg_1, ewhb_1, ewha_1, ewhg_2, ewhb_2, ewha_2,
                fHg, fO34363, fHb, fO35007, fHa, fN26583,
                fHg_b, fHb_b, fHa_b,
                logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH
                )

        return (
            f0, f1, f2, f3, f4, f5, f6, f7,
            f8*absrhg_1*absrhg_2, f9*absrhb_1*absrhb_2, f10*absrha_1*absrha_2,
            bk0*absrhg_1*absrhg_2, bk1*absrhb_1*absrhb_2,
            bk2*absrha_1*absrha_2,
            )

    def model_o2_voigt_blr_double_abs_fit_init(self):
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
        iron_waves = np.array((.4244, .42647, .4279, .4288, .4414))
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
            self.redshift_guess, 1., 1., 0.5, 0.01, 0.01, 0.01, 0.01,
            0., 25., 0.05, 0.15, 1.5,
            0., 1.20, 0.9, 3., 3., 3.,
            0., 1.20, 0.9, 3., 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 1.e-5, 0., 0., 0., 0., 0.,
             -5., 10., 0., 0., 0.001,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5., 5., 1., 1., 1., 1.,
              5., 70., 1., 1., 5.,
              5., 5., 1.0, 30., 30., 30.,
              5., 5., 1.0, 30., 30., 30.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             bkg20+10*bkg21, 100.)))
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])

        def lnprior(pars, *args):
            (z_n, sig_n_100, sig_m_100, Av, fO34363, fO35007, fHa, fN26583,
             v_blr_100, fwhm_blr_100, fHg_b, fHb_b, fHa_b,
             v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
             v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/.5)**2
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/.5)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.4, scale=0.05)
            lnprior += log_erfc_prior(fHg_b/fHa_b, mean=.2, scale=0.05)

            return lnprior

        return masks, guess, bounds, lnprior



    def model_o2_double_blr_double_abs(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_m_100': (r'$\sigma_\mathrm{m}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
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
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
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

        (z_n, sig_n_100, sig_m_100, Av, fO34363, fO35007, fHa, fN26583,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHg_b, fHb_b, fHa_b, frac1b,
         v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
         v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.Hgamma, self.OIII4363, self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hgamma, self.Hgamma, self.Hbeta, self.Hbeta, self.Halpha, self.Halpha,
            self.Hgamma, self.Hbeta, self.Halpha, self.Hgamma, self.Hbeta, self.Halpha))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*8
                  + (np.exp(v_blr_100/self.c_100_kms),)*6
                  + (np.exp(v_abs_1_100/self.c_100_kms),)*3
                  + (np.exp(v_abs_2_100/self.c_100_kms),)*3)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fHg = fHb*self.Hg2Hb
        fO34959 = fO35007 / 2.98
        fN26548 = fN26583 / 3.05
        fHg_b1 = fHg_b * frac1b
        fHg_b2 = fHg_b * (1.-frac1b)
        fHb_b1 = fHb_b * frac1b
        fHb_b2 = fHb_b * (1.-frac1b)
        fHa_b1 = fHa_b * frac1b
        fHa_b2 = fHa_b * (1.-frac1b)
        (fHg, fO34363, fHb, fO34959, fO35007, fN26548, fHa, fN26583,
         fHg_b1, fHg_b2, fHb_b1, fHb_b2, fHa_b1, fHa_b2,
             _, _, _, _, _, _) = np.array((
             fHg, fO34363, fHb, fO34959, fO35007, fN26548, fHa, fN26583,
             fHg_b1, fHg_b2, fHb_b1, fHb_b2, fHa_b1, fHa_b2,
             0., 0., 0., 0, .0, 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100, sig_m_100)
            + (sig_n_100,)*6
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)*3
            + (sig_abs_1_100,)*3 + (sig_abs_2_100,)*3)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHg)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO34363)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fHb)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO34959)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fO35007)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fN26548)
        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHa)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fN26583)
        f8 = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fHg_b1)
        f9 = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fHg_b2)
        f10= gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=fHb_b1)
        f11= gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=fHb_b2)
        f12= gauss_int2(self.wave, mu=w_mum[12], sig=sig_mum[12], flux=fHa_b1)
        f13= gauss_int2(self.wave, mu=w_mum[13], sig=sig_mum[13], flux=fHa_b2)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[14]
        tau_hg_1 = tau0hg_1 * gauss_int2(self.wave, mu=w_mum[14], sig=sig_mum[14], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[15]
        tau_hb_1 = tau0hb_1 * gauss_int2(self.wave, mu=w_mum[15], sig=sig_mum[15], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[16]
        tau_ha_1 = tau0ha_1 * gauss_int2(self.wave, mu=w_mum[16], sig=sig_mum[16], flux=tau0_norm)
        absrhg_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hg_1)
        absrhb_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hb_1)
        absrha_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_ha_1)

        # Second absorber.
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[17]
        tau_hg_2 = tau0hg_2 * gauss_int2(self.wave, mu=w_mum[17], sig=sig_mum[17], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[18]
        tau_hb_2 = tau0hb_2 * gauss_int2(self.wave, mu=w_mum[18], sig=sig_mum[18], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[19]
        tau_ha_2 = tau0ha_2 * gauss_int2(self.wave, mu=w_mum[19], sig=sig_mum[19], flux=tau0_norm)
        absrhg_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hg_2)
        absrhb_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hb_2)
        absrha_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_ha_2)

        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[3]) * b1
        bk2 = a2 + (self.wave-w_mum[6]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hg = self.fit_mask[0] & (np.abs(self.wave-w_mum[8])/sig_mum[8]<10)
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[10])/sig_mum[10]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[12])/sig_mum[12]<10)
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

            fwhm_blr_100 = get_fwhm(
                fwhm_blr_1_100, fwhm_blr_2_100, frac1b, guess=0.15)
            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)
    
            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Ha_b_ism    = fHa_b * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            log_L_Ha_b_ism = np.log10((L_Ha_b_ism * lum_fact).to('1e42 erg/s').value)
            logMBH          = 6.6 + 0.47*log_L_Ha_b_ism + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b_ism * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            return (
                ewhg_1, ewhb_1, ewha_1, ewhg_2, ewhb_2, ewha_2,
                fHg, fO34363, fHb, fO35007, fHa, fN26583,
                fHg_b1+fHg_b2, fHb_b1+fHb_b2, fHa_b1+fHa_b2,
                fwhm_blr_100, logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH
                )

        return (
            f0, f1, f2, f3, f4, f5, f6, f7,
            f8*absrhg_1*absrhg_2, f9*absrhg_1*absrhg_2,
            f10*absrhb_1*absrhb_2, f11*absrhb_1*absrhb_2,
            f12*absrha_1*absrha_2, f13*absrha_1*absrha_2,
            bk0*absrhg_1*absrhg_2, bk1*absrhb_1*absrhb_2,
            bk2*absrha_1*absrha_2,
            )

    def model_o2_double_blr_double_abs_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4340, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.05, 0.15, 0.20)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2]]

        # Mask prominent iron lines near Hgamma
        iron_waves = np.array((.4244, .42647, .4279, .4288, .4414))
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
            self.redshift_guess, 1., 1., 0.5, 0.01, 0.01, 0.01, 0.01,
            0., 15., 30., 0.05, 0.15, 1.5, .5,
            0., 1.20, 0.9, 3., 3., 3.,
            0., 1.20, 0.9, 3., 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 1.e-5, 0., 0., 0., 0., 0.,
             -5., 10., 20., 0., 0., 0.001, 0.,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 10., 1., 1., 5., 5., 5.,
              5., 20., 70., 5., 10., 30., 1.,
              5., 5., 1.0, 30., 30., 30.,
              5., 5., 1.0, 30., 30., 30.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             bkg20+10*bkg21, 100.)))
        #mask = masks[0] | masks[1] | masks[2]
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])
        #central_wave = 0.6600*(1+self.redshift_guess)
        #mask = np.abs(self.wave - central_wave)<0.10 # [um]
        #central_pix  = np.argmin(np.abs(self.wave - central_wave))
        #mask[central_pix-15:central_pix+15+1] = True
        #mask = mask & ~self.mask
        #func_name = traceback.extract_stack(None, 2)[-1][2]
        #assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'

        def lnprior(pars, *args):
            (z_n, sig_n_100, sig_m_100, Av, fO34363, fO35007, fHa, fN26583,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHg_b, fHb_b, fHa_b, frac1b,
             v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
             v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/.5)**2
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/.5)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.4, scale=0.05)
            lnprior += log_erfc_prior(fHg_b/fHa_b, mean=.2, scale=0.05)

            # erfc prior. Difference between absorber velocities less than 100 km/s.
            # lnprior += log_erfc_prior(v_abs_1_100-v_abs_2_100, mean=1., scale=1.)

            return lnprior

        return masks, guess, bounds, lnprior


    def model_o2_exponential_blr_double_abs(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_m_100': (r'$\sigma_\mathrm{m}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHgamma_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\gamma})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'tau_thom': (r'$\tau_\mathrm{BLR}$', 1., '[---]'),
                'T_thom': (r'$T$', 1., r'$[10^4\,\mathrm{K}]$'),
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
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
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
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b_ism'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                'W'     : (r'$W$', 1., r'$\mathrm{[km\,s^{-1}]}$', float),
                }

        (z_n, sig_n_100, sig_m_100, Av, fO34363, fO35007, fHa, fN26583,
         v_blr_100, fwhm_blr_100, fHg_b, fHb_b, fHa_b, tau_thom, T_thom,
         v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
         v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.Hgamma, self.OIII4363, self.Hbeta, self.OIII4959, self.OIII5007,
            self.NII6548, self.Halpha, self.NII6583,
            self.Hgamma, self.Hbeta, self.Halpha,
            self.Hgamma, self.Hbeta, self.Halpha,
            self.Hgamma, self.Hbeta, self.Halpha))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*8
                  + (np.exp(v_blr_100/self.c_100_kms),)*3
                  + (np.exp(v_abs_1_100/self.c_100_kms),)*3
                  + (np.exp(v_abs_2_100/self.c_100_kms),)*3)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fHg = fHb*self.Hg2Hb
        fO34959 = fO35007 / 2.98
        fN26548 = fN26583 / 3.05
        (fHg, fO34363, fHb, fO34959, fO35007, fN26548, fHa, fN26583,
         fHg_b, fHb_b, fHa_b, _, _, _, _, _, _) = np.array((
             fHg, fO34363, fHb, fO34959, fO35007, fN26548, fHa, fN26583,
             fHg_b, fHb_b, fHa_b, 0., 0., 0., 0, .0, 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100, sig_m_100)
            + (sig_n_100,)*6
            + (fwhm_blr_100/self.fwhm2sig,)*3
            + (sig_abs_1_100,)*3 + (sig_abs_2_100,)*3)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHg)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO34363)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fHb)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO34959)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fO35007)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fN26548)
        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHa)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fN26583)
        f8 = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fHg_b)
        f9 = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fHb_b)
        f10= gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=fHa_b)

        # First absorber.
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[11] 
        tau_hg_1 = tau0hg_1 * gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[12] 
        tau_hb_1 = tau0hb_1 * gauss_int2(self.wave, mu=w_mum[12], sig=sig_mum[12], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[13] 
        tau_ha_1 = tau0ha_1 * gauss_int2(self.wave, mu=w_mum[13], sig=sig_mum[13], flux=tau0_norm)
        absrhg_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hg_1)
        absrhb_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hb_1)
        absrha_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_ha_1)

        # Second absorber.
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[14] 
        tau_hg_2 = tau0hg_2 * gauss_int2(self.wave, mu=w_mum[14], sig=sig_mum[14], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[15] 
        tau_hb_2 = tau0hb_2 * gauss_int2(self.wave, mu=w_mum[15], sig=sig_mum[15], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[16] 
        tau_ha_2 = tau0ha_2 * gauss_int2(self.wave, mu=w_mum[16], sig=sig_mum[16], flux=tau0_norm)
        absrhg_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hg_2)
        absrhb_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hb_2)
        absrha_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_ha_2)

        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[3]) * b1
        bk2 = a2 + (self.wave-w_mum[6]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        W_kms = (428. * tau_thom + 370.) * np.sqrt(T_thom)

        # Scatter Hg.
        W_mum = W_kms/self.c_kms * w_mum[8]
        dw = np.argmin(np.abs(self.wave-w_mum[8]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f11 = convolve(f8, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f8 *= (1. - f_scatt)
        f11 *= f_scatt

        # Scatter Hb.
        W_mum = W_kms/self.c_kms * w_mum[9]
        dw = np.argmin(np.abs(self.wave-w_mum[9]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f12 = convolve(f9, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f9 *= (1. - f_scatt)
        f12 *= f_scatt
       
        # Scatter Ha.
        W_mum = W_kms/self.c_kms * w_mum[10]
        dw = np.argmin(np.abs(self.wave-w_mum[10]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f13 = convolve(f10, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f10 *= (1. - f_scatt)
        f13 *= f_scatt

        if print_blobs:
            mask_hg = self.fit_mask[0] & (np.abs(self.wave-w_mum[8])/sig_mum[8]<10)
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[9])/sig_mum[9]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[10])/sig_mum[10]<10)
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

            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)
    
            L_Ha_n = fHa * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s').value
            SFR_Ha = (L_Ha_n * units.Unit('1e42 erg/s') * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value
            logSFR_Ha        = np.log10(SFR_Ha)
            L_Ha_b_ism    = fHa_b * 1e2 * units.Unit('1e-18 erg/(s cm2)') / _g03_(self.Halpha, Av)
            log_L_Ha_b_ism = np.log10((L_Ha_b_ism * lum_fact).to('1e42 erg/s').value)
            logMBH          = 6.6 + 0.47*log_L_Ha_b_ism + 2.06*np.log10(fwhm_blr_100/10.)
            edrat = 130 * 10**log_L_Ha_b_ism * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            return (
                ewhg_1, ewhb_1, ewha_1, ewhg_2, ewhb_2, ewha_2,
                fHg, fO34363, fHb, fO35007, fHa, fN26583,
                fHg_b, fHb_b, fHa_b,
                logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH, W_kms
                )

        return (
            f0, f1, f2, f3, f4, f5, f6, f7,
            f8*absrhg_1*absrhg_2, f9*absrhb_1*absrhb_2, f10*absrha_1*absrha_2,
            f11*absrhg_1*absrhg_2, f12*absrhb_1*absrhb_2, f13*absrha_1*absrha_2,
            bk0*absrhg_1*absrhg_2, bk1*absrhb_1*absrhb_2,
            bk2*absrha_1*absrha_2, 
            )


    def model_o2_exponential_blr_double_abs_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4340, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.05, 0.15, 0.20)
        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2]]

        # Mask prominent iron lines near Hgamma
        iron_waves = np.array((.4244, .42647, .4279, .4288, .4414))
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
            self.redshift_guess, 1., 1., 0.5, 0.01, 0.01, 0.01, 0.01,
            0., 15., 0.08, 0.15, 1.5, 0.5, 1.,
            0., 1.20, 0.9, 3., 3., 3.,
            0., 1.20, 0.9, 3., 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 1.e-5, 0., 0., 0., 0., 0.,
             -5., 1., 0., 0., 0.001, 0., 0.1,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             -5., 0., 0.0, 0.1, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 10., 5., 1., 1., 1., 1.,
              5., 70., 1., 1., 5., 30., 10.,
              5., 5., 1.0, 30., 30., 30.,
              5., 5., 1.0, 30., 30., 30.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             bkg20+10*bkg21, 100.)))
        #mask = masks[0] | masks[1] | masks[2]
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])
        #central_wave = 0.6600*(1+self.redshift_guess)
        #mask = np.abs(self.wave - central_wave)<0.10 # [um]
        #central_pix  = np.argmin(np.abs(self.wave - central_wave))
        #mask[central_pix-15:central_pix+15+1] = True
        #mask = mask & ~self.mask
        #func_name = traceback.extract_stack(None, 2)[-1][2]
        #assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'

        def lnprior(pars, *args):
            (z_n, sig_n_100, sig_m_100, Av, fO34363, fO35007, fHa, fN26583,
             v_blr_100, fwhm_blr_100, fHg_b, fHb_b, fHa_b, tau_thom, T_thom,
             v_abs_1_100, sig_abs_1_100, C_f_1, tau0hg_1, tau0hb_1, tau0ha_1,
             v_abs_2_100, sig_abs_2_100, C_f_2, tau0hg_2, tau0hb_2, tau0ha_2,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/1.)**2
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/1.)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.3, scale=0.05)
            lnprior += log_erfc_prior(fHg_b/fHa_b, mean=.2, scale=0.05)

            # erfc prior. Difference between absorber velocities less than 100 km/s.
            # lnprior += log_erfc_prior(v_abs_1_100-v_abs_2_100, mean=1., scale=1.)

            return lnprior

        return masks, guess, bounds, lnprior



    def model_feii_forbidden(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """For fitting [Fe II] emission in LRDs."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_feii_100': (r'$\sigma_\mathrm{Fe\,II}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe24245': (r'$F(\mathrm{[Fe\,II]\lambda 4245})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe24278': (r'$F(\mathrm{[Fe\,II]\lambda 4278})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe24415': (r'$F(\mathrm{[Fe\,II]\lambda 4415})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe24289': (r'$F(\mathrm{[Fe\,II]\lambda 4289})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe25159': (r'$F(\mathrm{[Fe\,II]\lambda 5159})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe25160': (r'$F(\mathrm{[Fe\,II]\lambda 5160})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe25263': (r'$F(\mathrm{[Fe\,II]\lambda 5263})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fe25270': (r'$F(\mathrm{[Fe\,II]\lambda 5270})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
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
                'bk30': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk31': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        (z_n, sig_n_100, sig_feii_100, fO35007,
         fe24245, fe24278, fe24289, fe24415, fe25159, fe25160, fe25263, fe25270,
         a0, b0, a1, b1, a2, b2, a3, b3) = pars
        w_mum = np.array((
            self.FeII4245, self.FeII4278,
            self.FeII4289, self.FeII4415,
            self.FeII5159, self.FeII5160,
            self.FeII5263, self.FeII5270,
            self.OIII4959, self.OIII5007,
            ))
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*10
                  #+ (np.exp(v_blr_100/self.c_100_kms),)*2
            )
            )
        if print_waves: return w_mum

        fO34959 = fO35007 / 3.05

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_feii_100,)*8
            + (sig_n_100,)*2
            #+ (fwhm_blr_100/self.fwhm2sig,)*2
            )
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0  = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fe24245)
        f1  = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fe24278)
        f2  = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fe24289)
        f3  = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fe24415)
        f4  = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fe25159)
        f5  = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fe25160)
        f6  = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fe25263)
        f7  = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fe25270)
        f8  = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fO34959)
        f9  = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fO35007)

        bk0 = a0 + (self.wave-w_mum[1]) * b0
        bk1 = a1 + (self.wave-w_mum[3]) * b1
        bk2 = a2 + (self.wave-w_mum[8]) * b2
        bk3 = a3 + (self.wave-w_mum[5]) * b3
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)
        bk3 = np.where(self.fit_mask[3], bk3, 0)

        if print_blobs:
            return tuple()
        return f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, bk0, bk1, bk2, bk3

    def model_feii_forbidden_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.42649, .444, .5008, .5215))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.05, 0.05, 0.05, 0.05)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2],
            np.abs(self.wave-pivot_waves[3])<windw_sizes[3])

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg30 = np.nanmedian(self.flux[masks[3]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        bkg31 = np.nanstd(self.flux[masks[3]])
        guess = np.array((
            self.redshift_guess, 0.5, 0.5, 0.1,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21, bkg30, bkg31))
        bounds = np.array((
            (self.redshift_guess-self.dz, 0., 0., 0.,
             0., 0.0, 0., 0., 0., 0., 0., 0.,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100., bkg30-10*bkg31, -100.,),
            (self.redshift_guess+self.dz, 5., 10., 5.,
             1., 1., 1., 1., 1., 1., 1., 1.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
             bkg20+10*bkg21, 100., bkg30+10*bkg31, 100.,
            )))
        #mask = masks[0] | masks[1] | masks[2]
        masks = np.array([
            ~self.mask & mask & np.isfinite(self.flux*self.errs) & (self.errs>0)
            for mask in masks])
        #central_wave = 0.6600*(1+self.redshift_guess)
        #mask = np.abs(self.wave - central_wave)<0.10 # [um]
        #central_pix  = np.argmin(np.abs(self.wave - central_wave))
        #mask[central_pix-15:central_pix+15+1] = True
        #mask = mask & ~self.mask
        #func_name = traceback.extract_stack(None, 2)[-1][2]
        #assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'

        def lnprior(pars, *args):
            (z_n, sig_n_100, sig_feii_100, fO35007,
             fe24245, fe24278, fe24289, fe24415, fe25159, fe25160, fe25263, fe25270,
             a0, b0, a1, b1, a2, b2, a3, b3) = pars

            lnprior = 0.

            return lnprior

        return masks, guess, bounds, lnprior



class jwst_spec_fitter(jwst_spec_models, specfitt.jwst_spec_fitter):
    pass
