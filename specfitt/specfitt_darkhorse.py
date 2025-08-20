import functools
import os
import pickle
import re
import traceback
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, least_squares
from scipy.signal import convolve, medfilt
from scipy.interpolate import CubicSpline, splrep, splev

from astropy import constants, cosmology, table, units
cosmop = cosmology.Planck18

import emcee
import corner

from dust_extinction.parameter_averages import CCM89, G03_SMCBar

from . import spectrum 
from .specfitt_utils import gauss_int2, _g03_, voigt_profile, log_erfc_prior, get_fwhm, mask_line
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



    def model_forbidden(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """For fitting [Fe II] emission in LRDs."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fO23726': (r'$F(\mathrm{[O\,II]\lambda 3726})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_O2_nebular': (r'$F(\mathrm{[O\,II]\lambda 3729})$'+'\n'+r'$/F(\mathrm{[S\,II]\lambda 3726})$', 1., '[---]'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHe15875': (r'$F(\mathrm{He\,I\lambda 5875})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO16300': (r'$F(\mathrm{[O\,I]\lambda 6300})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fS26716': (r'$F(\mathrm{[S\,II]\lambda 6716})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_S2_nebular': (r'$F(\mathrm{[S\,II]\lambda 6731})$'+'\n'+r'$/F(\mathrm{[S\,II]\lambda 6716})$', 1., '[---]'),
                'v_nad_100': (r'$v_\mathrm{Na\,I}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_nad_100': (r'$\sigma_\mathrm{Na\,I}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'C_f_nad': (r'$C_{f}$', 1., '[---]'),
                'tau0_nad': (r'$\tau_\mathrm{Na\,I}$', 1., '[---]'),
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
                'bk40': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk41': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk50': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk51': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk60': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk61': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk70': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk71': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                 }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'fO23729': (r'$F(\mathrm{[O\,II]\lambda 3729})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fS26731': (r'$F(\mathrm{[S\,II]\lambda 6731})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO2_nebular': (r'$F(\mathrm{[O\,II]\lambda\lambda 3726,3729})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fS2_nebular': (r'$F(\mathrm{[S\,II]\lambda\lambda 6716,6731})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO34959': (r'$F(\mathrm{[O\,III]\lambda 4959})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO16363': (r'$F(\mathrm{[O\,I]\lambda 6363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'EW_NaI':      (r'$EW(\mathrm{Na\,I})$', 1., r'$[\AA]$', float),
                }
        (z_n, sig_n_100,
         fO23726, R_O2_nebular, fNe33869,
         fO34363, fO35007, fHe15875, fO16300, fS26716, R_S2_nebular,
         v_nad_100, sig_nad_100, C_f_nad, tau0_nad,
         a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6,
         a7, b7,
        ) = pars
        w_mum = np.array((
            self.OII3726,  self.OII3729,  # 0--1
            self.NeIII3869,               #    2
            self.OIII4363, self.OIII4959, self.OIII5007, # 20--22
            self.HeI5875,  # 23--24
            self.OI6300,   self.OI6363,   # 25--26
            self.SII6716,  self.SII6731,  # 27--28
            self.NaI5890,  self.NaI5896,  # 29--30
            ))

        w_mum = (w_mum * (1.+z_n)
            * np.array(
                  (1.,)*11
                  + (np.exp(v_nad_100/self.c_100_kms),)*2
                  )
                  #+ (np.exp(v_blr_100/self.c_100_kms),)*2
		    )
        if print_waves: return w_mum

        fO34959 = fO35007 / 3.05
        fS26731  = R_S2_nebular * fS26716
        fO16363 = fO16300 / 3.05
        fO23729  = R_O2_nebular * fO23726

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*11
            + (sig_nad_100,)*2
            #+ (fwhm_blr_100/self.fwhm2sig,)*2
            )
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0   = gauss_int2(self.wave, mu=w_mum[ 0], sig=sig_mum[ 0], flux=fO23726)
        f1   = gauss_int2(self.wave, mu=w_mum[ 1], sig=sig_mum[ 1], flux=fO23729)
        f2   = gauss_int2(self.wave, mu=w_mum[ 2], sig=sig_mum[ 2], flux=fNe33869)
        f3   = gauss_int2(self.wave, mu=w_mum[ 3], sig=sig_mum[ 3], flux=fO34363)
        f4   = gauss_int2(self.wave, mu=w_mum[ 4], sig=sig_mum[ 4], flux=fO34959)
        f5   = gauss_int2(self.wave, mu=w_mum[ 5], sig=sig_mum[ 5], flux=fO35007)
        f6   = gauss_int2(self.wave, mu=w_mum[ 6], sig=sig_mum[ 6], flux=fHe15875)
        f7   = gauss_int2(self.wave, mu=w_mum[ 7], sig=sig_mum[ 7], flux=fO16300)
        f8   = gauss_int2(self.wave, mu=w_mum[ 8], sig=sig_mum[ 8], flux=fO16363)
        f9   = gauss_int2(self.wave, mu=w_mum[ 9], sig=sig_mum[ 9], flux=fS26716)
        f10  = gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=fS26731)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[11]
        tau_nad_blue = 2. * tau0_nad * gauss_int2(
            self.wave, mu=w_mum[11], sig=sig_mum[11], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[12]
        tau_nad_red  = 1. * tau0_nad * gauss_int2(
            self.wave, mu=w_mum[12], sig=sig_mum[12], flux=tau0_norm)
        absr_nad = 1. - C_f_nad + C_f_nad * np.exp(-tau_nad_blue -tau_nad_red)

        bk0 = a0 + (self.wave-w_mum[0])  * b0
        bk1 = a1 + (self.wave-w_mum[2])  * b1
        bk2 = a2 + (self.wave-w_mum[3])  * b2
        bk3 = a3 + (self.wave-w_mum[4])  * b3
        bk4 = a4 + (self.wave-w_mum[5]) * b4
        bk5 = a5 + (self.wave-w_mum[6]) * b5
        bk6 = a6 + (self.wave-w_mum[7]) * b6
        bk7 = a7 + (self.wave-w_mum[9]) * b7

        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)
        bk3 = np.where(self.fit_mask[3], bk3, 0)
        bk4 = np.where(self.fit_mask[4], bk4, 0)
        bk5 = np.where(self.fit_mask[5], bk5, 0)
        bk6 = np.where(self.fit_mask[6], bk6, 0)
        bk7 = np.where(self.fit_mask[7], bk7, 0)

        if print_blobs:
            mask_nad  = (
                self.fit_mask[5] & (np.abs(self.wave-w_mum[11])/sig_mum[11]<6)
                )
            _ew_cont_ = (self.spline_model_cont + bk5)*absr_nad
            _ew_cont_ = 1. - _ew_cont_/(self.spline_model_cont + bk5)
            dw = np.gradient(self.wave[mask_nad])
            _ew_cont_ = np.sum(_ew_cont_[mask_nad]*dw)*1e4 # To [AA]
            ew_nad = _ew_cont_ / ((1+z_n)*np.exp((v_nad_100)/self.c_100_kms)) # Rest frame
            return (fO23729, fS26731,
                fO23726+fO23729, fS26716+fS26731, 
                fO34959, fO16363, ew_nad)
        return (
            f0, f1,
            f2, f3, f4, f5, f6, f7, f8, f9, f10,
            bk0, bk1, bk2, bk3, bk4, bk5*absr_nad,
            bk6, bk7, self.spline_model_cont*absr_nad,
            )

    def model_forbidden_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""

        self.model_forbidden_get_spline_cont_fit_init()

        backup_flux = np.copy(self.flux)
        self.flux -= self.spline_model_cont

        lwave = np.array((
            .3728, .3870, .43416, .48626, .5008,
            .5850, .6350, .6725))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.03, 0.02, 0.08, 0.05, 0.05,
            0.11, 0.05, 0.05)

        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2],
            np.abs(self.wave-pivot_waves[3])<windw_sizes[3],
            np.abs(self.wave-pivot_waves[4])<windw_sizes[4],
            np.abs(self.wave-pivot_waves[5])<windw_sizes[5],
            np.abs(self.wave-pivot_waves[6])<windw_sizes[6],
            np.abs(self.wave-pivot_waves[7])<windw_sizes[7],
        ]
        masks[2] = (
            masks[2]
            & (np.abs(self.wave-pivot_waves[2])>0.006)
            )
        masks[3] = (
            masks[3]
            & (np.abs(self.wave-pivot_waves[3])>0.006)
            )

        bkg_0 = [
            np.nanmedian(self.flux[mask]) for mask in masks]
        bkg_1 = [
            np.nanstd(   self.flux[mask]) for mask in masks]
        bkg_0 = np.where(np.isfinite(bkg_0),
            bkg_0, np.nanmedian(bkg_0)) # Replace invalid intervals with median
        bkg_1 = np.where(np.isfinite(bkg_1),
            bkg_1, np.nanmedian(bkg_1)) # Replace invalid intervals with median

        bkg_guess = np.empty((bkg_0.size+bkg_1.size))
        bkg_guess[0::2] = bkg_0
        bkg_guess[1::2] = bkg_1
        bkg_mins = np.empty((bkg_0.size+bkg_1.size))
        bkg_mins[0::2] = bkg_0 - 10.*bkg_1
        bkg_mins[1::2] = -100.
        bkg_maxs = np.empty((bkg_0.size+bkg_1.size))
        bkg_maxs[0::2] = bkg_0 + 10.*bkg_1
        bkg_maxs[1::2] = 100.

        self.flux = backup_flux

        guess = np.hstack((
            (self.redshift_guess, 0.5,
             0.01, 0.700, 0.01,
             0.01, 0.01, 0.01, 0.01,
             0.01, 1.00,
             0.,   0.5, 0.5, 0.5,),
            bkg_guess))
        bounds = np.array((
            np.hstack((
                (self.redshift_guess-self.dz, 0.,
                 0., 0.337, 0.,
                 0., 0., 0., 0.,
                 0., 0.6773,
                 -1., 0.01, 0.0, 0.0,
                ),
                bkg_mins)),
            np.hstack(((
                (self.redshift_guess+self.dz, 5.,
                 5., 1.510, 5.,
                 5., 5., 5., 5.,
                 5., 2.372,
                 1., 10., 1.0, 15.0,
                ),
                bkg_maxs)),
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
            (z_n, sig_n_100,
             fO23726, R_O2_nebular, fNe33869,
             fO34363, fO35007, fHe15875, fO16300, fS26716, R_S2_nebular,
             v_nad_100, sig_nad_100, C_f_nad, tau0_nad,
             a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6,
             a7, b7,
            ) = pars

            lnprior = 0.

            return lnprior

        return masks, guess, bounds, lnprior

    def model_forbidden_get_spline_cont_fit_init(self):
        dv = 350*units.km/units.s
        db = 5000*units.km/units.s
        AA = units.AA
        fitting_region = (
            (self.errs>0) & np.isfinite(self.errs*self.flux)
            & mask_line(self.wave, 3727.0*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 3870.1*AA, 1.5*dv, self.redshift_guess)
            & mask_line(self.wave, 4069.7*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4078.0*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4179.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4245.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4267.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4277.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4287.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4306.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4359.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4365.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4415.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4417.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4815.9*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4862.0*AA, db, self.redshift_guess)
            & mask_line(self.wave, 4891.0*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4906.7*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 5044.9*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 5159.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 5160.2*AA, 2.3*dv, self.redshift_guess)
            & mask_line(self.wave, 5263.9*AA, 2.3*dv, self.redshift_guess)
            & mask_line(self.wave, 4960.3*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 5008.2*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 5756.2*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 5877.2*AA, 2*dv, self.redshift_guess)
            & mask_line(self.wave, 5892.0*AA, 4*dv, self.redshift_guess)
            & mask_line(self.wave, 5897.6*AA, 4*dv, self.redshift_guess)
            & mask_line(self.wave, 6302.0*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6365.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6549.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6564.5*AA, db, self.redshift_guess)
            & mask_line(self.wave, 6584.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6717.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6732.5*AA, 2.0*dv, self.redshift_guess)
            )
        tckKnown = splrep(# s controls smoothness (higher = smoother)
            self.wave[fitting_region], self.flux[fitting_region], s=5)
        self.spline_model_cont = splev(
            self.wave[fitting_region], tckKnown)
        model_smooth = medfilt(self.spline_model_cont, 15)
        model_interp = np.interp(
            self.wave, 
            self.wave[fitting_region], model_smooth)
        model_interp[fitting_region] = self.spline_model_cont
        self.spline_model_cont = model_interp
        # (0) Spline
        #tck = splrep(wave[fitting_region], flux[fitting_region], s=0.25)
        #cubesview_utils.plot_badpixels_regions(
        #    self.wave, axes=ax0, facecolor='silver', edgecolor='none',
        #    mask=fitting_region, alpha=0.3)
        plotting_region = (
            (self.wave>=self.wave[fitting_region][0])
            & (self.wave<=self.wave[fitting_region][-1])
            )
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10))
        ax0.step(
            self.wave[plotting_region], self.flux[plotting_region],
            label='$\mathrm{Data}$', where='mid', color='k', ls='-')
        ax0.step(
            self.wave[plotting_region], self.spline_model_cont[plotting_region],
            color='crimson', ls='-', label='$\mathrm{Model}$', where='mid')
        ax0.semilogy()
        ax1.step(
            self.wave[plotting_region],
            self.flux[plotting_region]-self.spline_model_cont[plotting_region],
            label='$\mathrm{Data}$', where='mid', color='k', ls='-')
        line_waves = (
            3727.1, 3729.9, 3869.9,
            4069.7, 4078., 4179., 4245., 4267.,  4277.,  4287.,  4306., 
            4359.,  4365.,  4415.,  4417.5, 4815.9, 4891.0, 4906.7,
            5044.9, 5159.5, 5160.2, 5263.9, 4960.3, 5008.2,
            5756.2, 5274.8, 5877.2, 5891.5, 5897.6, 6302.0, 6365.5, 6718.2, 6732.7,
        )
        for ax in (ax0,ax1):
            for lw in line_waves:
                lw = lw*(1+self.redshift_guess)/1.e4
                ax.axvline(lw, color='grey', ls='--', alpha=0.5)
        std = np.nanpercentile(
            self.flux[plotting_region]-self.spline_model_cont[plotting_region],
            (16, 84))
        std = 3*(std[1]-std[0])/2.
        ax1.set_ylim(-std, std)
        ax1.axhline(0., ls='--', color='grey', alpha=0.5)
        plt.subplots_adjust(hspace=0.0)
        plt.savefig(f'{self.name}_test_cont.pdf', bbox_inches='tight',
            pad_inches=0.01)

        plt.close(fig)
        #ax0.plot(
        #    wave[fitting_region], splev(wave[fitting_region], tck),
        #    color='crimson', ls='--', lw=2.0, label='$\mathrm{Smooth\,Data}$')
        return



    def model_all_lines(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """For fitting [Fe II] emission in LRDs."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO23726': (r'$F(\mathrm{[O\,II]\lambda 3726})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_O2_nebular': (r'$F(\mathrm{[O\,II]\lambda 3729})$'+'\n'+r'$/F(\mathrm{[S\,II]\lambda 3726})$', 1., '[---]'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHe15875': (r'$F(\mathrm{He\,I\lambda 5875})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO16300': (r'$F(\mathrm{[O\,I]\lambda 6300})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fS26716': (r'$F(\mathrm{[S\,II]\lambda 6716})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_S2_nebular': (r'$F(\mathrm{[S\,II]\lambda 6731})$'+'\n'+r'$/F(\mathrm{[S\,II]\lambda 6716})$', 1., '[---]'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}     ]$'),
                'fHgamma_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\gamma})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'tau_thom': (r'$\tau_\mathrm{BLR}$', 1., '[---]'),
                'T_thom': (r'$T$', 1., r'$[10^4\,\mathrm{K}]$'),
                'v_Ha_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$     '),
                'sig_Ha_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^     {-1}}]$'),
                'C_f_Ha': (r'$C_{f}$', 1., '[---]'),
                'tau0_Ha': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'v_nad_100': (r'$v_\mathrm{Na\,I}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_nad_100': (r'$\sigma_\mathrm{Na\,I}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'C_f_nad': (r'$C_{f}$', 1., '[---]'),
                'tau0_nad': (r'$\tau_\mathrm{Na\,I}$', 1., '[---]'),
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
                'bk40': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk41': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk50': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk51': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk60': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk61': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk70': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk71': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                 }
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'fO23729': (r'$F(\mathrm{[O\,II]\lambda 3729})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHgamma': (r'$F_\mathrm{n}(\mathrm{H\gamma})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta':  (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fS26731': (r'$F(\mathrm{[S\,II]\lambda 6731})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO2_nebular': (r'$F(\mathrm{[O\,II]\lambda\lambda 3726,3729})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fS2_nebular': (r'$F(\mathrm{[S\,II]\lambda\lambda 6716,6731})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO34959': (r'$F(\mathrm{[O\,III]\lambda 4959})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO16363': (r'$F(\mathrm{[O\,I]\lambda 6363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'EW_Ha_cont':  (r'$EW(\mathrm{H\alpha;\,cont.})$', 1., r'$[\AA]$', float),
                'EW_Ha_blr':   (r'$EW(\mathrm{H\alpha;\,BLR})$', 1., r'$[\AA]$', float),
                'EW_NaI':      (r'$EW(\mathrm{Na\,I})$', 1., r'$[\AA]$', float),
                'logSFR_Ha'    : (r'$\log\,SFR(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{M_\odot\,yr^{-1}}]$', float),
                'log_L_Ha_b_ism'   : (r'$\log\,L_\mathrm{b}(\mathrm{H\alpha})$', 1.,
                            r'$[\mathrm{10^{42}\,erg\,s^{-1}}]$', float),
                'logMBH'      : (r'$\log\,(M_\bullet)$', 1.,
                            r'$[\mathrm{M_\odot}]$', float),
                'lEddMBH'     : (r'$\lambda_\mathrm{Edd}$', 1.,
                            '[---]', float),
                }
        (z_n, sig_n_100, Av,
         fO23726, R_O2_nebular, fNe33869,
         fO34363, fO35007, fHe15875, fO16300, fHalpha, fN26583, fS26716, R_S2_nebular,
         v_blr_100, fwhm_blr_100, fHgamma_blr, fHbeta_blr, fHalpha_blr,
         tau_thom, T_thom,
         v_Ha_100, sig_Ha_100, C_f_Ha, tau0_Ha,
         v_nad_100, sig_nad_100, C_f_nad, tau0_nad,
         a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6,
         a7, b7,
        ) = pars
        w_mum = np.array((
            self.OII3726,  self.OII3729,  # 0--1
            self.NeIII3869,               #    2
            self.Hgamma, self.OIII4363,
            self.Hbeta, self.OIII4959, self.OIII5007, # 20--22
            self.HeI5875,  # 23--24
            self.OI6300,   self.OI6363,   # 25--26
            self.NII6548, self.Halpha, self.NII6583,
            self.SII6716,  self.SII6731,  # 27--28
            self.Hgamma, self.Hbeta, self.Halpha,
            self.Halpha,
            self.NaI5890,  self.NaI5896,  # 29--30
            ))

        fHalpha_intr = fHalpha / _g03_(self.Halpha, Av)
        atten_Hgamma, atten_Hbeta = _g03_(np.array((self.Hgamma, self.Hbeta)), Av)


        fHbeta  = fHalpha_intr/self.Ha2Hb
        fHgamma = fHbeta * self.Hg2Hb
        fHgamma *= atten_Hgamma
        fHbeta  *= atten_Hbeta
        
        w_mum = (w_mum * (1.+z_n)
            * np.array(
                  (1.,)*16
                  + (np.exp(v_blr_100/self.c_100_kms),)*3
                  + (np.exp(v_Ha_100/self.c_100_kms),)*1
                  + (np.exp(v_nad_100/self.c_100_kms),)*2
                  )
            )
        if print_waves: return w_mum

        fO34959 = fO35007 / 2.98
        fS26731  = R_S2_nebular * fS26716
        fO16363 = fO16300 / 3.13
        fO23729  = R_O2_nebular * fO23726
        fN26548 = fN26583 / 3.05

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*16
            + (fwhm_blr_100/self.fwhm2sig,)*3
            + (sig_Ha_100,) + (sig_nad_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0   = gauss_int2(self.wave, mu=w_mum[ 0], sig=sig_mum[ 0], flux=fO23726)
        f1   = gauss_int2(self.wave, mu=w_mum[ 1], sig=sig_mum[ 1], flux=fO23729)
        f2   = gauss_int2(self.wave, mu=w_mum[ 2], sig=sig_mum[ 2], flux=fNe33869)
        f3   = gauss_int2(self.wave, mu=w_mum[ 3], sig=sig_mum[ 3], flux=fHgamma)
        f4   = gauss_int2(self.wave, mu=w_mum[ 4], sig=sig_mum[ 4], flux=fO34363)
        f5   = gauss_int2(self.wave, mu=w_mum[ 5], sig=sig_mum[ 5], flux=fHbeta)
        f6   = gauss_int2(self.wave, mu=w_mum[ 6], sig=sig_mum[ 6], flux=fO34959)
        f7   = gauss_int2(self.wave, mu=w_mum[ 7], sig=sig_mum[ 7], flux=fO35007)
        f8   = gauss_int2(self.wave, mu=w_mum[ 8], sig=sig_mum[ 8], flux=fHe15875)
        f9   = gauss_int2(self.wave, mu=w_mum[ 9], sig=sig_mum[ 9], flux=fO16300)
        f10  = gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=fO16363)
        f11  = gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=fN26548)
        f12  = gauss_int2(self.wave, mu=w_mum[12], sig=sig_mum[12], flux=fHalpha)
        f13  = gauss_int2(self.wave, mu=w_mum[13], sig=sig_mum[13], flux=fN26583)
        f14  = gauss_int2(self.wave, mu=w_mum[14], sig=sig_mum[14], flux=fS26716)
        f15  = gauss_int2(self.wave, mu=w_mum[15], sig=sig_mum[15], flux=fS26731)
        f16  = gauss_int2(self.wave, mu=w_mum[16], sig=sig_mum[16], flux=fHgamma_blr)
        f17  = gauss_int2(self.wave, mu=w_mum[17], sig=sig_mum[17], flux=fHbeta_blr)
        f18  = gauss_int2(self.wave, mu=w_mum[18], sig=sig_mum[18], flux=fHalpha_blr)

        # Ha absorber.
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[19] 
        tau_Ha = tau0_Ha * gauss_int2(self.wave, mu=w_mum[19], sig=sig_mum[19], flux=tau0_norm)
        absr_Ha = 1. - C_f_Ha + C_f_Ha * np.exp(-tau_Ha)

        # NaI absorber.
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[20]
        tau_nad_blue = 2. * tau0_nad * gauss_int2(
            self.wave, mu=w_mum[20], sig=sig_mum[20], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[21]
        tau_nad_red  = 1. * tau0_nad * gauss_int2(
            self.wave, mu=w_mum[21], sig=sig_mum[21], flux=tau0_norm)
        absr_nad = 1. - C_f_nad + C_f_nad * np.exp(-tau_nad_blue -tau_nad_red)


        W_kms = (428. * tau_thom + 370.) * np.sqrt(T_thom)

        # Scatter Hg.
        W_mum = W_kms/self.c_kms * w_mum[16]
        dw = np.argmin(np.abs(self.wave-w_mum[16]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f19 = convolve(f16, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f16 *= (1. - f_scatt)
        f19 *= f_scatt

        # Scatter Hb.
        W_mum = W_kms/self.c_kms * w_mum[17]
        dw = np.argmin(np.abs(self.wave-w_mum[17]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f20 = convolve(f17, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f17 *= (1. - f_scatt)
        f20 *= f_scatt
       
        # Scatter Ha.
        W_mum = W_kms/self.c_kms * w_mum[18]
        dw = np.argmin(np.abs(self.wave-w_mum[18]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f21 = convolve(f18, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f18 *= (1. - f_scatt)
        f21 *= f_scatt

        bk0 = a0 + (self.wave-w_mum[0])  * b0
        bk1 = a1 + (self.wave-w_mum[2])  * b1
        bk2 = a2 + (self.wave-w_mum[3])  * b2
        bk3 = a3 + (self.wave-w_mum[5])  * b3
        bk4 = a4 + (self.wave-w_mum[8])  * b4
        bk5 = a5 + (self.wave-w_mum[9])  * b5
        bk6 = a6 + (self.wave-w_mum[12]) * b6
        bk7 = a7 + (self.wave-w_mum[14]) * b7

        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)
        bk3 = np.where(self.fit_mask[3], bk3, 0)
        bk4 = np.where(self.fit_mask[4], bk4, 0)
        bk5 = np.where(self.fit_mask[5], bk5, 0)
        bk6 = np.where(self.fit_mask[6], bk6, 0)
        bk7 = np.where(self.fit_mask[7], bk7, 0)

        if print_blobs:

            mask_ha = (
                self.fit_mask[6] & (np.abs(self.wave-w_mum[18])/sig_mum[18]<10)
                )
            _ew_cont_ = (self.spline_model_cont + bk6)*absr_Ha
            _ew_cont_ = 1. - _ew_cont_/(self.spline_model_cont + bk6)
            dw = np.gradient(self.wave[mask_ha])
            _ew_cont_ = np.sum(_ew_cont_[mask_ha]*dw)*1e4 # To [AA]
            ew_ha_cont = _ew_cont_ / ((1+z_n)*np.exp((v_Ha_100)/self.c_100_kms)) # Rest frame

            mask_ha = (
                self.fit_mask[6] & (np.abs(self.wave-w_mum[18])/sig_mum[18]<10)
                )
            _ew_cont_ = (f18 + f21 + self.spline_model_cont + bk6)*absr_Ha
            _ew_cont_ = 1. - _ew_cont_/(self.spline_model_cont + bk6 + f18 + f21)
            dw = np.gradient(self.wave[mask_ha])
            _ew_cont_ = np.sum(_ew_cont_[mask_ha]*dw)*1e4 # To [AA]
            ew_ha_blr = _ew_cont_ / ((1+z_n)*np.exp((v_Ha_100)/self.c_100_kms)) # Rest frame

            mask_nad  = (
                self.fit_mask[4] & (np.abs(self.wave-w_mum[20])/sig_mum[20]<6)
                )
            _ew_cont_ = (self.spline_model_cont + bk4)*absr_nad
            _ew_cont_ = 1. - _ew_cont_/(self.spline_model_cont + bk4)
            dw = np.gradient(self.wave[mask_nad])
            _ew_cont_ = np.sum(_ew_cont_[mask_nad]*dw)*1e4 # To [AA]
            ew_nad = _ew_cont_ / ((1+z_n)*np.exp((v_nad_100)/self.c_100_kms)) # Rest frame
            lum_fact = (4 * np.pi * cosmop.luminosity_distance(z_n)**2)
                
            L_Ha_n = fHalpha_intr * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            L_Ha_n = (L_Ha_n * lum_fact).to('1e42 erg/s')
            SFR_Ha = (L_Ha_n * self.SFR_to_L_Ha_Sh23).to('Msun/yr').value              
            logSFR_Ha = np.log10(SFR_Ha)
            L_Ha_b_ism = fHalpha_blr * 1e2 * units.Unit('1e-18 erg/(s cm2)')
            L_Ha_b_ism /= _g03_(self.Halpha, Av)
            log_L_Ha_b_ism = np.log10((L_Ha_b_ism * lum_fact).to('1e42 erg/s').value)
            logMBH = 6.6 + 0.47*log_L_Ha_b_ism + 2.06*np.log10(fwhm_blr_100/10.)                         
            edrat = 130 * 10**log_L_Ha_b_ism * units.Unit('1e42 erg/s')
            lEdd = 4.*np.pi*constants.G*constants.m_p*constants.c/constants.sigma_T
            lEdd = lEdd*10**logMBH*units.Msun
            lEddMBH = (edrat/lEdd).to(1).value

            return (fO23729, fHgamma, fHbeta, fS26731,
                fO23726+fO23729, fS26716+fS26731, 
                fO34959, fO16363, ew_ha_cont, ew_ha_blr, ew_nad,
                logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH,
                )
        return (
            f0, f1,
            f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15,
            f16, f17, f18*absr_Ha, f19, f20, f21*absr_Ha,
            bk0, bk1, bk2, bk3, bk4*absr_nad, bk5, bk6*absr_Ha,
            bk7, self.spline_model_cont*absr_nad*absr_Ha,
            )

    def model_all_lines_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""

        self.model_all_lines_get_spline_cont_fit_init()

        backup_flux = np.copy(self.flux)
        self.flux -= self.spline_model_cont

        lwave = np.array((
            .3728, .3870, .43416, .4900,
            .5850, .6350, .6565, .6725))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.03, 0.02, 0.08, 0.15,
            0.11, 0.05, 0.12, 0.05)

        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2],
            np.abs(self.wave-pivot_waves[3])<windw_sizes[3],
            np.abs(self.wave-pivot_waves[4])<windw_sizes[4],
            np.abs(self.wave-pivot_waves[5])<windw_sizes[5],
            np.abs(self.wave-pivot_waves[6])<windw_sizes[6],
            np.abs(self.wave-pivot_waves[7])<windw_sizes[7],
        ]
        """
        masks[2] = (
            masks[2]
            & (np.abs(self.wave-pivot_waves[2])>0.006)
            )
        masks[3] = (
            masks[3]
            & (np.abs(self.wave-pivot_waves[3])>0.006)
            )
        """

        bkg_0 = [
            np.nanmedian(self.flux[mask]) for mask in masks]
        bkg_1 = [
            np.nanstd(   self.flux[mask]) for mask in masks]
        bkg_0 = np.where(np.isfinite(bkg_0),
            bkg_0, np.nanmedian(bkg_0)) # Replace invalid intervals with median
        bkg_1 = np.where(np.isfinite(bkg_1),
            bkg_1, np.nanmedian(bkg_1)) # Replace invalid intervals with median

        bkg_guess = np.empty((bkg_0.size+bkg_1.size))
        bkg_guess[0::2] = bkg_0
        bkg_guess[1::2] = bkg_1
        bkg_mins = np.empty((bkg_0.size+bkg_1.size))
        bkg_mins[0::2] = bkg_0 - 10.*bkg_1
        bkg_mins[1::2] = -100.
        bkg_maxs = np.empty((bkg_0.size+bkg_1.size))
        bkg_maxs[0::2] = bkg_0 + 10.*bkg_1
        bkg_maxs[1::2] = 100.

        self.flux = backup_flux

        guess = np.hstack((
            (self.redshift_guess, 0.5, 0.1,
             0.01, 0.700, 0.01,
             0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
             0.01, 1.00,
             0., 15., 0.08, 0.15, 1.5, 0.5, 1.,
             0.,   0.5, 0.5, 0.5,
             0.,   0.5, 0.5, 0.5,),
            bkg_guess))
        bounds = np.array((
            np.hstack((
                (self.redshift_guess-self.dz, 0., 0.,
                 0., 0.337, 0.,
                 0., 0., 0., 0., 0., 0.,
                 0., 0.6773,
                 -5., .7, 0., 0., 0.001, 0., 0.1,
                 -1., 0.01, 0.0, 0.0,
                 -1., 0.01, 0.0, 0.0,
                ),
                bkg_mins)),
            np.hstack(((
                (self.redshift_guess+self.dz, 5., 3.,
                 5., 1.510, 5.,
                 5., 5., 5., 5., 5., 5.,
                 5., 2.372,
                 5., 70., 1., 1., 5., 30., 10.,
                 1., 10., 1.0, 15.0,
                 1., 10., 1.0, 15.0,
                ),
                bkg_maxs)),
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
            (z_n, sig_n_100, Av,
             fO23726, R_O2_nebular, fNe33869,
             fO34363, fO35007, fHe15875, fO16300, fHalpha, fN26583, fS26716, R_S2_nebular,
             v_blr_100, fwhm_blr_100, fHgamma_blr, fHbeta_blr, fHalpha_blr,
             tau_thom, T_thom,
             v_Ha_100, sig_Ha_100, C_f_Ha, tau0_Ha,
             v_nad_100, sig_nad_100, C_f_nad, tau0_nad,
             a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6,
             a7, b7,
            ) = pars

            lnprior = 0.

            return lnprior

        return masks, guess, bounds, lnprior

    def model_all_lines_get_spline_cont_fit_init(self):
        dv = 350*units.km/units.s
        db = 7000*units.km/units.s
        AA = units.AA
        fitting_region = (
            (self.errs>0) & np.isfinite(self.errs*self.flux)
            & mask_line(self.wave, 3727.0*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 3870.1*AA, 1.5*dv, self.redshift_guess)
            & mask_line(self.wave, 4069.7*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4078.0*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4179.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4245.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4267.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4277.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4287.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4306.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4340.*AA,  db, self.redshift_guess)
            & mask_line(self.wave, 4359.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4365.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4415.*AA,  dv, self.redshift_guess)
            & mask_line(self.wave, 4417.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4815.9*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4862.0*AA, db, self.redshift_guess)
            & mask_line(self.wave, 4891.0*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4906.7*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 4960.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 5008.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 5044.9*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 5159.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 5160.2*AA, 2.3*dv, self.redshift_guess)
            & mask_line(self.wave, 5263.9*AA, 2.3*dv, self.redshift_guess)
            & mask_line(self.wave, 4960.3*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 5008.2*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 5756.2*AA, 3*dv, self.redshift_guess)
            & mask_line(self.wave, 5877.2*AA, 2*dv, self.redshift_guess)
            & mask_line(self.wave, 5892.0*AA, 4*dv, self.redshift_guess)
            & mask_line(self.wave, 5897.6*AA, 4*dv, self.redshift_guess)
            & mask_line(self.wave, 6302.0*AA, 2.*dv, self.redshift_guess)
            & mask_line(self.wave, 6365.5*AA, 2.*dv, self.redshift_guess)
            & mask_line(self.wave, 6549.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6564.5*AA, db, self.redshift_guess)
            & mask_line(self.wave, 6584.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6717.5*AA, dv, self.redshift_guess)
            & mask_line(self.wave, 6732.5*AA, 2.0*dv, self.redshift_guess)
            )
        tckKnown = splrep(# s controls smoothness (higher = smoother)
            self.wave[fitting_region], self.flux[fitting_region], s=5)
        self.spline_model_cont = splev(
            self.wave[fitting_region], tckKnown)

        model_smooth = medfilt(self.spline_model_cont, 15)
        model_interp = np.interp(
            self.wave, 
            self.wave[fitting_region], model_smooth)
        model_interp[fitting_region] = self.spline_model_cont
        self.spline_model_cont = model_interp
        # (0) Spline
        #tck = splrep(wave[fitting_region], flux[fitting_region], s=0.25)
        #cubesview_utils.plot_badpixels_regions(
        #    self.wave, axes=ax0, facecolor='silver', edgecolor='none',
        #    mask=fitting_region, alpha=0.3)
        plotting_region = (
            (self.wave>=self.wave[fitting_region][0])
            & (self.wave<=self.wave[fitting_region][-1])
            )
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10))
        ax0.step(
            self.wave[plotting_region], self.flux[plotting_region],
            label='$\mathrm{Data}$', where='mid', color='k', ls='-')
        ax0.step(
            self.wave[plotting_region], self.spline_model_cont[plotting_region],
            color='crimson', ls='-', label='$\mathrm{Model}$', where='mid')
        ax0.semilogy()
        ax1.step(
            self.wave[plotting_region],
            self.flux[plotting_region]-self.spline_model_cont[plotting_region],
            label='$\mathrm{Data}$', where='mid', color='k', ls='-')
        line_waves = (
            3727.1, 3729.9, 3869.9,
            4069.7, 4078., 4179., 4245., 4267.,  4277.,  4287.,  4306., 
            4359.,  4365.,  4415.,  4417.5, 4815.9, 4891.0, 4906.7,
            5044.9, 5159.5, 5160.2, 5263.9, 4960.3, 5008.2,
            5756.2, 5274.8, 5877.2, 5891.5, 5897.6, 6302.0, 6365.5, 6564.5,
            6718.2, 6732.7,
        )
        for ax in (ax0,ax1):
            for lw in line_waves:
                lw = lw*(1+self.redshift_guess)/1.e4
                ax.axvline(lw, color='grey', ls='--', alpha=0.5)
        std = np.nanpercentile(
            self.flux[plotting_region]-self.spline_model_cont[plotting_region],
            (16, 84))
        std = 3*(std[1]-std[0])/2.
        ax1.set_ylim(-std, std)
        ax1.axhline(0., ls='--', color='grey', alpha=0.5)
        plt.subplots_adjust(hspace=0.0)
        plt.savefig(f'{self.name}_test_cont.pdf', bbox_inches='tight',
            pad_inches=0.01)

        plt.close(fig)
        #ax0.plot(
        #    wave[fitting_region], splev(wave[fitting_region], tck),
        #    color='crimson', ls='--', lw=2.0, label='$\mathrm{Smooth\,Data}$')
        return


    def model_feii_forbidden_A(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """For fitting [Fe II] emission in LRDs."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_feii_100': (r'$\sigma_\mathrm{Fe\,II}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fFe24179': (r'$F(\mathrm{[Fe\,II]\lambda 4179})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fFe24245': (r'$F(\mathrm{[Fe\,II]\lambda 4245})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fFe24266': (r'$F(\mathrm{[Fe\,II]\lambda 4266})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fFe24277': (r'$F(\mathrm{[Fe\,II]\lambda 4277})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fFe24287': (r'$F(\mathrm{[Fe\,II]\lambda 4287})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
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
                }
        (z_n, sig_feii_100,
         fFe24179, fFe24245, fFe24266, fFe24277, fFe24287,
         a0, b0, a1, b1) = pars
        w_mum = np.array((
            self.FeII4179, self.FeII4245, self.FeII4266,
            self.FeII4277, self.FeII4287,
            ))

        w_mum = (w_mum * (1.+z_n)
            * np.array(
                  (1.,)*5
                  )
                  #+ (np.exp(v_blr_100/self.c_100_kms),)*2
		    )
        if print_waves: return w_mum

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_feii_100,)*5
            #+ (fwhm_blr_100/self.fwhm2sig,)*2
            )
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0   = gauss_int2(self.wave, mu=w_mum[ 0], sig=sig_mum[ 0], flux=fFe24179)
        f1   = gauss_int2(self.wave, mu=w_mum[ 1], sig=sig_mum[ 1], flux=fFe24245)
        f2   = gauss_int2(self.wave, mu=w_mum[ 2], sig=sig_mum[ 2], flux=fFe24266)
        f3   = gauss_int2(self.wave, mu=w_mum[ 3], sig=sig_mum[ 3], flux=fFe24277)
        f4   = gauss_int2(self.wave, mu=w_mum[ 4], sig=sig_mum[ 4], flux=fFe24287)

        bk0 = a0 + (self.wave-w_mum[1])  * b0
        bk1 = a1 + (self.wave-w_mum[3])  * b1
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)

        if print_blobs:
            return tuple()
        return (
            f0, f1, f2, f3, f4,
            bk0, bk1
            )

    def model_feii_forbidden_A_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.4180, .4277,))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.02, 0.035)

        masks = [
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
        ]

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        guess = np.array((
            self.redshift_guess, 0.5,
            0.01, 0.01, 0.01, 0.01, 0.01,
            bkg00, bkg01, bkg10, bkg11
            ))
        bounds = np.array((
            (self.redshift_guess-self.dz, 0.,
             0., 0., 0., 0., 0.,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             ),
            (self.redshift_guess+self.dz, 10.,
             5., 5., 5., 5., 5.,
             bkg00+10*bkg01, 100., bkg10+10*bkg11, 100.,
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
            (z_n, sig_feii_100,
             fFe24179, fFe24245, fFe24266, fFe24277, fFe24287,
             a0, b0, a1, b1) = pars

            lnprior = 0.

            return lnprior

        return masks, guess, bounds, lnprior



class jwst_spec_fitter(jwst_spec_models, specfitt.jwst_spec_fitter):
    pass
