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

__all__ = ['jwst_spec_models', 'jwst_spec_fitter']

class jwst_spec_models(spectrum.jwst_spec):
    Lyalpha   = .121567
    CIV1548   = .154819
    CIV1551   = .155077
    HeII1640  = .164048
    CIII1907  = .190668
    CIII1909  = .190873
    OII3726   = .37271
    OII3729   = .3729875
    OII3727   = .37285
    NeIII3869 = .386986
    Hzeta     = .389011
    NeIII3968 = .396859
    Hepsilon  = .3971198      
    SII4069   = .4069750      
    SII4076   = .407750  
    Hdelta    = .41028598
    Hgamma    = .43416
    OIII4363  = .43644
    HeII4686  = .46867
    Hbeta     = .48626
    OIII4959  = .49603
    OIII5007  = .50082
    HeI5875   = .5877249
    NaI5890   = .5891583
    NaI5896   = .5897558
    OI6300    = .6302046
    OI6363    = .6365535
    NII6548   = .654986
    Halpha    = .65645
    NII6583   = .6585273
    SII6716   = .6718295
    SII6731   = .6732674
    HeI7065   = .7067138
    OII7325   = .7325000
    OII7319   = .732094
    OII7320   = .732201
    OII7330   = .733168
    OII7331   = .733275
    SIII9069  = .90711
    SIII9532  = .95332
    HeI10829  = 1.0832057
    Pagamma   = 1.094097787
    FeII4180  = .4180143
    FeII4245  = .4245169
    FeII4268  = .4267548
    FeII4278  = .4278041
    FeII4289  = .4288599
    FeII4233  = .4232755
    FeII4307  = .4307109
    FeII4415  = .4415022
    FeII4418  = .4417508
    FeII4472  = .4471552
    FeII4476  = .4476156
    FeII4689  = .4688868
    FeII4816  = .4815890
    FeII4907  = .4906721
    FeII4891  = .4890988
    FeII4895  = .4895147
    FeII5159  = .5159451
    FeII5160  = .5160229
    FeII5263  = .5263097
    FeII5264  = .5263932
    FeII5270  = .5270352
    FeII5275 = .5274831
    FeII5378  = .5377961
    FeII5435  = .5434659
    # Fe I  	444.54712   	0.00006  	444.54707   	0.00003   	 427  	2.44e+02  	  A 	0.08728574  	-  	2.87550352  	 3d64s2  	 a 5D  	 2  	 3d6(5D)4s4p(3P°)  	 z 7F°  	 2 

    Hd2Hb     = 0.2589
    Hg2Hb     = 0.4683
    Ha2Hb     = 2.8632
    c_100_kms = constants.c.to('100 km/s').value
    c_kms     = constants.c.to('km/s').value
    fwhm2sig  = np.sqrt(np.log(256.))

    SFR_to_L_Ha_Sh23 = 10**(-41.67) * units.Unit('Msun s / (yr erg)')

    def __init__(self, *args, **kwargs):
        super(jwst_spec_models, self).__init__(*args, **kwargs)
        self.dz = 0.03 if self.disperser=='r100' else 0.01
        self.__organize_models__()

    def chi(self, pars, *args):
        mask, model, *_ = args
        #if mask.ndim>1: mask = np.logical_or(*mask)
        if mask.ndim>1: mask = functools.reduce(np.logical_or, mask)

        return (
            (np.sum(model(pars), axis=0) - self.flux)/self.errs)[
            mask & ~self.mask]

    def chi2(self, pars, *args):
        chi = self.chi(pars, *args)

        return chi@chi

    def lnp(self, pars, *args):
        mask, model, lnprior, bounds = args
        if mask.ndim>1: mask = functools.reduce(np.logical_or, mask)
        if not all([b[0]<=p<=b[1] for p,b in zip(pars, bounds)]):
            return (-np.inf,) + (None,)*self.n_blobs
        lnprior = 0 if lnprior is None else lnprior(pars, *args)
        lnlike = (
            -np.sum(np.log(self.errs[mask]))-0.5*np.log(2*np.pi)
            -0.5*np.sum(self.chi(pars, mask, model)**2)
            )
        """
        if not np.isfinite(lnlike*lnprior): breakpoint()
        print(pars, lnprior, lnlike)
        """
        blobs = model(
            pars, *args, print_names=False, print_waves=False,
            print_blobs=True)
        return (lnprior + lnlike, lnlike) + blobs

    @classmethod
    def __organize_models__(cls):
        # Grab all models.
        unsorted_models = sorted([m for m in dir(cls)
            if (m.startswith('model_') and (not m.endswith('_fit_init'))
            and (not m=='model_families'))])
      
        # Sort models by bluest first.
        # Grab number of parameters, to be able to call the model function.
        n_pars = [
            getattr(cls, m)(cls, None, print_names=True)
            for m in unsorted_models]
        n_pars = [len(m.keys()) for m in n_pars]
        waves  = [np.zeros(n) for n in n_pars]
        waves  = [
            getattr(cls, m)(cls, pars, print_waves=True)[0]
            for m,pars in zip(unsorted_models, waves)]
        cls.sorted_models = [unsorted_models[index] for index in np.argsort(waves)]
        model_families = list(set([
            um.replace('_blr', '').replace('_outf', '') for um in cls.sorted_models]))
        cls.model_families = dict(zip(model_families,
            [[m for m in cls.sorted_models if m.startswith(mf)]
              for mf in model_families]))



    # Models
    def model_o2_ne3(self, pars, *args,
        print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):
        """[OII] and [Ne3]."""
        if print_names:
            return {
                'z_n_o2ne3': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_o2ne3': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fO23726': (r'$F(\mathrm{[O\,II]\lambda 3726})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_O2': (r'$F(\mathrm{[O\,II]\lambda 3729})$'+'\n'+r'$/F(\mathrm{[O\,II]\lambda 3726})$', 1., '[---]'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'bk0': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk1': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:                                                       
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        if print_blobs:                                                      
            return tuple()
                
        z_n, sig_n_100, fO23726, R, fNe33869, bk0, bk1 = pars
        fO23729 = fO23726 * R
        w_mum = np.array((self.OII3726, self.OII3729, self.NeIII3869)) * (1. + z_n)
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fO23726)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fO23729)
        f2 = gauss_int2(self.wave,  mu=w_mum[2], sig=sig_mum[2], flux=fNe33869)
        bk = bk1 * (self.wave - w_mum[0]) + bk0
        return f0, f1, f2, bk

    def model_o2_ne3_fit_init(self):
        """[OII] and [Ne3]. Bounds on R:=[OII]3729/[OII]3726 are from
        Sanders et al. (2015)."""
        central_wave = 0.3800*(1+self.redshift_guess)
        mask = np.abs(self.wave - central_wave)<0.15 # [um]
        central_pix  = np.argmin(np.abs(self.wave - central_wave))
        mask[central_pix-15:central_pix+15+1] = True
        mask = mask & ~self.mask
        func_name = traceback.extract_stack(None, 2)[-1][2]
        assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'
        bkg_med = np.nanmedian(self.flux[mask])
        bkg_std = np.nanstd(self.flux[mask])
        guess = np.array((
            self.redshift_guess, 1., 0.01, 1., 0.01,
            bkg_med, 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1e-5, 0., 0.3839, 0.,
             bkg_med - 10*bkg_std, -10.),
            (self.redshift_guess+self.dz, 5., 1., 1.4558, 1.,
             bkg_med + 10*bkg_std,  10.)))
        lnprior = None
        return mask, guess, bounds, lnprior
        
    def model_hg_o3(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):
        """Hgamma and [OIII]4363."""
        if print_names:
            return {
                'z_n_hgo3': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_hgo3': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHgamma': (r'$F(\mathrm{H\gamma})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 4363})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'bk0': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk1': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:                                                       
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        if print_blobs:                                                      
            return tuple()
        z_n, sig_n_100, fHgamma, fO34363, bk0, bk1 = pars
        w_mum = np.array((self.Hgamma, self.OIII4363)) * (1. + z_n)
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fHgamma)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fO34363)
        bk = bk1 * (self.wave - w_mum[0]) + bk0
        return f0, f1, bk

    def model_hg_o3_fit_init(self):
        """Hgamma and [OIII]4363."""
        central_wave = 0.4350*(1+self.redshift_guess)
        mask = np.abs(self.wave - central_wave)<0.045
        central_pix  = np.argmin(np.abs(self.wave - central_wave))
        mask[central_pix-15:central_pix+15+1] = True
        mask = mask & ~self.mask
        func_name = traceback.extract_stack(None, 2)[-1][2]
        #assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'
        bkg_med = np.nanmedian(self.flux[mask])
        bkg_std = np.nanstd(self.flux[mask])
        guess = np.array((
            self.redshift_guess, 1., 0.01, 0.01, bkg_med, 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0.,
             bkg_med - 10*bkg_std, -10.),
            (self.redshift_guess+self.dz, 5., 1., 1.,
             bkg_med + 10*bkg_std,  10.)))
        lnprior = None
        return mask, guess, bounds, lnprior
        

    # +---+-------------------------------------------------------------------+
    # | 3.| H\\beta and [OIII] models.                                        |
    # +---+-------------------------------------------------------------------+
    def model_hb_o3(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):
        """Hbeta and [OIII]4959,5007. Narrow only."""
        if print_names:
            return {
                'z_n_hbo3': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_hbo3': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta': (r'$F(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'bk0': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk1': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:                                                       
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        if print_blobs:                                                      
            return tuple()
        z_n, sig_n_100, fHbeta, fO35007, bk0, bk1 = pars
        fO34959 = fO35007 * 0.338
        w_mum = np.array((self.Hbeta, self.OIII4959, self.OIII5007)) * (1. + z_n)
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fHbeta)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fO34959)
        f2 = gauss_int2(self.wave,  mu=w_mum[2], sig=sig_mum[2], flux=fO35007)
        bk = bk1 * (self.wave - w_mum[0]) + bk0
        return f0, f1, f2, bk

    def model_hb_o3_fit_init(self):
        """Hbeta and [OIII]4959,5007. Narrow only."""
        central_wave = 0.4960*(1+self.redshift_guess)
        mask = np.abs(self.wave - central_wave)<0.15 # [um]
        central_pix  = np.argmin(np.abs(self.wave - central_wave))
        mask[central_pix-15:central_pix+15+1] = True
        mask = mask & ~self.mask
        func_name = traceback.extract_stack(None, 2)[-1][2]
        assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'
        bkg_med = np.nanmedian(self.flux[mask])
        bkg_std = np.nanstd(self.flux[mask])
        guess = np.array((
            self.redshift_guess, 1., 0.01, 0.01, bkg_med, 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0.,
             bkg_med - 10*bkg_std, -10.),
            (self.redshift_guess+self.dz, 5., 1., 1.,
             bkg_med + 10*bkg_std,  10.)))
        lnprior = None
        return mask, guess, bounds, lnprior
        

    def model_hb_o3_blr(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):
        """Hbeta and [OIII]4959,5007. Narrow and BLR."""
        if print_names:
            return {
                'z_n_hbo3': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_hbo3': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta': (r'$F(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_Hb_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_Hb_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr': (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'bk0': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk1': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:                                                       
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        if print_blobs:                                                      
            return tuple()
        z_n, sig_n_100, fHbeta, fO35007, v_blr_100, fwhm_blr_100, fHbeta_blr, bk0, bk1 = pars
        fO34959 = fO35007 * 0.338
        w_mum = (
            np.array((self.Hbeta, self.OIII4959, self.OIII5007, self.Hbeta))
            * (1.+z_n) * np.array((1., 1., 1., np.exp(v_blr_100/self.c_100_kms)))
            )
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.array((sig_n_100,)*3 + (fwhm_blr_100/self.fwhm2sig,))
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fHbeta)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fO34959)
        f2 = gauss_int2(self.wave,  mu=w_mum[2], sig=sig_mum[2], flux=fO35007)
        f3 = gauss_int2(self.wave,  mu=w_mum[3], sig=sig_mum[3], flux=fHbeta_blr)
        bk = bk1 * (self.wave - w_mum[0]) + bk0
        return f0, f1, f2, f3, bk

    def model_hb_o3_blr_fit_init(self):
        """Hbeta and [OIII]4959,5007. Narrow only."""
        central_wave = 0.4960*(1+self.redshift_guess)
        mask = np.abs(self.wave - central_wave)<0.15 # [um]
        central_pix  = np.argmin(np.abs(self.wave - central_wave))
        mask[central_pix-15:central_pix+15+1] = True
        mask = mask & ~self.mask
        func_name = traceback.extract_stack(None, 2)[-1][2]
        assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'
        bkg_med = np.nanmedian(self.flux[mask])
        bkg_std = np.nanstd(self.flux[mask])
        guess = np.array((
            self.redshift_guess, 1., 0.01, 0.01,
            0., 15., 0.1,
            bkg_med, 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0.,
             -5., 5., 0.,
             bkg_med - 10*bkg_std, -10.),
            (self.redshift_guess+self.dz, 5., 1., 1.,
              5., 150., 10.,
             bkg_med + 10*bkg_std,  10.)))
        def lnprior(pars, *args):
            (z_n, sig_n_100, fHbeta, fO35007,
             v_blr_100, fwhm_blr_100, fHbeta_blr, bk0, bk1) = pars
            if (sig_n_100 > fwhm_blr_100/self.fwhm2sig):
                return -np.inf
            # v_BLR within 200 km/s (Gaussian sigma) from NLR
            lnprior = -np.log(2.) - 0.5*(v_blr_100/2.)**2.
            # erfc prior
            lnprior += log_erfc_prior(fwhm_blr_100, mean=100., scale=10.)
            return lnprior
        return mask, guess, bounds, lnprior

    def model_hb_o3_outf(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):
        """Hbeta and [OIII]4959,5007. Narrow and outflows."""
        if print_names:
            return {
                'z_n_hbo3': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_hbo3': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta': (r'$F(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_out_hbo3_100': (r'$v_\mathrm{out}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_out_hbo3_100': (r'$\sigma_\mathrm{out}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_out': (r'$F_\mathrm{out}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007_out': (r'$F_\mathrm{out}(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'bk0': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk1': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:                                                       
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        if print_blobs:                                                      
            return tuple()
        (z_n, sig_n_100, fHbeta, fO35007,
         v_out_100, sig_out_100, fHbeta_out, fO35007_out, bk0, bk1) = pars
        fO34959 = fO35007 * 0.338
        fO34959_out = fO35007_out * 0.338
        w_mum = (
            np.array((self.Hbeta, self.OIII4959, self.OIII5007,
                      self.Hbeta, self.OIII4959, self.OIII5007))
            * (1.+z_n) * np.array((1.,)*3 + (np.exp(v_out_100/self.c_100_kms),)*3)
            )
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.array((sig_n_100,)*3 + (sig_out_100,)*3)
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fHbeta)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fO34959)
        f2 = gauss_int2(self.wave,  mu=w_mum[2], sig=sig_mum[2], flux=fO35007)
        f3 = gauss_int2(self.wave,  mu=w_mum[3], sig=sig_mum[3], flux=fHbeta_out)
        f4 = gauss_int2(self.wave,  mu=w_mum[4], sig=sig_mum[4], flux=fO34959_out)
        f5 = gauss_int2(self.wave,  mu=w_mum[5], sig=sig_mum[5], flux=fO35007_out)
        bk = bk1 * (self.wave - w_mum[0]) + bk0
        return f0, f1, f2, f3, f4, f5, bk

    def model_hb_o3_outf_fit_init(self):
        """Hbeta and [OIII]4959,5007. Narrow only."""
        central_wave = 0.4960*(1+self.redshift_guess)
        mask = np.abs(self.wave - central_wave)<0.15 # [um]
        central_pix  = np.argmin(np.abs(self.wave - central_wave))
        mask[central_pix-15:central_pix+15+1] = True
        mask = mask & ~self.mask
        func_name = traceback.extract_stack(None, 2)[-1][2]
        assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'
        bkg_med = np.nanmedian(self.flux[mask])
        bkg_std = np.nanstd(self.flux[mask])
        guess = np.array((
            self.redshift_guess, 1., 0.01, 0.01,
            0., 3., 0.01, 0.01,
            bkg_med, 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0.,
             -20., 2., 0., 0.,
             bkg_med - 10*bkg_std, -10.),
            (self.redshift_guess+self.dz, 5., 1., 1.,
              20., 20., 1., 1.,
              bkg_med + 10*bkg_std, 10.)))
        def lnprior(pars, *args):
            (z_n, sig_n_100, fHbeta, fO35007,
             v_out_100, sig_out_100, fHbeta_out, fO35007_out, bk0, bk1) = pars
            if (sig_n_100 > sig_out_100) or (fO35007_out < fHbeta_out):
                return -np.inf
            else:
                return 0.
        return mask, guess, bounds, lnprior

    # +---+-------------------------------------------------------------------+
    # | 4.| H\\alpha and [NII] models.                                        |
    # +---+-------------------------------------------------------------------+
    def model_ha_n2_s2(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha': (r'$F(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583': (r'$F(\mathrm{[N\,II]\lambda 6583})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fS26731': (r'$F(\mathrm{[S\,II]\lambda 6731})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_S2': (r'$F(\mathrm{[S\,II]\lambda 6716})$'+'\n'+r'$/F(\mathrm{[S\,II]\lambda 6731})$', 1., '[---]'),
                'bk0': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk1': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk2': (r'$bk_2$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-3}}]$')}
        if print_blob_dtypes:                                                       
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        if print_blobs:                                                      
            return tuple()
        z_n, sig_n_100, fHalpha, fN26583, fS26731, R, bk0, bk1, bk2 = pars
        fN26548 = fN26583 * 0.338
        fS26716 = fS26731 * R
        w_mum = np.array((
            self.NII6548, self.Halpha, self.NII6583,
            self.SII6716, self.SII6731)) * (1. + z_n)
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fN26548)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fHalpha)
        f2 = gauss_int2(self.wave,  mu=w_mum[2], sig=sig_mum[2], flux=fN26583)
        f3 = gauss_int2(self.wave,  mu=w_mum[3], sig=sig_mum[3], flux=fS26716)
        f4 = gauss_int2(self.wave,  mu=w_mum[4], sig=sig_mum[4], flux=fS26731)
        #bk = bk0 * (self.wave - w_mum[0]) + bk1
        bk = np.polynomial.legendre.legval(self.wave - w_mum[0], (bk0, bk1, bk2))
        return f0, f1, f2, f3, f4, bk

    def model_ha_n2_s2_fit_init(self):
        """Ha, [NII]. No longer using SII Bounds on R:=[SII]6716/[SII]6731 are from
        Sanders et al. (2015)."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        central_wave = 0.6600*(1+self.redshift_guess)
        mask = np.abs(self.wave - central_wave)<0.10 # [um]
        central_pix  = np.argmin(np.abs(self.wave - central_wave))
        mask[central_pix-15:central_pix+15+1] = True
        mask = mask & ~self.mask
        func_name = traceback.extract_stack(None, 2)[-1][2]
        assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'
        bkg_med = np.nanmedian(self.flux[mask])
        bkg_std = np.nanstd(self.flux[mask])
        guess = np.array((
            self.redshift_guess, 1., 0.01, 0.01, 0.01, 1.,
            bkg_med, 0., 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1e-5, 0., 0., 0., 0.4375,
             bkg_med - 10*bkg_std, -100., -100.),
            (self.redshift_guess+self.dz, 5., 1., 1., 1., 1.4484,
             bkg_med + 10*bkg_std, 100., 100.)))
        def lnprior(pars, *args):
            z_n, sig_n_100, fHalpha, fN26583, fS26731, R, bk0, bk1, bk2 = pars
            if (fHalpha < 0.3*fN26583) or (fHalpha < 0.3*fS26731):
                return -np.inf

            # erfc prior
            R_N2Ha = fN26583/fHalpha if fHalpha>0. else np.inf
            lnprior += log_erfc_prior(R_N2Ha, mean=3., scale=0.2)
            return lnprior
        return mask, guess, bounds, lnprior


    """
    def model_ha_n2_outf(self, pars, *args):
        (z_n, sig_n_100, fHalpha, fN26583,
         v_out, sig_out_100, fHalpha_out, fN26583_out, bk0, bk1) = pars
        fN26548 = fN26583 * 0.338
        fN26548_out = fN26583_out * 0.338
        w_mum = (
            np.array((self.NII6548, self.Halpha, self.NII6583,
                      self.NII6548, self.Halpha, self.NII6583))
            * (1. + z_n) * np.array((1.,)*3 + (np.exp(v_out/self.c_kms),)*3)
            )
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.array((sig_n_100,)*3 + (sig_out_100,)*3)
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fN26548)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fHalpha)
        f2 = gauss_int2(self.wave,  mu=w_mum[2], sig=sig_mum[2], flux=fN26583)
        f3 = gauss_int2(self.wave,  mu=w_mum[3], sig=sig_mum[3], flux=fN26548_out)
        f4 = gauss_int2(self.wave,  mu=w_mum[4], sig=sig_mum[4], flux=fHalpha_out)
        f5 = gauss_int2(self.wave,  mu=w_mum[5], sig=sig_mum[5], flux=fN26583_out)
        bk = bk0 * (self.wave - w_mum[0]) + bk1
        return f0, f1, f2, f3, f4, f5, bk
    """

    def model_ha_n2_s2_blr(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fN26583': (r'$F_\mathrm{n}(\mathrm{[N\,II]\lambda 6583})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fS26731': (r'$F(\mathrm{[S\,II]\lambda 6731})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_S2': (r'$F(\mathrm{[S\,II]\lambda 6716})$'+'\n'+r'$/F(\mathrm{[S\,II]\lambda 6731})$', 1., '[---]'),
                'v_blr_Ha_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_Ha_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'bk0': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk1': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$'),
                'bk2': (r'$bk_2$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-3}}]$')}
        if print_blob_dtypes:                                                       
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                }
        if print_blobs:                                                      
            return tuple()

        (z_n, sig_n_100, fHalpha, fN26583, fS26731, R_S2,
         v_blr_100, fwhm_blr_100, fHalpha_blr, bk0, bk1, bk2) = pars
        fN26548 = fN26583 * 0.338
        fS26716 = fS26731 * R_S2
        w_mum = (
            np.array((self.NII6548, self.Halpha, self.NII6583,
                      self.SII6716, self.SII6731, self.Halpha))
            * (1.+z_n)
            * np.array((1., 1., 1., 1., 1., np.exp(v_blr_100/self.c_100_kms)))
            )
        if print_waves: return w_mum
        sig_lsf = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig_n_100 = np.array((sig_n_100,)*5 + (fwhm_blr_100/self.fwhm2sig,))
        sig_n_100 = np.sqrt(sig_n_100**2 + sig_lsf**2)
        sig_mum = sig_n_100 / self.c_100_kms * w_mum
        f0 = gauss_int2(self.wave,  mu=w_mum[0], sig=sig_mum[0], flux=fN26548)
        f1 = gauss_int2(self.wave,  mu=w_mum[1], sig=sig_mum[1], flux=fHalpha)
        f2 = gauss_int2(self.wave,  mu=w_mum[2], sig=sig_mum[2], flux=fN26583)
        f3 = gauss_int2(self.wave,  mu=w_mum[3], sig=sig_mum[3], flux=fS26716)
        f4 = gauss_int2(self.wave,  mu=w_mum[4], sig=sig_mum[4], flux=fS26731)
        f5 = gauss_int2(self.wave,  mu=w_mum[5], sig=sig_mum[5], flux=fHalpha_blr)
        bk = np.polynomial.legendre.legval(self.wave - w_mum[0], (bk0, bk1, bk2))
        #bk = bk0 * (self.wave - w_mum[0])**2 + bk1 * (self.wave - w_mum[0]) + bk2
        return f0, f1, f2, f3, f4, f5, bk

    def model_ha_n2_s2_blr_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        central_wave = 0.6600*(1+self.redshift_guess)
        mask = np.abs(self.wave - central_wave)<0.10 # [um]
        central_pix  = np.argmin(np.abs(self.wave - central_wave))
        mask[central_pix-15:central_pix+15+1] = True
        mask = mask & ~self.mask
        func_name = traceback.extract_stack(None, 2)[-1][2]
        assert mask.sum()>24, f'{self.name} at z={self.redshift_guess} has less than 25 valid pixels for {func_name}'
        bkg_med = np.nanmedian(self.flux[mask])
        bkg_std = np.nanstd(self.flux[mask])
        guess = np.array((
            self.redshift_guess, 1., 0.01, 0.01, 0.01, 1.,
            0., 15., 0.1,
            bkg_med, 0., 0.))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0., 0., 0.4375,
             -5., 5., 0.,
             bkg_med - 10*bkg_std, -100., -100.),
            (self.redshift_guess+self.dz, 5., 1., 1., 1., 1.4484,
              5., 150., 10.,
             bkg_med + 10*bkg_std,  100.,  100.)))
        def lnprior(pars, *args):
            (z_n, sig_n_100, fHalpha, fN26583, fS26731, R_S2,
             v_blr_100, fwhm_blr_100, fHalpha_blr, bk0, bk1, bk2) = pars
            if (sig_n_100 > fwhm_blr_100/self.fwhm2sig):
                return -np.inf
            if (fHalpha < 0.3*fN26583) or (fHalpha < 0.3*fS26731):
                return -np.inf

            # v_BLR within 200 km/s (Gaussian sigma) from NLR
            lnprior = -np.log(2.) - 0.5*(v_blr_100/2.)**2.
            # erfc prior; FWHM_blr < 1e4, with a sigma of 1000.
            lnprior += log_erfc_prior(fwhm_blr_100, mean=100., scale=10.)
            # erfc prior; N2/Ha < 3. with a sigma of 0.2
            lnprior += log_erfc_prior(R_S2, mean=3., scale=0.2)
            return lnprior
        return mask, guess, bounds, lnprior


    def model_o2_double_blr(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_f$', 1., '[---]'),
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
                'bk20': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk21': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewhb'  : (r'$\mathrm{EW(H\beta)}$', 1., r'[\AA]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fNe33869_obs': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                                 r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float)
                }
        (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHb_b, fHa_b, frac1b,
         v_abs_100, sig_abs_100, C_f, tau0hb, tau0ha,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.NeIII3869, self.Hbeta, self.OIII4959, self.OIII5007, self.Halpha,
            self.Hbeta, self.Hbeta, self.Halpha, self.Halpha,
            self.Hbeta, self.Halpha))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*5
                  + (np.exp(v_blr_100/self.c_100_kms),)*4
                  + (np.exp(v_abs_100/self.c_100_kms),)*2)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 3.05
        fHb_b1 = fHb_b * frac1b
        fHb_b2 = fHb_b * (1.-frac1b)
        fHa_b1 = fHa_b * frac1b
        fHa_b2 = fHa_b * (1.-frac1b)
        (fNe33869, fHb, fO34959, fO35007, fHa,
         fHb_b1, fHb_b2, fHa_b1, fHa_b2, _, _) = np.array((
             fNe33869, fHb, fO34959, fO35007, fHa,
             fHb_b1, fHb_b2, fHa_b1, fHa_b2, 0., 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*5
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)*2
            + (sig_abs_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fNe33869)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fHb)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO34959)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO35007)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fHb_b1)
        f6 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[6], flux=fHb_b2)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHa_b1)
        f8 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[8], flux=fHa_b2)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[9]
        tau_hb = tau0hb * gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[10]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=tau0_norm)
        absrhb = 1. - C_f + C_f * np.exp(-tau_hb)
        absrha = 1. - C_f + C_f * np.exp(-tau_ha)
        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[2]) * b1
        bk2 = a2 + (self.wave-w_mum[4]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[5])/sig_mum[6]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[7])/sig_mum[8]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
            ewha /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
            return ewhb, ewha, fNe33869, fHb, fO35007, fHa, fHb_b1+fHb_b2, fHa_b1+fHa_b2
        return f0, f1, f2, f3, f4, f5*absrhb, f6*absrhb, f7*absrha, f8*absrha, bk0, bk1, bk2

    def model_o2_double_blr_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.3798, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.09, 0.15, 0.25)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2])

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        guess = np.array((
            self.redshift_guess, 1., 0.5, 0.01, 0.01, 0.01,
            0., 15., 30., 0.15, 1.5, .5,
            0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0., 0., 0.,
             -5., 10., 20., 0., 0.001, 0.,
             -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5., 1., 1., 1.,
              5., 20., 70., 1., 5., 1.,
              5., 5., 1.0, 30., 30.,
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
            (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHb_b, fHa_b, frac1b,
             v_abs_100, sig_abs_100, C_f, tau0hb, tau0ha,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.1)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=0.3, scale=0.05)

            # erfc prior; FWHM_blr < 1e4, with a sigma of 1000.
            lnprior += log_erfc_prior(fwhm_blr_100, mean=100., scale=10.)

            # erfc prior; N2/Ha < 3. with a sigma of 0.2
            R_N2Ha = fN26583/fHalpha if fHalpha>0. else np.inf
            lnprior += log_erfc_prior(R_N2Ha, mean=3., scale=0.2)

            return lnprior

        return masks, guess, bounds, lnprior

    def model_o2_voigt_blr(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_abs_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_f$', 1., '[---]'),
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
                'bk20': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk21': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:
            return [
                ("lnlike", float), ("ewhb", float), ("ewha", float),
                ("fNe33869", float), ("fHb", float), ("fO35007", float),
                ("fHa", float), ("fHb_b", float), ("fHa_b", float)]
        (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
         v_blr_100, fwhm_blr_100, fHb_b, fHa_b,
         v_abs_100, sig_abs_100, C_f, tau0hb, tau0ha,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.NeIII3869, self.Hbeta, self.OIII4959, self.OIII5007, self.Halpha,
            self.Hbeta, self.Halpha,
            self.Hbeta, self.Halpha))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*5
                  + (np.exp(v_blr_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_100/self.c_100_kms),)*2)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 3.05
        (fNe33869, fHb, fO34959, fO35007, fHa,
         fHa_b, fHa_b, _, _) = np.array((
             fNe33869, fHb, fO34959, fO35007, fHa,
             fHa_b, fHa_b, 0., 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*5
            + (0.,)*2 # FWHM for Voigt applied separately.
            + (sig_abs_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        hwhm_blr_mum = np.array((fwhm_blr_100,)*2) / constants.c.to('1e2 km/s').value
        hwhm_blr_mum *= (w_mum[5:7] / 2.)

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fNe33869)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fHb)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO34959)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO35007)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa)
        f5 = voigt_profile(self.wave-w_mum[5], sig_mum[5], hwhm_blr_mum[0])*fHb_b
        f6 = voigt_profile(self.wave-w_mum[6], sig_mum[6], hwhm_blr_mum[1])*fHa_b

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[7]
        tau_hb = tau0hb * gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[8]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=tau0_norm)
        absrhb = 1. - C_f + C_f * np.exp(-tau_hb)
        absrha = 1. - C_f + C_f * np.exp(-tau_ha)
        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[2]) * b1
        bk2 = a2 + (self.wave-w_mum[4]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[5])/sig_mum[5]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[6])/sig_mum[6]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
            ewha /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
            return ewhb, ewha, fNe33869, fHb, fO35007, fHa, fHb_b, fHa_b
        return f0, f1, f2, f3, f4, f5*absrhb, f6*absrha, bk0, bk1, bk2

    def model_o2_voigt_blr_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.3798, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.09, 0.15, 0.25)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2])

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        guess = np.array((
            self.redshift_guess, 1., 0.5, 0.01, 0.01, 0.01,
            0., 25., 0.15, 1.5,
            0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0., 0., 0.,
             -5., 10., 0., 0.001,
             -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5., 1., 1., 1.,
              5., 70., 1., 5.,
              5., 5., 1.0, 30., 30.,
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
            (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
             v_blr_100, fwhm_blr_100, fHb_b, fHa_b,
             v_abs_100, sig_abs_100, C_f, tau0hb, tau0ha,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.1)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=0.3, scale=0.05)

            return lnprior

        return masks, guess, bounds, lnprior


    def model_double_blr_ha_only(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha narrow and double BLR; absorber."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
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
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float)
                }
        (z_n, sig_n_100, fHa,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHa_b, frac1b,
         v_abs_100, sig_abs_100, C_f, tau0ha,
         a0, b0) = pars
        w_mum = np.array((self.Halpha, self.Halpha, self.Halpha, self.Halpha))
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)
                  + (np.exp(v_blr_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_100/self.c_100_kms),))
            )
        if print_waves: return w_mum

        fHa_b1 = fHa_b * frac1b
        fHa_b2 = fHa_b * (1.-frac1b)
        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)
            + (sig_abs_100,))
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHa)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fHa_b1)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fHa_b2)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[3]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=tau0_norm)
        absrha = 1. - C_f + C_f * np.exp(-tau_ha)
        bk0 = a0 + (self.wave-w_mum[0]) * b0

        if print_blobs:
            mask_ha = self.fit_mask[0] & (np.abs(self.wave-w_mum[3])/sig_mum[3]<10)
            dwha = np.gradient(self.wave[mask_ha])
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewha /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame

            return ewha, fHa, fHa_b1+fHa_b2
        return f0, f1*absrha, f2*absrha, bk0

    def model_double_blr_ha_only_fit_init(self):
        """Halpha narrow and double BLR; absorber."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.6565,))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.125,)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],)

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        guess = np.array((
            self.redshift_guess, 0.5, 0.01,
            0., 15., 30., 1.5, .5,
            0., 1.20, 0.9, 3.,
            bkg00, bkg01))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0.,
             -5., 10., 20., 0.001, 0.,
             -5., 0., 0.0, 0.1,
             bkg00-10*bkg01, -100.),
            (self.redshift_guess+self.dz, 5., 2.,
              5., 20., 70., 10., 1.,
              5., 5., 1.0, 30.,
             bkg00+10*bkg01, 100.)))
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
            (z_n, sig_n_100, fHa,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHa_b, frac1b,
             v_abs_100, sig_abs_100, C_f, tau0ha,
             a0, b0) = pars

            lnprior = 0.
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.1)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            return lnprior

        return masks, guess, bounds, lnprior

    def model_voigt_blr_ha_only(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha narrow and voigt BLR; absorber."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_abs_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_f$', 1., '[---]'),
                'tauHa': (r'$\tau_\mathrm{H\alpha}$', 1., '[---]'),
                'bk00': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk01': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float)
                }
        (z_n, sig_n_100, fHa,
         v_blr_100, fwhm_blr_100, fHa_b,
         v_abs_100, sig_abs_100, C_f, tau0ha,
         a0, b0) = pars
        w_mum = np.array((self.Halpha, self.Halpha, self.Halpha))
        w_mum = (w_mum * (1.+z_n)
            * np.array((1., np.exp(v_blr_100/self.c_100_kms),
                  np.exp(v_abs_100/self.c_100_kms),))
            )
        if print_waves: return w_mum

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array( # FWHM for Voigt applied separately.
            (sig_n_100, 0., sig_abs_100,))
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        hwhm_blr_mum = fwhm_blr_100 / constants.c.to('1e2 km/s').value
        hwhm_blr_mum *= w_mum[1] / 2.

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fHa)
        f1 = voigt_profile(self.wave-w_mum[1], sig_mum[1], hwhm_blr_mum)*fHa_b
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[2]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=tau0_norm)
        absrha = 1. - C_f + C_f * np.exp(-tau_ha)
        bk0 = a0 + (self.wave-w_mum[0]) * b0

        if print_blobs:
            mask_ha = self.fit_mask[0] & (np.abs(self.wave-w_mum[2])/sig_mum[2]<10)
            dwha = np.gradient(self.wave[mask_ha])
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewha /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
            return ewha, fHa, fHa_b
        return f0, f1*absrha, bk0

    def model_voigt_blr_ha_only_fit_init(self):
        """Halpha narrow and voigt BLR; absorber."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.6565,))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.125,)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],)

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        guess = np.array((
            self.redshift_guess, 0.5, 0.01,
            0., 30., 1.5,
            0., 1.20, 0.9, 3.,
            bkg00, bkg01))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0.,
             -5., 10., 0.001,
             -5., 0., 0.0, 0.1,
             bkg00-10*bkg01, -100.),
            (self.redshift_guess+self.dz, 5., 2.,
              5., 70., 10.,
              5., 5., 1.0, 30.,
             bkg00+10*bkg01, 100.)))
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
            (z_n, sig_n_100, fHa,
             v_blr_100, fwhm_blr_100, fHa_b,
             v_abs_100, sig_abs_100, C_f, tau0ha,
             a0, b0) = pars

            lnprior = 0.
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.1)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            return lnprior

        return masks, guess, bounds, lnprior

    def model_o2_double_blr_abs(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
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
                'bk20': (r'$bk_0$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-1}}]$'),
                'bk21': (r'$bk_1$', 1., 
                       r'$[10^{-20} \, \mathrm{erg\,s^{-1}\,cm^{-2}\,\\mu m^{-2}}]$')}
        if print_blob_dtypes:
            return {
                'lnlike': (r'$\log\,L$', 1., '[---]', float),
                'ewhb'  : (r'$\mathrm{EW(H\beta)}$', 1., r'[\AA]', float),
                'ewha'  : (r'$\mathrm{EW(H\alpha)}$', 1., r'[\AA]', float),
                'fNe33869_obs': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                                 r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
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

        (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHb_b, fHa_b, frac1b,
         v_abs_100, sig_abs_100, C_f, tau0hb, tau0ha,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.NeIII3869, self.Hbeta, self.OIII4959, self.OIII5007, self.Halpha,
            self.Hbeta, self.Hbeta, self.Halpha, self.Halpha,
            self.Hbeta, self.Halpha))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*5
                  + (np.exp(v_blr_100/self.c_100_kms),)*4
                  + (np.exp(v_abs_100/self.c_100_kms),)*2)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 2.98
        fHb_b1 = fHb_b * frac1b
        fHb_b2 = fHb_b * (1.-frac1b)
        fHa_b1 = fHa_b * frac1b
        fHa_b2 = fHa_b * (1.-frac1b)
        (fNe33869, fHb, fO34959, fO35007, fHa,
         fHb_b1, fHb_b2, fHa_b1, fHa_b2, _, _) = np.array((
             fNe33869, fHb, fO34959, fO35007, fHa,
             fHb_b1, fHb_b2, fHa_b1, fHa_b2, 0., 0)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*5
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)*2
            + (sig_abs_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fNe33869)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fHb)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO34959)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO35007)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fHb_b1)
        f6 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[6], flux=fHb_b2)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHa_b1)
        f8 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[8], flux=fHa_b2)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[9]
        tau_hb = tau0hb * gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[10]
        tau_ha = tau0ha * gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=tau0_norm)
        absrhb = 1. - C_f + C_f * np.exp(-tau_hb)
        absrha = 1. - C_f + C_f * np.exp(-tau_ha)
        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[2]) * b1
        bk2 = a2 + (self.wave-w_mum[4]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[5])/sig_mum[6]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[7])/sig_mum[8]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha = np.sum((1. - np.exp(-tau_ha[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
            ewha /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
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

            return (ewhb, ewha, fNe33869, fHb, fO35007, fHa, fHb_b1+fHb_b2, fHa_b1+fHa_b2,
                fwhm_blr_100, logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH)

        return f0, f1, f2, f3, f4, f5*absrhb, f6*absrhb, f7*absrha, f8*absrha, bk0, bk1, bk2

    def model_o2_double_blr_abs_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.3798, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.09, 0.15, 0.25)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2])

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        guess = np.array((
            self.redshift_guess, 1., 0.5, 0.01, 0.01, 0.01,
            0., 15., 30., 0.15, 1.5, .5,
            0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0., 0., 0.,
             -5., 10., 20., 0., 0.001, 0.,
             -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5., 1., 1., 1.,
              5., 20., 70., 1., 5., 1.,
              5., 5., 1.0, 30., 30.,
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
            (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHb_b, fHa_b, frac1b,
             v_abs_100, sig_abs_100, C_f, tau0hb, tau0ha,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.5)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.3, scale=0.05)

            return lnprior

        return masks, guess, bounds, lnprior



    def model_o2_double_blr_double_abs(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """Halpha and [NII]6548,6583. Narrow and BLR."""
        if print_names:
            return {
                'z_n_han2': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_1_100': (r'$FWHM_\mathrm{BLR,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_2_100': (r'$FWHM_\mathrm{BLR,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha_blr': (r'$F_\mathrm{BLR}(\mathrm{H\alpha})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'frac1b': (r'$F_\mathrm{BLR,1}/F_\mathrm{BLR}$', 1., '[---]'),
                'v_abs_1_100': (r'$v_\mathrm{abs,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_1_100': (r'$\sigma_\mathrm{abs,1}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf_1': (r'$C_{f,1}$', 1., '[---]'),
                'tauHb_1': (r'$\tau_\mathrm{H\beta,1}$', 1., '[---]'),
                'tauHa_1': (r'$\tau_\mathrm{H\alpha,1}$', 1., '[---]'),
                'v_abs_2_100': (r'$v_\mathrm{abs,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_2_100': (r'$\sigma_\mathrm{abs,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf_2': (r'$C_{f,2}$', 1., '[---]'),
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
                'ewhb_1'  : (r'$\mathrm{EW(H\beta,1)}$', 1., r'[\AA]', float),
                'ewha_1'  : (r'$\mathrm{EW(H\alpha,1)}$', 1., r'[\AA]', float),
                'ewhb_2'  : (r'$\mathrm{EW(H\beta,2)}$', 1., r'[\AA]', float),
                'ewha_2'  : (r'$\mathrm{EW(H\alpha,2)}$', 1., r'[\AA]', float),
                'fNe33869_obs': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                                 r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
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

        (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
         v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHb_b, fHa_b, frac1b,
         v_abs_1_100, sig_abs_1_100, C_f_1, tau0hb_1, tau0ha_1,
         v_abs_2_100, sig_abs_2_100, C_f_2, tau0hb_2, tau0ha_2,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.NeIII3869, self.Hbeta, self.OIII4959, self.OIII5007, self.Halpha,
            self.Hbeta, self.Hbeta, self.Halpha, self.Halpha,
            self.Hbeta, self.Halpha, self.Hbeta, self.Halpha))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*5
                  + (np.exp(v_blr_100/self.c_100_kms),)*4
                  + (np.exp(v_abs_1_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_2_100/self.c_100_kms),)*2)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 2.98
        fHb_b1 = fHb_b * frac1b
        fHb_b2 = fHb_b * (1.-frac1b)
        fHa_b1 = fHa_b * frac1b
        fHa_b2 = fHa_b * (1.-frac1b)
        (fNe33869, fHb, fO34959, fO35007, fHa,
         fHb_b1, fHb_b2, fHa_b1, fHa_b2, _, _, _, _) = np.array((
             fNe33869, fHb, fO34959, fO35007, fHa,
             fHb_b1, fHb_b2, fHa_b1, fHa_b2, 0., 0, .0, 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*5
            + (fwhm_blr_1_100/self.fwhm2sig, fwhm_blr_2_100/self.fwhm2sig)*2
            + (sig_abs_1_100,)*2 + (sig_abs_2_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fNe33869)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fHb)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO34959)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO35007)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fHb_b1)
        f6 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[6], flux=fHb_b2)
        f7 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHa_b1)
        f8 = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[8], flux=fHa_b2)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[9]
        tau_hb_1 = tau0hb_1 * gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[10]
        tau_ha_1 = tau0ha_1 * gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=tau0_norm)
        absrhb_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hb_1)
        absrha_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_ha_1)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[11]
        tau_hb_2 = tau0hb_2 * gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[12]
        tau_ha_2 = tau0ha_2 * gauss_int2(self.wave, mu=w_mum[12], sig=sig_mum[12], flux=tau0_norm)
        absrhb_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hb_2)
        absrha_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_ha_2)
        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[2]) * b1
        bk2 = a2 + (self.wave-w_mum[4]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[5])/sig_mum[6]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[7])/sig_mum[8]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb_1 = np.sum((1. - np.exp(-tau_hb_1[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha_1 = np.sum((1. - np.exp(-tau_ha_1[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb_1 /= (1+z_n)*np.exp((v_blr_100+v_abs_1_100)/self.c_100_kms) # Rest frame
            ewha_1 /= (1+z_n)*np.exp((v_blr_100+v_abs_1_100)/self.c_100_kms) # Rest frame
            ewhb_2 = np.sum((1. - np.exp(-tau_hb_2[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha_2 = np.sum((1. - np.exp(-tau_ha_2[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb_2 /= (1+z_n)*np.exp((v_blr_100+v_abs_2_100)/self.c_100_kms) # Rest frame
            ewha_2 /= (1+z_n)*np.exp((v_blr_100+v_abs_2_100)/self.c_100_kms) # Rest frame

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
                ewhb_1, ewha_1, ewhb_2, ewha_2, fNe33869, fHb, fO35007, fHa, fHb_b1+fHb_b2, fHa_b1+fHa_b2,
                fwhm_blr_100, logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH
                )

        return f0, f1, f2, f3, f4, f5*absrhb_1*absrhb_2, f6*absrhb_1*absrhb_2, f7*absrha_1*absrha_2, f8*absrha_1*absrha_2, bk0, bk1, bk2

    def model_o2_double_blr_double_abs_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.3798, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.09, 0.15, 0.25)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2])

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        guess = np.array((
            self.redshift_guess, 1., 0.5, 0.01, 0.01, 0.01,
            0., 15., 30., 0.15, 1.5, .5,
            0., 1.20, 0.9, 3., 3.,
            0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0., 0., 0.,
             -5., 10., 20., 0., 0.001, 0.,
             -5., 0., 0.0, 0.1, 0.1,
             -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5., 1., 5., 5.,
              5., 20., 70., 10., 30., 1.,
              5., 5., 1.0, 30., 30.,
              5., 5., 1.0, 30., 30.,
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
            (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
             v_blr_100, fwhm_blr_1_100, fwhm_blr_2_100, fHb_b, fHa_b, frac1b,
             v_abs_1_100, sig_abs_1_100, C_f_1, tau0hb_1, tau0ha_1,
             v_abs_2_100, sig_abs_2_100, C_f_2, tau0hb_2, tau0ha_2,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/.5)**2
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/.5)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.3, scale=0.05)

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
                'sig_n_100_han2': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHalpha': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
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
                'tauHb_1': (r'$\tau_\mathrm{H\beta,1}$', 1., '[---]'),
                'tauHa_1': (r'$\tau_\mathrm{H\alpha,1}$', 1., '[---]'),
                'v_abs_2_100': (r'$v_\mathrm{abs,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_2_100': (r'$\sigma_\mathrm{abs,2}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf_2': (r'$C_{f,2}$', 1., '[---]'),
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
                'ewhb_1'  : (r'$\mathrm{EW(H\beta,1)}$', 1., r'[\AA]', float),
                'ewha_1'  : (r'$\mathrm{EW(H\alpha,1)}$', 1., r'[\AA]', float),
                'ewhb_2'  : (r'$\mathrm{EW(H\beta,2)}$', 1., r'[\AA]', float),
                'ewha_2'  : (r'$\mathrm{EW(H\alpha,2)}$', 1., r'[\AA]', float),
                'fNe33869_obs': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                                 r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHalpha_obs': (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
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

        (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
         v_blr_100, fwhm_blr_100, fHb_b, fHa_b, tau_thom, T_thom,
         v_abs_1_100, sig_abs_1_100, C_f_1, tau0hb_1, tau0ha_1,
         v_abs_2_100, sig_abs_2_100, C_f_2, tau0hb_2, tau0ha_2,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.NeIII3869, self.Hbeta, self.OIII4959, self.OIII5007, self.Halpha,
            self.Hbeta, self.Halpha,
            self.Hbeta, self.Halpha, self.Hbeta, self.Halpha))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*5
                  + (np.exp(v_blr_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_1_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_2_100/self.c_100_kms),)*2)
            )
        if print_waves: return w_mum

        fHb = fHa/self.Ha2Hb
        fO34959 = fO35007 / 2.98
        (fNe33869, fHb, fO34959, fO35007, fHa,
         fHb_b, fHa_b, _, _, _, _) = np.array((
             fNe33869, fHb, fO34959, fO35007, fHa,
             fHb_b, fHa_b, 0., 0, .0, 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*5
            + (fwhm_blr_100/self.fwhm2sig, fwhm_blr_100/self.fwhm2sig)
            + (sig_abs_1_100,)*2 + (sig_abs_2_100,)*2)
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0 = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fNe33869)
        f1 = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fHb)
        f2 = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fO34959)
        f3 = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fO35007)
        f4 = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHa)
        f5 = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fHb_b)
        f6 = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fHa_b)

        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[7]
        tau_hb_1 = tau0hb_1 * gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[8]
        tau_ha_1 = tau0ha_1 * gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=tau0_norm)
        absrhb_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_hb_1)
        absrha_1 = 1. - C_f_1 + C_f_1 * np.exp(-tau_ha_1)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[9]
        tau_hb_2 = tau0hb_2 * gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=tau0_norm)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[10]
        tau_ha_2 = tau0ha_2 * gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=tau0_norm)
        absrhb_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_hb_2)
        absrha_2 = 1. - C_f_2 + C_f_2 * np.exp(-tau_ha_2)
        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[2]) * b1
        bk2 = a2 + (self.wave-w_mum[4]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        W_kms = (428. * tau_thom + 370.) * np.sqrt(T_thom)
        W_mum = W_kms/self.c_kms * w_mum[5]

        # Scatter Hb.
        dw = np.argmin(np.abs(self.wave-w_mum[5]))
        dw = np.gradient(self.wave)[dw]
        _w_ = np.arange(0., W_mum*25+dw, dw)
        _w_ = np.hstack([-_w_[1::][::-1], _w_])

        compton_kernel = np.exp(-np.abs(-np.abs(_w_)/W_mum))/(2.*W_mum)*dw # To unity.
        f7 = convolve(f5, compton_kernel, mode='same')
        f_scatt = 1 - np.exp(-tau_thom)
        f5 *= (1. - f_scatt)
        f7 *= f_scatt
       
        # Scatter Ha.
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

        if print_blobs:
            mask_hb = self.fit_mask[1] & (np.abs(self.wave-w_mum[5])/sig_mum[5]<10)
            mask_ha = self.fit_mask[2] & (np.abs(self.wave-w_mum[6])/sig_mum[6]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            dwha = np.gradient(self.wave[mask_ha])
            ewhb_1 = np.sum((1. - np.exp(-tau_hb_1[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha_1 = np.sum((1. - np.exp(-tau_ha_1[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb_1 /= (1+z_n)*np.exp((v_blr_100+v_abs_1_100)/self.c_100_kms) # Rest frame
            ewha_1 /= (1+z_n)*np.exp((v_blr_100+v_abs_1_100)/self.c_100_kms) # Rest frame
            ewhb_2 = np.sum((1. - np.exp(-tau_hb_2[mask_hb]))*dwhb)*1e4 # To [AA]
            ewha_2 = np.sum((1. - np.exp(-tau_ha_2[mask_ha]))*dwha)*1e4 # To [AA]
            ewhb_2 /= (1+z_n)*np.exp((v_blr_100+v_abs_2_100)/self.c_100_kms) # Rest frame
            ewha_2 /= (1+z_n)*np.exp((v_blr_100+v_abs_2_100)/self.c_100_kms) # Rest frame

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
                ewhb_1, ewha_1, ewhb_2, ewha_2, fNe33869, fHb, fO35007, fHa, fHb_b, fHa_b,
                logSFR_Ha, log_L_Ha_b_ism, logMBH, lEddMBH, W_kms
                )

        return f0, f1, f2, f3, f4, f5*absrhb_1*absrhb_2, f6*absrha_1*absrha_2, f7*absrhb_1*absrhb_2, f8*absrha_1*absrha_2, bk0, bk1, bk2

    def model_o2_exponential_blr_double_abs_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.3798, .4935, .6565))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.09, 0.15, 0.25)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2])

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        guess = np.array((
            self.redshift_guess, 1., 0.5, 0.01, 0.01, 0.01,
            0., 15., 0.15, 1.5, 0.5, 1.,
            0., 1.20, 0.9, 3., 3.,
            0., 1.20, 0.9, 3., 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0., 0., 0., 0.,
             -5., 1., 0., 0.001, 0., 0.1,
             -5., 0., 0.0, 0.1, 0.1,
             -5., 0., 0.0, 0.1, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5., 1., 1., 1.,
              5., 70., 1., 5., 30., 10.,
              5., 5., 1.0, 30., 30.,
              5., 5., 1.0, 30., 30.,
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
            (z_n, sig_n_100, Av, fNe33869, fO35007, fHa,
             v_blr_100, fwhm_blr_100, fHb_b, fHa_b, tau_thom, T_thom,
             v_abs_1_100, sig_abs_1_100, C_f_1, tau0hb_1, tau0ha_1,
             v_abs_2_100, sig_abs_2_100, C_f_2, tau0hb_2, tau0ha_2,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_1_100-(0.))/1.)**2
            lnprior += -np.log(.5) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_2_100-(0.))/1.)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.3, scale=0.05)

            # erfc prior. Difference between absorber velocities less than 100 km/s.
            # lnprior += log_erfc_prior(v_abs_1_100-v_abs_2_100, mean=1., scale=1.)

            return lnprior

        return masks, guess, bounds, lnprior


    def model_o2_single_blr(self, pars, *args, print_names=False, print_waves=False,
        print_blobs=False, print_blob_dtypes=False):

        """For 599..."""
        if print_names:
            return {
                'z_n': (r'$z_\mathrm{n}$', 1., '[---]'),
                'sig_n_100': (r'$\sigma_\mathrm{n}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'A_V': (r'$A_V$', 1., r'$[\mathrm{mag}]$'),
                'fO23726': (r'$F(\mathrm{[O\,II]\lambda 3726})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'R_O2': (r'$F(\mathrm{[O\,II]\lambda 3729})$'+'\n'+r'$/F(\mathrm{[O\,II]\lambda 3726})$', 1., '[---]'),
                'fNe33869': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO34363': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHbeta' : (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fO35007': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_blr_100': (r'$v_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fwhm_blr_100': (r'$FWHM_\mathrm{BLR}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'fHgamma_blr': (r'$F_\mathrm{BLR}(\mathrm{H\gamma})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'fHbeta_blr' : (r'$F_\mathrm{BLR}(\mathrm{H\beta})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$'),
                'v_abs_100': (r'$v_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'sig_abs_100': (r'$\sigma_\mathrm{abs}$', 100., r'$[\mathrm{km\,s^{-1}}]$'),
                'Cf': (r'$C_f$', 1., '[---]'),
                'tauHb': (r'$\tau_\mathrm{H\beta}$', 1., '[---]'),
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
                'ewhb'  : (r'$\mathrm{EW(H\beta)}$', 1., r'[\AA]', float),
                'fO23726_obs': (r'$F(\mathrm{[O\,II]\lambda 3726})$', 100.,
                                r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO33729_obs': (r'$F(\mathrm{[O\,II]\lambda 3729})$', 100.,
                                r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fNe33869_obs': (r'$F(\mathrm{[Ne\,III]\lambda 3869})$', 100.,
                                 r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fNe33968_obs': (r'$F(\mathrm{[Ne\,III]\lambda 3968})$', 100.,
                                 r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHdelta_obs': (r'$F_\mathrm{n}(\mathrm{H\delta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHgamma_obs': (r'$F_\mathrm{n}(\mathrm{H\gamma})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO34363_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                                r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_obs' : (r'$F_\mathrm{n}(\mathrm{H\alpha})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO34959_obs': (r'$F(\mathrm{[O\,III]\lambda 4959})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fO35007_obs': (r'$F(\mathrm{[O\,III]\lambda 5007})$', 100.,
                             r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHgamma_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\gamma})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float),
                'fHbeta_b_obs' : (r'$F_\mathrm{b}(\mathrm{H\beta})$', 100.,
                            r'$[10^{-18} \, \mathrm{erg\,s^{-1}\,cm^{-2}}]$', float)
                }
        (z_n, sig_n_100, Av,
         fO23726, R_O2, fNe33869, fO34363, fHb, fO35007,
         v_blr_100, fwhm_blr_100, fHg_b, fHb_b,
         v_abs_100, sig_abs_100, C_f, tau0hb,
         a0, b0, a1, b1, a2, b2) = pars
        w_mum = np.array((
            self.OII3726, self.OII3729, self.NeIII3869, self.NeIII3968,
            self.Hdelta, self.Hgamma, self.OIII4363,
            self.Hbeta, self.OIII4959, self.OIII5007,
            self.Hgamma, self.Hbeta, self.Hbeta))
        atten = _g03_(w_mum, Av)
        w_mum = (w_mum * (1.+z_n)
            * np.array((1.,)*10
                  + (np.exp(v_blr_100/self.c_100_kms),)*2
                  + (np.exp(v_abs_100/self.c_100_kms),))
            )
        if print_waves: return w_mum

        fHd = fHb * self.Hd2Hb
        fHg = fHb * self.Hg2Hb
        fO34959 = fO35007 / 3.05
        fNe33968 = fNe33869 / 3.05
        fO23729 = fO23726 * R_O2
        (fO23726, fO23729, fNe33869, fNe33968, fHd, fHg, fO34363,
         fHb, fO34959, fO35007, fHg_b, fHb_b, _) = np.array((
             fO23726, fO23729, fNe33869, fNe33968, fHd, fHg, fO34363,
             fHb, fO34959, fO35007, fHg_b, fHb_b, 0.)) * atten

        sig_lsf_100 = self.lsf_sigma_kms(w_mum) / 1.e2 # LSF in [1e2 km/s]
        sig100 = np.array(
            (sig_n_100,)*10
            + (fwhm_blr_100/self.fwhm2sig,)*2
            + (sig_abs_100,))
        sig100  = np.sqrt(sig100**2 + sig_lsf_100**2)
        sig_mum = sig100 / constants.c.to('1e2 km/s').value * w_mum

        f0  = gauss_int2(self.wave, mu=w_mum[0], sig=sig_mum[0], flux=fO23726)
        f1  = gauss_int2(self.wave, mu=w_mum[1], sig=sig_mum[1], flux=fO23729)
        f2  = gauss_int2(self.wave, mu=w_mum[2], sig=sig_mum[2], flux=fNe33869)
        f3  = gauss_int2(self.wave, mu=w_mum[3], sig=sig_mum[3], flux=fNe33968)
        f4  = gauss_int2(self.wave, mu=w_mum[4], sig=sig_mum[4], flux=fHd)
        f5  = gauss_int2(self.wave, mu=w_mum[5], sig=sig_mum[5], flux=fHg)
        f6  = gauss_int2(self.wave, mu=w_mum[6], sig=sig_mum[6], flux=fO34363)
        f7  = gauss_int2(self.wave, mu=w_mum[7], sig=sig_mum[7], flux=fHb)
        f8  = gauss_int2(self.wave, mu=w_mum[8], sig=sig_mum[8], flux=fO34959)
        f9  = gauss_int2(self.wave, mu=w_mum[9], sig=sig_mum[9], flux=fO35007)
        f10 = gauss_int2(self.wave, mu=w_mum[10], sig=sig_mum[10], flux=fHg_b)
        f11 = gauss_int2(self.wave, mu=w_mum[11], sig=sig_mum[11], flux=fHb_b)
        tau0_norm = np.sqrt(2.*np.pi) * sig_mum[12]
        tau_hb = tau0hb * gauss_int2(self.wave, mu=w_mum[12], sig=sig_mum[12], flux=tau0_norm)
        absrhb = 1. - C_f + C_f * np.exp(-tau_hb)
        bk0 = a0 + (self.wave-w_mum[0]) * b0
        bk1 = a1 + (self.wave-w_mum[2]) * b1
        bk2 = a2 + (self.wave-w_mum[4]) * b2
        bk0 = np.where(self.fit_mask[0], bk0, 0)
        bk1 = np.where(self.fit_mask[1], bk1, 0)
        bk2 = np.where(self.fit_mask[2], bk2, 0)

        if print_blobs:
            mask_hb = self.fit_mask[2] & (np.abs(self.wave-w_mum[11])/sig_mum[11]<10)
            dwhb = np.gradient(self.wave[mask_hb])
            ewhb = np.sum((1. - np.exp(-tau_hb[mask_hb]))*dwhb)*1e4 # To [AA]
            ewhb /= (1+z_n)*np.exp((v_blr_100+v_abs_100)/self.c_100_kms) # Rest frame
            return (
                ewhb, fO23726, fO23729, fNe33869, fNe33968, fHd, fHg, fO34363,
                fHb, fO34959, fO35007, fHg_b, fHb_b)
        return f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11*absrhb, bk0, bk1, bk2

    def model_o2_single_blr_fit_init(self):
        """Halpha and [NII]6548,6583. Narrow and BLR."""
        #mask = mask & ~(np.abs(self.wave - 0.6717*(1.+self.redshift_guess))<0.005)
        #mask = mask & ~(np.abs(self.wave - 0.6731*(1.+self.redshift_guess))<0.005)
        lwave = np.array((.3798, .4250, .4935))
        pivot_waves = lwave * (1+self.redshift_guess)
        windw_sizes = (0.09, 0.25, 0.15)
        masks = (
            np.abs(self.wave-pivot_waves[0])<windw_sizes[0],
            np.abs(self.wave-pivot_waves[1])<windw_sizes[1],
            np.abs(self.wave-pivot_waves[2])<windw_sizes[2])

        bkg00 = np.nanmedian(self.flux[masks[0]])
        bkg10 = np.nanmedian(self.flux[masks[1]])
        bkg20 = np.nanmedian(self.flux[masks[2]])
        bkg01 = np.nanstd(self.flux[masks[0]])
        bkg11 = np.nanstd(self.flux[masks[1]])
        bkg21 = np.nanstd(self.flux[masks[2]])
        guess = np.array((
            self.redshift_guess, 1., 0.5,
            0.01, 1., 0.01, 0.01, 0.01, 0.01,
            0., 25., 0.01, 0.2,
            0., 1.20, 0.9, 3.,
            bkg00, bkg01, bkg10, bkg11, bkg20, bkg21))
        bounds = np.array((
            (self.redshift_guess-self.dz, 1.e-5, 0.,
             0., 0.3839, 0., 0., 0., 0.,
             -5., 10., 0., 0.001,
             -5., 0., 0.0, 0.1,
             bkg00-10*bkg01, -100., bkg10-10*bkg11, -100.,
             bkg20-10*bkg21, -100.),
            (self.redshift_guess+self.dz, 5., 5.,
             1., 1.4558, 1., 1., 1., 1.,
             5., 70., 1., 5.,
             5., 5., 1.0, 30.,
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
            (z_n, sig_n_100, Av,
             fO23726, R_O2, fNe33869, fO34363, fHb, fO35007,
             v_blr_100, fwhm_blr_100, fHg_b, fHb_b,
             v_abs_100, sig_abs_100, C_f, tau0hb,
             a0, b0, a1, b1, a2, b2) = pars

            lnprior = 0.
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_abs_100-(0.))/.1)**2
            lnprior += -np.log(.1) - 0.5*np.log(2.*np.pi) - 0.5*((v_blr_100-(0.))/.1)**2

            # erfc prior; broad Hb/Ha<1/3 iwth a sigma of 0.05
            lnprior += log_erfc_prior(fHb_b/fHa_b, mean=.3, scale=0.05)

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



class jwst_spec_fitter(jwst_spec_models, emcee.EnsembleSampler):

    def __new__(cls, name=None, folder=None, model=None, fit_name=None,
        nwalkers=100, nsteps=1000, burnin=0.5, img_exten='png',
        generous_factor=2., force_reload=False, **kwargs):

        # If mandatory `__init__` arguments are set to `None`, then this must
        # be a call through `pickle.load`. Initializer is skipped thanks to
        # the `initialized` attribute set to True in the pickled instance.
        if (name is None) and (folder is None) and (model is None):
            return super().__new__(cls)#, name, folder, model, fit_name=fit_name,

        # If mandatory `__init__` arguments are not `None`, then this is a call
        # to create a new instance, or resume from disk (if it exists).
        # Start by checking that the `model` is a valid function name.
        assert hasattr(cls, model), f'{model} must be a method of `jwst_spec_models`'

        fit_name = model if fit_name is None else fit_name
        output_folder = (
            f'{name}' if kwargs.get('output_folder', None) is None
            else kwargs['output_folder'])
        disperser = kwargs.get('disperser', 'r1000')
        output_file = os.path.join(
            output_folder, f'{name}_{disperser}_{fit_name}.pckl')

        if not force_reload and os.path.isfile(output_file):
            print(f'Resuming {name} from disk, from file {output_file}.')
            inst = pickle.load(
                open(output_file, 'rb'))
            tmp_model = getattr(cls, model)
            model_pars = tmp_model(cls, None, print_names=True)
            assert all([
                k0==k1 for k0,k1 in zip(model_pars.keys(), inst.model_pars.keys())
                ]), f'Model {model} seems incompatible with {output_file}'
            assert inst.disperser==disperser, f'{disperser} incompatible with {output_file}.{disperser}'
            return inst

        return super().__new__(cls)#, name, folder, model, fit_name=fit_name,
            #nwalkers=nwalkers, nsteps=nsteps, burnin=burnin, img_exten=img_exten,
            #generous_factor=generous_factor, **kwargs)


    def __init__(self, name, folder, model, fit_name=None,
        nwalkers=100, nsteps=1000, burnin=0.5, img_exten='png',
        generous_factor=2., **kwargs):
        """
        Parameters
        ----------
        generous_factor : float, or n-paramaters array. Positive
            When initializing the sampler chains, use a Gaussian having as mean the
            least-squares solution, and as dispersion `generous_factor` times the
            least-squares uncertainty.
        Return
        ------
        """

        if getattr(self, 'initialized', False):
            return
        self.initialized = True

        super(jwst_spec_fitter, self).__init__(name, folder, **kwargs)

        self.fit_name = model if fit_name is None else fit_name

        output_file = os.path.join(
            self.output_folder,
            f'{self.name}_{self.disperser}_{self.fit_name}.pckl')
        output_img  = os.path.join(
            self.output_folder,
            f'{self.name}_{self.disperser}_{self.fit_name}_spectrum.{img_exten}')

        self.model = getattr(self, model)
        model_pars = self.model(None, print_names=True)

        self.generous_factor = generous_factor
        self.model_pars = model_pars
        self.model_blobs = self.model(None, print_blob_dtypes=True)
        fit_init = getattr(self, f'{model}_fit_init')
        self.fit_mask, self.guess, self.bounds, lnprior = fit_init()

        self.fit_or_mask = (
            self.fit_mask if self.fit_mask.ndim==1
            #else np.logical_or(*self.fit_mask))
            else functools.reduce(np.logical_or, self.fit_mask))

        self.n_pars, self.n_data = len(self.guess), self.fit_mask.sum()
        args = (self.fit_mask, self.model, lnprior, self.bounds)

        sol_lsq = least_squares(
            self.chi, x0=self.guess, args=args, bounds=self.bounds)
        # TBD assert sol_lsq['x'][0] > 0, 'Wat'
        self.pars_lsq = sol_lsq['x']
        J = sol_lsq.jac; prec = J.T.dot(J)

        unused_pars = np.diag(prec)==0.
        self.errs_lsq = np.zeros_like(self.pars_lsq)
        if any(unused_pars): # Use 1/5 of the bounds range to initialize.
            prec = prec[np.ix_(~unused_pars, ~unused_pars)]
            self.errs_lsq[unused_pars] = np.squeeze(np.diff(
                self.bounds[:, unused_pars], axis=0))/5.
        cov = np.linalg.inv(prec)
        self.errs_lsq[~unused_pars] = np.sqrt(np.diagonal(cov))
        unused_pars = ~np.isfinite(self.errs_lsq)
        if any(unused_pars): # Use 1/5 of the bounds range to initialize.
            warnings.warn(
                'Found nan in LSQ uncertainties. Using bounds to inizialize chains',
                RuntimeWarning)
            self.errs_lsq[unused_pars] = np.squeeze(np.diff(
                self.bounds[:, unused_pars], axis=0))/5.

        if kwargs.get('diffevo', False):
            sol_diffevo = differential_evolution(
                self.chi2, self.bounds.T, args=args)
            self.pars_lsq = sol_diffevo['x']
            print('Successfuly ran Differential Evolution')
       
        model_lsq = self.model(self.pars_lsq)
        fig = plt.figure()
        lsq_output_file = f'{self.output_folder}/{self.name}_{self.fit_name}_lsq.pdf'
        plt.step(self.wave, self.flux, 'k-', where='mid')
        plt.step(self.wave, np.sum(model_lsq, axis=0), 'r-', where='mid')
        plt.savefig(lsq_output_file, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

        pos = np.array([
            np.random.normal(self.pars_lsq, self.generous_factor*self.errs_lsq)
            for _ in range(nwalkers)])
        pos = np.array([np.clip(p, self.bounds[0], self.bounds[1]) for p in pos])

        self.bounds_emcee = self.bounds.T
        args = (self.fit_mask, self.model, lnprior, self.bounds_emcee)

        try:
            self.blobs_dtype = self.model(None, print_blob_dtypes=True)
            self.blobs_dtype = [(key, item[-1]) for key,item in self.blobs_dtype.items()]
            self.n_blobs = len(self.blobs_dtype)
        except:
            self.blobs_dtype = [("lnlike", float),]
            self.n_blobs = 1
        emcee.EnsembleSampler.__init__(self,
            nwalkers, len(self.guess), self.lnp, args=args,
            blobs_dtype=self.blobs_dtype)
        
        self.run_mcmc(pos, nsteps, progress=True)
        burnin = int(nsteps*burnin)
        self.samples = self.chain[:, burnin:, :].reshape((-1, len(self.guess)))
        extra_samples = np.array([
             self.get_blobs()[blob][burnin:, :].flatten()
             for blob in list(zip(*self.blobs_dtype))[0]]).T
        self.extended_samples = np.column_stack([
            self.samples, extra_samples])
        lnlike = self.get_blobs()["lnlike"][burnin:, ].flatten()
        self.lnlike = np.nanmax(lnlike)
        self.bic    = self.n_pars * np.log(self.n_data) - 2. * self.lnlike
        
        self.pars = np.array(list(map(
            lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(self.samples, [16, 50, 84], axis=0)))))
        self.extended_pars = np.array(list(map(
            lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(self.extended_samples, [16, 50, 84], axis=0)))))
        self.pars_mcmc = self.pars[:, 0] 
        self.model_spec = self.model(self.samples[np.nanargmax(lnlike), :])

        self.samples_denorm = np.copy(self.samples)
        self.pars_denorm    = np.copy(self.pars)
        for i,mp in enumerate(self.model_pars.values()):
            self.samples_denorm[:, i] *= mp[1] # De-normalise.
            self.pars_denorm[i, :] *= mp[1]    # De-normalise.
        self.extended_samples_denorm = np.copy(self.extended_samples)
        self.extended_pars_denorm    = np.copy(self.extended_pars)

        for i,mp in enumerate(list(self.model_pars.values())+list(self.model_blobs.values())):
            self.extended_samples_denorm[:, i] *= mp[1] # De-normalise.
            self.extended_pars_denorm[i, :] *= mp[1]    # De-normalise.

        # As a test.
        del self.log_prob_fn # This contains non-serializable `lnprior`
                             # instance, and prevents `pickle.dump`
        pickle.dump(self, open(output_file, 'wb'))
        print(f'Written {output_file} to disk.')

        return None # __init__ bic, lnlike, model_spec, mask, samples


    def plot_corner(self, trim=None, corner_kwargs={}):

        output_file = f'{self.output_folder}/{self.name}_{self.fit_name}_corner.png'
        corner_kwargs = (
            {'show_titles': False, 'plot_datapoints': False, 'smooth': 1.,
             'bins': 30, 'smooth1d': 3., 'hist_bin_factor': 3,
             'quantiles': (.16, .50, .84), 'label_kwargs': {'fontsize': 20}}
            if not corner_kwargs else corner_kwargs)

        trim = (trim if trim is not None else
            np.array([
                True if not re.search(r'bk[0-9]{1,2}', k) else False
                for k in self.model_pars.keys()])
            )
        
        model_labels_trim = np.array(
            [_v_[0] for _v_ in self.model_pars.values()])[trim]

        fig = corner.corner(
            self.samples_denorm[:, trim], labels=model_labels_trim, **corner_kwargs)

        # Replace titles with values using just sign. digits.
        for i,(m,p) in enumerate(zip(model_labels_trim, self.pars_denorm[trim, :])):
            title = m + '=' + self.round_to_significant_digits(p, to_string=True)
            fig.axes[i*len(model_labels_trim)+i].set_title(
                title, fontsize=20)

        ax, rax = self.ax_from_cornerfig(fig)
        self.plot_best_fit(ax=ax, rax=rax, fontsize=24)

        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.01)

        return fig

    def plot_corner_custom(self, keys, corner_kwargs={}):

        output_file = f'{self.output_folder}/{self.name}_{self.fit_name}_corner_custom.png'
        corner_kwargs = (
            {'show_titles': False, 'plot_datapoints': False, 'smooth': 1.,
             'bins': 30, 'smooth1d': 3., 'hist_bin_factor': 3,
             'quantiles': (.16, .50, .84), 'label_kwargs': {'fontsize': 20}}
            if not corner_kwargs else corner_kwargs)

        all_labels = (
            [_v_[0] for _v_ in self.model_pars.values()]
            + [_v_[0] for _v_ in self.model_blobs.values()])
        all_keys = (
            [_v_ for _v_ in self.model_pars.keys()]
            + [_v_ for _v_ in self.model_blobs.keys()])
        #assert all([k in all_labels for k in keys]), (
        #    f'Some keys in {keys} not found in the model. Check spelling')

        indices = np.array([i for i in range(len(all_keys))
            if all_keys[i] in keys])

        samples = self.extended_samples_denorm[:, indices]
        labels  = np.array(all_labels)[indices]

        fig = corner.corner(
            samples, labels=labels, **corner_kwargs)

        # Replace titles with values using just sign. digits.
        for i,(m,p) in enumerate(zip(labels, self.extended_pars_denorm[indices, :])):
            title = m + '=' + self.round_to_significant_digits(p, to_string=True)
            fig.axes[i*len(keys)+i].set_title(
                title, fontsize=20)

        ax, rax = self.ax_from_cornerfig(fig)
        self.plot_best_fit(ax=ax, rax=rax, fontsize=24)

        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.01)

        return fig


    def plot_best_fit(self, ax=None, rax=None, fontsize=24):

        ax = plt.gca() if ax is None else ax
        plot_mask = np.arange(self.wave.size)
        plot_mask = plot_mask[self.fit_or_mask][[0, -1]]
        plot_mask = np.array([True if (plot_mask[0]<=pm<=plot_mask[1])
            else False for pm in np.arange(self.wave.size)])
        ax.step(
            self.wave[plot_mask], self.flux[plot_mask],
            'k-', alpha=0.5, where='mid')
        ax.fill_between(
            self.wave[plot_mask],
            self.flux[plot_mask]-self.errs[plot_mask],
            self.flux[plot_mask]+self.errs[plot_mask],
            facecolor='darkgrey', edgecolor='none',
            alpha=0.2, step='mid')
        model_sum = np.sum(self.model_spec, axis=0)
        model_sum[~self.fit_or_mask] = np.nan
        ax.step(
           self.wave, model_sum,
           color='crimson', where='mid')
        ax.set_xlabel(r'$\lambda \; \mathrm{(obs.) \; [\mu m]}$', fontsize=fontsize)
        ax.set_ylabel(
            r'$F_\lambda \; {\rm [10^{-20}\,erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]}$',
            fontsize=fontsize)
        ax.text(0.05, 0.95, r'$\mathrm{{BIC}}$'+f'$={self.bic:.1f}$',
            va='top', ha='left', fontsize=fontsize, transform=ax.transAxes)
        line_waves = self.model(self.pars[:, 0], print_waves=True)
        for lw in line_waves:
            for _ax_ in (ax, rax):
                if _ax_: _ax_.axvline(lw, ls='--', alpha=0.5, color='k')
        if rax is None:
            return
        chi = ( (self.flux - model_sum) / self.errs)
        rax.step(self.wave, chi,
             'k-', alpha=0.5, where='mid')
        rax.axhline(0., linestyle='--', color='grey')
        ax.set_xlabel('')
        rax.set_xlabel(r'$\lambda \; \mathrm{(obs.) \; [\mu m]}$', fontsize=fontsize)
        rax.set_ylabel(r'$\chi$', fontsize=fontsize)

    @staticmethod
    def ax_from_cornerfig(fig, pad=(0.4,0.1)):
        """Grab a set of axes for plotting the spectrum, fiducial model and
        fit residuals."""

        n_ax = int(np.sqrt(len(fig.axes)))

        ax_tr = fig.axes[n_ax-1].get_position()
        ax_bl = fig.axes[(n_ax//2)*n_ax-1 + int(np.ceil(n_ax/2.))].get_position()
        ax_dx = ax_bl.x1 - ax_bl.x0
        ax_dy = ax_bl.y1 - ax_bl.y0

        x_min, x_max = ax_bl.x1, ax_tr.x1
        y_min, y_max = ax_bl.y1, ax_tr.y1
        x_min += pad[0]*ax_dx
        y_min += pad[1]*ax_dy
        # Normally, one would correct for the figure shape, but for corner plots the
        # figure is generally very close to a square.
        dx = x_max - x_min
        dy = dx / 1.618
        dy_rax = dy*0.4
        y_min = y_max - dy - dy_rax
        ax = fig.add_axes([x_min, y_min+dy_rax, dx, dy])
        rax = fig.add_axes([x_min, y_min, dx, dy_rax], sharex=ax)

        return ax, rax

    @staticmethod
    def round_to_significant_digits(parameter, to_string=False):
        """Round the median and inter-percentile range (or noise) to the first
        significant digits of the uncertainty."""
        p50, hi, lo = (
            (parameter[0], parameter[1], parameter[1])
            if len(parameter)==2 else parameter)

        prec = int(np.min([np.floor(np.log10(lo)), np.floor(np.log10(hi))]))
        p50, lo, hi = [np.round(_x_, -prec) for _x_ in (p50, lo, hi)]
        prec = int(np.clip(-prec, 0, np.inf))
        if to_string:
            return f'${p50:.{prec}f}^{{+{hi:.{prec}f}}}_{{-{lo:.{prec}f}}}$'
        return p50, lo, hi, prec



    def write_table(self, percentiles=(16, 50, 84), n_sigma_detection=3.,
        ):

        keys = list(self.model_pars.keys())
        trim = [True if not re.search(r'bk[0-9]{1,2}', k) else False for k in keys]

        output_model_pars = {k: self.model_pars[k] for t,k in zip(
            trim, self.model_pars) if t}

        colnames = tuple(
            [f'{k}_{prc}' for k in output_model_pars.keys()
             for prc in ('50', 'hi', 'lo')])

        units = list([mp[2] for mp in output_model_pars.values()
            for prc in ('50', 'hi', 'lo')])
        for i,u in enumerate(units):
            if r'km\,s^{-1}' in u: u = 'km/s'
            if r'AA' in u: u = 'Angstrom'
            if r'erg\,s^{-1}' in u: u = '1e-18 erg/(s cm2)'
            if '---' in u: u=''
            units[i] = u.strip('[]')

        values = [None,] * len(units)
        fmts   = [None,] * len(units)
        
        for i,mp in enumerate(output_model_pars.values()):
            detected = True
            p50, lo, hi = self.pars_denorm[i, :]
            is_this_a_detectable = (
                (r'10^{-18} \, \mathrm{erg' in mp[2]) or mp[0].startswith('F(')
                or ('sigma_n' in mp[0]))
            if is_this_a_detectable:
                if p50 - 3*lo<=0:
                    pars_symmetrized_err = 0.5*(hi + lo)
                    hi = n_sigma_detection * pars_symmetrized_err
                    lo = n_sigma_detection * pars_symmetrized_err
                    detected = False
                    
            p50, lo, hi, prec = self.round_to_significant_digits(
                 (p50, hi, lo))
            if not detected:
                p50, lo = 0., 0.
            values[i*3]   = p50
            values[i*3+1] = hi
            values[i*3+2] = lo
            fmts[i*3:i*3+3] = [f'%.{prec}f',]*3

        colnames = (
            ('id', 'folder', 'input_z', 'model_name', 'bic') + colnames)
        units = ['', '', '', ''] + units
        values = (
            [int(self.name), self.folder, self.redshift_guess, self.fit_name, self.bic]
            + values)
        fmts = ['%10d', '', '%6.5f', '%<35s', '%4.1f'] + fmts

        values = [[v,] for v in values]
        output_table = table.Table(data=values, names=colnames)
        for c,u,f in zip(output_table.colnames, units, fmts):
            output_table[c].unit = u
            output_table[c].format = f

        output_file = os.path.join(
            self.output_folder, f'{self.name}_{self.fit_name}_table.fits')
        output_table.write(output_file, overwrite=True)


    def append_tables(self, input_table=None):

        if not hasattr(self, 'model_families'): self.__organize_models__()

        input_table = (
            make_master_table(self) if input_table is None else input_table)

        selected_tables = []
        for mf,model_names in self.model_families.items():
            tables = [
                os.path.join(self.output_folder, f'{self.name}_{m}_table.fits')
                for m in model_names]
            tables = [table.Table.read(f) for f in tables if os.path.isfile(f)]
            if len(tables)==0:
                continue
            bics   = [t['bic'] for t in tables]
            reference_model = model_names.index(mf)
            reference_bic = bics[reference_model]
            selected_model = np.argmin(bics)
            # If minimum bic is reference (simplest) model, take that model; otherwise
            # enter this `if`.
            if not model_names[selected_model]==mf:
                # If minimum bic beats reference-model bic by 10+, confirm selection.
                # Othewrise, enter this `if` and opt for the reference model.
                if not reference_bic-bics[selected_model]>10:
                   selected_model = reference_model
                
            selected_table = tables[selected_model]
            for c in selected_table.colnames:
                if c in ('bic', 'model_name'):
                    mf_short = mf.replace('model_', '')
                    selected_table[c].name = f'{c}_{mf_short}'
                if any([c in st.colnames for st in selected_tables]):
                    selected_table.remove_column(c)
            selected_tables.append(tables[selected_model])

        joined_selected_table = table.hstack(selected_tables)
        return table.vstack((input_table, joined_selected_table))
        


def make_master_table(inst):

    tables = [
        os.path.join(inst.output_folder, f'{inst.name}_{m}_table.fits')
        for m in inst.sorted_models]
    tables = [table.Table.read(f) for f in tables if os.path.isfile(f)]
    table_keys, table_formats, table_units, table_dtypes = [], [], [], []
    for _tab_,mf in zip(tables, inst.sorted_models):
        for col in _tab_.colnames:
            if col in ('bic', 'model_name'):
                mf_short = mf.replace('model_', '').replace('_blr', '').replace('_outf', '')
                col_updated = f'{col}_{mf_short}'
            else:
                col_updated = col
            if col_updated not in table_keys:
                table_keys.append(col_updated)
                table_formats.append(_tab_[col].format)
                table_units.append(_tab_[col].unit)
                table_dtypes.append(_tab_[col].dtype)

    output_table = table.Table(
        names=table_keys, dtype=table_dtypes, units=table_units)
    for col,fmt in zip(output_table.colnames, table_formats):
        output_table[col].format = fmt

    return output_table
