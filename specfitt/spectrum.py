import glob
import functools
import os
import pickle
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate

from astropy.io import fits
from astropy import constants, table, units

#from astropy.io.fits.verify import VerifyWarning
#warnings.simplefilter('ignore', category=VerifyWarning)
from astropy.utils.exceptions import AstropyWarning


import spectres

__all__ = ['jwst_spec',]

files_path = os.path.dirname(__file__)
LRD_TABLE     = os.path.join(files_path, '00_data/redshifts.csv')
JADES_Z_TABLE = '00_data/jades_z_table_v0.7.1.fits'
HAILMARY_Z_TABLE = '00_data/galaxy_list.txt'
DARKHORSE_Z_TABLE = '00_data/goods-s-darkhorse000.csv'



class jwst_spec:
    files_path = os.path.dirname(__file__)
    dispersers = {'r100': '(prism|PRISM)',
        'r1000': '(g140m|G140M|g235m|G235M|g395m|G395M)',
        'r2700': '(g140h|G140H|g235h|G235H|g395h|G395H)',
        'wfss': '(f356w|f444w|F356W|F444W)'}
    lsf_anna = { 
        'r100': os.path.join(files_path,
            '00_data/01_disp/point_source_lsf_clear_prism_QD4_i185_j85.csv'),
        'r1000':os.path.join(files_path,
            '00_data/01_disp/point_source_lsf_.*(g140m|g235m|g395m)_QD4_i185_j85.csv'),
    }
    lsf_nominal = {
        'r100':os.path.join(files_path,
            '00_data/01_disp/jwst_nirspec_prism_disp.fits'),
        'r1000':os.path.join(files_path,
            '00_data/01_disp/jwst_nirspec_(g140m|g235m|g395m)_disp.fits'),
        'r2700':os.path.join(files_path,
            '00_data/01_disp/jwst_nirspec_(g140h|g235h|g395h)_disp.fits'),
        'wfss':os.path.join(files_path,
            '00_data/01_disp/jwst_nircam_wfss_disp.fits')
    }
    flux_unit = units.Unit('1e-20 erg/(s cm2 AA)')

    def __init__(self, name, folder, disperser='r1000', output_folder=None,
        noise_rescale=1., **kwargs):

        self.disperser = disperser
        self.name = name
        self.folder = folder
        self.output_folder = f'{name}' if output_folder is None else output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.wave, self.flux, self.errs, self.mask = self.__read_files__(
            name, folder, disperser, **kwargs)
        if callable(noise_rescale):
            noise_rescale = noise_rescale(self.wave)
        self.errs *= noise_rescale
        if name==57146:
            self.flux[3] = np.nan
            warnings.warn(f'Replacing data artefact at pixel 3 for galaxy {name}')
   
        self.lsf_type = kwargs.get('lsf_type', 'nominal')
        self.lsf_sigma_kms = self.__get_lsf__()
        self.redshift_guess = self.__redshift_guess__(**kwargs)


    @classmethod
    def __find_files__(cls, name, folder, disperser, **kwargs):
        """Search for appropriate files in the folder `folder`, matching
        `name` and one or more dispersers.

        Return
        ------
        list of strings
            file names matching the appropriate regex.
        """

        dispersers = cls.dispersers[disperser]
        file_path = os.path.join(folder, f'*{name}*.fits')
        
        print(f'.*{dispersers}.*{name}.*.fits')
        matching_files = []
        for f in sorted(glob.glob(file_path)):
            match = re.search(f'.*{name}.*{dispersers}.*.fits', f)
            if match:
                matching_files.append(match.group())
            match = re.search(f'.*{dispersers}.*{name}.*.fits', f)
            if match:
                matching_files.append(match.group())
   
        assert len(matching_files)<=3, (
            f'Can only use up to three dispersers, found {matching_files}')
        assert len(matching_files)>0, (
            f'No file matching {name} and {dispersers} in {folder}')

        return matching_files



    @classmethod
    def __read_files__(cls, name, folder, disperser, **kwargs):
        """Given a galaxy `name` and a `folder` name, dispatch the
        methods to identify suitable file names, and read them.

        Return
        ------
        wave, flux, errs, mask
        """

        matching_files = cls.__find_files__(name, folder, disperser, **kwargs)

        #if len(matching_files)==1:
        #    return cls.__read_single_file__(filename)

        # In this case, there must be 2 or 3 files
        _spectra_ = [cls.__read_single_file__(f) for f in matching_files]

        return functools.reduce(cls.__splice_spectra__, _spectra_)
        
        



    @classmethod
    def __read_single_file__(cls, filename):
        """Return wave, flux and error read from `filename`. Uses different
        methods for different data distributions. All fluxes are in F_lambda,
        with units of `cls.flux_unit`."""
        with fits.open(filename) as hdu:
            extensions = set([exten.header.get('EXTNAME', None) for exten in hdu])
            # This is a file from the GTO pipeline.
            required_extensions = {'DATA', 'ERR', 'WAVELENGTH'}
            if required_extensions.issubset(extensions):
                wave = hdu['WAVELENGTH'].data * 1e6 * units.um
                flux = (hdu['DATA'].data * units.Unit('W/m3')).to(
                    cls.flux_unit)
                errs = (hdu['ERR'].data * units.Unit('W/m3')).to(
                    cls.flux_unit)
                mask = ~np.isfinite(flux*errs) | (errs<=0.)
                return wave.value, flux.value, errs.value, mask

            # This is a public-release file from the GTO pipeline, re-formatted to
            # comply with MAST.
            if 'EXTRACT1D' in extensions:
                if 'FLUX_ERR' in hdu['EXTRACT1D'].data.names:
                    wave = hdu['EXTRACT1D'].data['WAVELENGTH'] * units.um
                    flux = hdu['EXTRACT1D'].data['FLUX'] * 1e20 * cls.flux_unit
                    errs = hdu['EXTRACT1D'].data['FLUX_ERR'] * 1e20 * cls.flux_unit
                    mask = ~np.isfinite(flux*errs) | (errs<=0.)
                    return wave.value, flux.value, errs.value, mask
    
            # This is a public-release file from MAST, with the `jwst` pipeline.
            if 'EXTRACT1D' in extensions:
                if 'FLUX_ERROR' in hdu['EXTRACT1D'].data.names:
                    wave = hdu['EXTRACT1D'].data['WAVELENGTH'] * units.um
                    flux = hdu['EXTRACT1D'].data['FLUX'] * units.Jy
                    errs = hdu['EXTRACT1D'].data['FLUX_ERROR'] * units.Jy
                    flux = (flux*constants.c/wave**2).to(cls.flux_unit)
                    errs = (errs*constants.c/wave**2).to(cls.flux_unit)
                    mask = ~np.isfinite(flux*errs) | (errs<=0.)
                    return wave.value, flux.value, errs.value, mask

            # This is a file from DJA.
            if 'SPEC1D' in extensions:
                wave = hdu['SPEC1D'].data['wave'] * units.um
                flux = hdu['SPEC1D'].data['flux'] * units.uJy
                errs = hdu['SPEC1D'].data['err'] * units.uJy
                flux = (flux*constants.c/wave**2).to(cls.flux_unit)
                errs = (errs*constants.c/wave**2).to(cls.flux_unit)
                mask = ~np.isfinite(flux*errs) | (errs<=0.)
                return wave.value, flux.value, errs.value, mask

            if 'WFSSTAB' in extensions:
                wave = hdu['WFSSTAB'].data['wavelength_um'] * units.um
                flux = hdu['WFSSTAB'].data['flux_mJy'] * units.mJy
                errs = hdu['WFSSTAB'].data['fluxerr_mJy'] * units.mJy
                flux = (flux*constants.c/wave**2).to(cls.flux_unit)
                errs = (errs*constants.c/wave**2).to(cls.flux_unit)
                mask = ~np.isfinite(flux*errs) | (errs<=0.)
                return wave.value, flux.value, errs.value, mask

            # We should never be here. This is a different file not implemented.
            raise ValueError(
                f'File {filename} seems to have an unforeseen format.')



    @classmethod
    def __splice_spectra__(
        cls, spec_list0, spec_list1):
        """Take two spectra, and overlap them by using the bluest part of the
        overlapping spectrum.

        spec_list0: list of four nd-arrays
            The four arrays represent wavelength, flux density, uncertainty and
            a bad pixel mask.
        spec_list1: list of four nd-arrays
        Return
        ------

        wave, flux, errs, mask

        """

        wave0, flux0, errs0, mask0 = spec_list0
        wave1, flux1, errs1, mask1 = spec_list1

        # Always use red in overlap (higher SNR).
        blue_slice = wave0<wave1[0]
        red__slice = wave1>wave0[-1]

        wave_overl, flux_overl, errs_overl, mask_overl = [], [], [], []
        blue_overl = wave0>=wave1[0]
        red__overl = wave1<=wave0[-1]
        if any(blue_overl) and any(red__overl):
     
            dwav_overl = wave1[1]-wave1[0]
            
            wave_overl = np.arange(
                wave0[blue_overl][0], wave1[red__overl][-1], dwav_overl)
            flux_overl0 = spectres.spectral_resampling.spectres(
                wave_overl, wave0[blue_overl], flux0[blue_overl], verbose=False)
            flux_overl1 = spectres.spectral_resampling.spectres(
                wave_overl, wave1[red__overl], flux1[red__overl], verbose=False)
            errs_overl0 = spectres.spectral_resampling.spectres(
                wave_overl, wave0[blue_overl], errs0[blue_overl]**2, verbose=False)
            errs_overl1 = spectres.spectral_resampling.spectres(
                wave_overl, wave1[red__overl], errs1[red__overl]**2, verbose=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                flux_overl = np.nanmean(np.vstack((flux_overl0, flux_overl1)), axis=0)
                errs_overl = np.nanmean(np.vstack((errs_overl0, errs_overl1)), axis=0)
            errs_overl = np.sqrt(errs_overl)
            mask_overl = ~np.isfinite(flux_overl*errs_overl) | (errs_overl<=0.)

        wave = np.hstack([wave0[blue_slice], wave_overl, wave1[red__slice]])
        flux = np.hstack([flux0[blue_slice], flux_overl, flux1[red__slice]])
        errs = np.hstack([errs0[blue_slice], errs_overl, errs1[red__slice]])
        mask = np.hstack([mask0[blue_slice], mask_overl, mask1[red__slice]])
        mask = mask.astype(bool) # If mask_overl==[], `hstack` casts as float.

        return wave, flux, errs, mask



    def __get_lsf__(self):
        if self.lsf_type=='anna':
            return self.__get_lsf_anna__()
        return self.__get_lsf_nominal__()
            

    def __get_lsf_anna__(self):
        lsf_file_regex = self.lsf_anna[self.disperser]

        matching_files = []
        for f in sorted(glob.glob(
            os.path.join(self.file_path, '00_data/01_disp/*csv'))):

            match = re.search(lsf_file_regex, f)
            if not match:
                continue
            matching_files.append(match.group())
   
        assert len(matching_files)<=3, (
            f'Can only use up to three dispersers, found {matching_files}')
        assert len(matching_files)>0, (
            f'No disperser files matching {lsf_file_regex}')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            _lsf_ = [table.Table.read(f) for f in matching_files]

        # Read the LSF resolution, pad with dummy values to be able to use
        # `__splice_spectra__`, which expects tuples looking like
        # (wavelength, flux, uncertainty, mask).
        # We want to average `R' as if it were a flux.
        _lsf_ = [(
            t['wave']/1e4, t['sigma']/1e4, np.zeros_like(t['sigma']),
            np.zeros_like(t['sigma'], dtype=bool)) for t in _lsf_]
 
        _wave_, _lsf_sigma_um_, _, _ = functools.reduce(self.__splice_spectra__, _lsf_)
        _lsf_sigma_kms_ = ( # Convert from FWHM to sigma assuming a Gaussian
            constants.c.to('km/s').value * _lsf_sigma_um_ / _wave_)
        return scipy.interpolate.interp1d(
            _wave_, _lsf_sigma_kms_, 'linear',
            fill_value=tuple(_lsf_sigma_kms_[[0, -1]]), bounds_error=False)



    def __get_lsf_nominal__(self, fudge_factor=0.7):
        """Create and attach a linear interpolator for the local dispersion."""
        lsf_file_regex = self.lsf_nominal[self.disperser]

        matching_files = []
        for f in sorted(glob.glob(os.path.join(
            self.files_path, '00_data/01_disp/*fits'))):
            match = re.search(lsf_file_regex, f)
            if not match:
                continue
            matching_files.append(match.group())
   
        assert len(matching_files)<=3, (
            f'Can only use up to three dispersers, found {matching_files}')
        assert len(matching_files)>0, (
            f'No disperser files matching {lsf_file_regex}')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            _lsf_ = [table.Table.read(f) for f in matching_files]
        # Read the LSF resolution, pad with dummy values to be able to use
        # `__splice_spectra__`, which expects tuples looking like
        # (wavelength, flux, uncertainty, mask).
        # We want to average `R' as if it were a flux.
        _lsf_ = [(
            t['WAVELENGTH'], t['R'], np.zeros_like(t['R']),
            np.zeros_like(t['R'], dtype=bool)) for t in _lsf_]
 
        _wave_, _lsf_R_, _, _ = functools.reduce(self.__splice_spectra__, _lsf_)
        _lsf_sigma_kms_ = ( # Convert from FWHM to sigma assuming a Gaussian
            constants.c.to('km/s').value / _lsf_R_ / np.sqrt(np.log(256.)))
        _lsf_sigma_kms_ *= fudge_factor
        return scipy.interpolate.interp1d(
            _wave_, _lsf_sigma_kms_, 'linear',
            fill_value=tuple(_lsf_sigma_kms_[[0, -1]]), bounds_error=False)



    def __redshift_guess__(self, **kwargs):
        if kwargs.get('z', 0.) > 0.:
            return kwargs['z']

        # Last, desperate attempt.
        z_tab = table.Table.read(LRD_TABLE)
        match = np.where(z_tab['ID']==self.name)[0]
        if len(match)==1 and z_tab[match]['z']>0:
            return z_tab[match]['z'][0]

        """
        z_tab = table.Table.read(JADES_Z_TABLE)
        match = np.where(z_tab['NIRSpec_ID']==self.name)[0]
        if len(match)==1 and z_tab[match]['z_visinsp']>0:
            return z_tab[match]['z_visinsp'][0]
        if len(match)>1 and np.all(z_tab[match]['z_visinsp']>0):
            return np.mean(z_tab[match]['z_visinsp'])
        if len(match)>1 and np.any(z_visinsp[match]['z_visinsp']>0):
            good_z = z_tab[match]['z_visinsp']>0
            return z_tab[match]['z_visinsp'][good_z]

        # Last, desperate attempt.
        z_tab = table.Table.read(
            HAILMARY_Z_TABLE, format='ascii.commented_header')
        match = np.where(z_tab['NIRSpec']==self.name)[0]
        if len(match)==1 and z_tab[match]['z']>0:
            return z_tab[match]['z'][0]
        """

        # Dark Horse Survey
        z_tab = table.Table.read(
            DARKHORSE_Z_TABLE, format='csv')
        match = np.where(z_tab['ID']==self.name)[0]
        if len(match)==1 and z_tab[match]['z_visinsp']>0:
            return z_tab[match]['z_visinsp'][0]

        raise NotImplementedError(f"Only JADES redshifts available for now. Failed {self.name}")
        # ETC
