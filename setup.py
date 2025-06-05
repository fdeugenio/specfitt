import subprocess
from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as file:
        return(file.read())

version = "0.0.1"

setup(name='specfitt',
      version=version,
      description='Fitting needlessly complicated models to spectra',
      long_description=readme(),
      classifiers=[
        'Development Status :: 0 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='https://github.com/fdeugenio/specfitt',
      author='Francesco D\'Eugenio',
      author_email='fdeugenio@gmail.com',
      license='Other/Proprietary License',
      packages=find_packages(),
      install_requires=[
        'matplotlib<3.8.0',
        'numpy==1.24.4',
        'scipy==1.6.3',
        'astropy==6.1.0',
        'tqdm>=4.45.0',
        'corner==2.2.2',
        'emcee==3.1.4',
        'spectres==1.24.4',
        'dust_extinction==1.4.1',
      ],
      python_requires='>=3.10.9',
      entry_points={
        'console_scripts': [
            'gamappxf        = gamappxf.gamappxf_script:main',
            'sxdfppxf        = gamappxf.sxdfppxf_script:main',
            'gamappxfm       = gamappxf.gamappxf_mscript:main',
            ],
      },
      #include_package_data=True,
      package_data={'': [
          '00_data/redshifts.csv',
          '00_data/egs-nelson08-v4_g395m-f290lp_4106_57146.spec.fits',
          '00_data/01_disp/*fits',
          ]}
     )
