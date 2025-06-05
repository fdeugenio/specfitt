# specfitt
Quick joint fitting of emission lines

Obtain this package
```git clone git@github.com:fdeugenio/specfitt.git specfitt_src```

Create a virtual environment (may use `anaconda` or any other alternative)
```virtualenv --python=python3.12 pytest_specfitt```

Activate environment (may vary depending on what virtual environment was used, and on console type)
```source pytest_specfitt/bin/activate```

Install (local folder; recommended to edit the code. Remove `-e` to install in a hidden folder, which is the normal behaviour) 
`pip3 install specfitt_src/.`

I also install other stuff that helps with plotting, but it may depend on your setup. Some people use `jupyter-notebook`. I use `ipython`:
`pip3 install pyGObject pycairo ipython`

An example test would be (from inside a python interpreter)
```import specfitt
   folder = os.path.join(specfitt.__path__[0], '00_data') # Search inside default folder.
   spec = specfitt.jwst_spec_fitter(
       57146, folder,
       'model_o2_double_blr_double_abs', # Must be a valid model name function in `specfitt.py`
       disperser='r1000', force_reload=False, # False implies reload old fits, 
       nsteps=500, burnin=0.5, # For fast turnaround. Probably too few steps
       diffevo=False, # True uses a better guess than LSQ, but takes longer
       output_folder='Delete_me')
```
To plot the full corner plot, use
```spec.plot_corner()```
This takes some time, but one can be more sensible and plot just a subset of the parameters (these are the keys of the parameter dictionary inside the model function).
