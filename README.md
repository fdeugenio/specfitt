# specfitt
Quick joint fitting of emission lines

Obtain this package
```git clone git@github.com:fdeugenio/specfitt.git```

Create a virtual environment (may use `anaconda` or any other alternative)
```virtualenv --python=python3.12 pytest_specfitt```

Activate environment (may vary depending on what virtual environment was used, and on console type)
```source pytest_specfitt/bin/activate```

Install (local folder; recommended to edit the code. Remove `-e` to install in a hidden folder, which is the normal behaviour) 
`pip3 install specfitt/.`

I also install other stuff that helps with plotting, but it may depend on your setup. Some people use `jupyter-notebook`. I use `ipython`:
`pip3 install pyGObject pycairo ipython`

