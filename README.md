# global-fit

Requirements: python 3.6, numpy, matplotlib, lmfit. To install these modules in Anaconda run:
>`conda install -c anaconda numpy`
>`conda install -c conda-forge matplotlib`
>`conda install -c conda-forge lmfit`

Download and save globalfit.py, laplace.py and data.csv files in your working directory. File 'data.csv' is given as an example (UvrD translocation on ssDNA with 7 different lengths). Save your own data in csv file in the working directory.

Open python console in the working directory. Import globalfit.py running:
>`from globalfit import *`

Start fitting model:
>`mymodel = GlobalFit()`

Plot the data:
>`mymodel.plot()`

Fit the data:
>`mymodel.fit()`

Print parameters:
>`mymodel.params()`

Update parameters:
>`mymodel.update()`

Save fit parameters into a file:
>`mymodel.save()`
