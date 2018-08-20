# global-fit

Globalfit is a wrapper around lmfit (https://lmfit.github.io/lmfit-py) providing an interface for multiple curves fitting with global parameters.

To use this program you will need to install Python3 on your computer. A convenient way to install Python3 and required modules is to use Anaconda. After installing Anaconda with Python3 install following modules: numpy, scipy, matplotlib, lmfit, emcee, corner, os, and pandas.
To install these modules in Anaconda run:
>`conda install -c anaconda numpy`

>`conda install -c anaconda scipy`

>`conda install -c conda-forge matplotlib`

>`conda install -c conda-forge lmfit`

>`conda install -c astropy emcee`

>`conda install -c anaconda panda`

>`conda install -c astropy corner`

Download the folder containing globalfit.py which will be the working directory.

Open python console in the working directory. Import GlobalModel class from globalfit.py:
>`from globalfit import GlobalModel`

Create a fitting model:
>`my_model = GlobalModel()`
* Select a model for fitting. New user-defined models can be added in models.py file.
* Enter path name of the folder containing data file. Your data should be saved in the data.csv file with independent variable in the first column: t, y1, y2, y3, ...
* Type in global variables (comma separated) that will be shared among all datasets.
* Parameter initial guesses will be read from params.gss file. If params.gss file is absent then you will be prompted to enter initial guesses which will be saved as params.gss file.

After initiating a model you can use following methods to fit and save your data.

Fit the data:
>`my_model.fit()`
* Enter parameter names that will be constrained during fitting.

Overwrite model parameter values with parameter values obtained from the fit:
>`my_model.write()`

Read parameters from params.gss file:
>`my_model.read()`

Plot data and simulation curves:
>`my_model.plot()`

Fit report (fit values, std errors, correlations):
>`my_model.report()`

Save model parameters, simulation curves, and residuals into a file:
>`my_model.save()`

Perform Markov-Chain Monte-Carlo simulations to obtain Bayesian sampling of the posterior distribution:
>`my_model.emcee()`

Tutorials:
* [Example1](https://github.com/ordabayev/global-fit/blob/master/example1.ipynb): simple n-step unwinding model.
* [Example2](https://github.com/ordabayev/global-fit/blob/master/example2.ipynb): n-step translocation with two-step dissociation. Here I also compare globalfit to Conlin program.
