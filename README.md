# global-fit

Requirements:
python 3.6, numpy, matplotlib, lmfit

Download globalfit.py, laplace.py and data.csv\
Save your data in csv file in the same directory with globalfit.py and laplace.py.\
File 'data.csv' is given as an example (UvrD translocation on ssDNA with 7 different lengths).\

Open Anaconda prompt and navigate to the directory containing globalfit.py\
Open python console (in Anaconda prompt type "jupyter console")\
Type and run:\
from globalfit import *<br>

To create a new fitting model type:\
yourmodel = GlobalFit()\
At the prompt type "twostepdiss"\
Next type at the prompt type "kt, kd, kc, kend, r, C"\
Load data:\
yourmodel.data()\
Type "data.csv" and enter\
Initialize parameters using:\
yourmodel.initialize()\
yourmodel.fit()\
