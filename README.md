So, you've downloaded some UV-vis code. This readme explains the contents of your
download and how to use it. 

You have downloaded this file, as well as a folder named Python. Within that 
folder is the various codes used for evaluating thin film thickness. 

Your first step is to enter the Python folder, and run setup.py. By doing so, the
necessary path tree will be built. You should a path tree as follows:

Home
 -- Python
   -- Code(s).py
 -- Data
   -- Plots
     -- Approx
     -- MCMC
     -- Manual
     -- data.csv
     
I also included a practice data file! Move it to the correct folder first. Then 
go ahead and play around with it. 

You are now ready to measure some films. Begin by scanning** your film, and uploading
the .csv to the Data folder. Then, enter the Python folder, and run the desired
code. There are several to choose from. Each code includes a basic UI to guide you.

** scans MUST begin with a blank, for background subtraction

mcmc.py employs Bayesian statistics in order to evaluate film thickness. However, 
due to its Markov Chain, its run time is significantly longer than least squares
or other similar techniques. However, using a Markov Chain circumvents completely
the issue of least squares instability. 

manual.py carries no quantitative measure. Instead, the user uses sliders in order
to create a qualitative fit. This is an effective and quick method of producing an
estimate of the thickness. 

unc.py uses a Monte Carlo simulation and least squares data fitting in order to find
both the thickness of the film and the uncertainty in the thickness. It is powerful
in the regard that it both data fits (however prior knowledge is necessary) and
returns a distribution of fits according to the uncertainty in the independent
varaibles. 

Each mcmc.py, manual.py, and unc.py utilize a qualitative data normalization. Doing so
operates under the assumption that the sinusoidal behavior is independent of the 
baseline attenuation which muddles the sinusoid. 

approx.py uses a technique from literature to estimate the thickness of the film
from the index of refraction and location of critical points in the sinusoid. 

The other codes are various tools. distr.py includes the films which we built and
the measured thicknesses. omcmc.py and omanual.py are the outdated versions of 
mcmc.py and manual.py. plot_spec.py is a basic tool to plot a .csv without any
normalization or other processing. cauchy.py is meant to demonstrate the wavelength
dependent index of refraction.

Matteo Fulghieri 


