import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import norm
from scipy.signal import savgol_filter, argrelmin, argrelmax
import random as r
import os

## Written by Matteo Fulghieri with help from Elizabeth Ryland, Natalia Powers-Riggs, and Trevor Wiechmann

## This code is meant to use Monte Carlo Markov Chain techniques to determine the thickness of a thin film from
## its interference pattern. In this code, you get to see the Markov Chain (via histogram and walk path), the best
## achieved parameters according to model, and finally the model overlayed with the data. Like manual.py, you also
## maually control the data normalization using preset fits. They often work, though not always. 


def init():
## This function simply reads the polymer as an input,
## and that's it. It's the first user input when 
## code is executed
	
	## Ask for polymer type
	polymer = input("Please type polymer (e.g. PVC, PS, PMMA): ")
	## Verify that code recognizes the polymer
	if polymer == "PVC":
		print("Polyvinyl Chloride")
		return polymer
	if polymer == "PS":
		print("Polystyrene")
		return polymer
	if polymer == "PET":
		print("Polyethylene terephthalate")
		return polymer
	if polymer == "PP":
		print("Polypropylene")
		return polymer
	if polymer == "PTFE":
		print("Polytetrafluoroethylene")
		return polymer
	if polymer == "PMMA":
		print("Poly(methyl methacrylate)")
		return polymer
	else: 
		print("Polymer not recognized")
		return False


def read_file(file_name):
## This function reads in the .csv file after the file name has
## been given. It then calls the data correction function, which handles
## normalization. Data correction includes subtraction of blank scan, 
## interpretation of non-data from .csv, and normalization of attenuation.

	## Change into the data directory
	os.chdir("../Data")
	
	## Obtain headers from file
	header = np.genfromtxt("{0}".format(file_name), delimiter=',', dtype=str, max_rows=2)
	
	## Get number of scans from the length of the first row, minus the extra comma at the end
	row1 = header[0,:]
	num_of_scans = int((row1.shape[0] - 1) / 2)
	print("This file contains {0} scans (including blank).".format(num_of_scans))

	## Generate data
	#### The skip_footer can throw an error; 27-29 usually works. It depends on scan settings
	data = np.genfromtxt("{0}".format(file_name), delimiter=",", dtype=float, skip_header=2, skip_footer=29*num_of_scans) 

	## Remove the trailing column of Nan's
	#### This is a result of the trailing commas
	data = np.delete(data,num_of_scans * 2,1)
	
	## Subtract out the noise from the blank scan
	#### Data is stored in every even row (or I guess odd since 0-indexed)
	data[:,1::2] -= data[:,1,None]

	## Remove the first two columns, since it is just the blank
	for i in range(2):
		data = np.delete(data,0,1)
		header = np.delete(header,0,1)
	
	## Remove blank scan from the count
	num_of_scans -= 1

	## Normalize data (see normalize function)
	data = normalize(data,num_of_scans)	

	## Plot to verify normalization
	color = iter(plt.cm.rainbow(np.linspace(0, 1, num_of_scans)))	
	plt.figure()
	for j in range(num_of_scans):
		c = next(color)
		plt.scatter(data[:,2*j],data[:,2*j+1],label=header[0,2*j],s=0.5,color=c)
		plt.title("Corrected Data")
		plt.legend()
	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Absorbance (au)")
	plt.show()

	return header, data, num_of_scans


def normalize(data,num_of_scans):
## This function uses the relative min and max of the data
## to then extract just the sinusoid, without the baseline 
## absorption interfering with the data
	
	## Choose how many data points to leave out. 100-300 is good
	cut = 200

	## Ignore the noisy high wavelength stuff
	temp1 = data[cut:,:]

	## Iterate through each scan
	for j in range(num_of_scans):
		
		## Smooth out scan using scipy module
		temp2 = savgol_filter(temp1[:,2*j+1],61,2)

		## Plot the smoothed data, to see if smoothing works fine
		#plt.figure()
		#plt.plot(temp1[:,2*j],temp1[:,2*j+1],label="Data")
		#plt.plot(temp1[:,2*j],temp2,label="Smoothed")
		#plt.title("Raw and Smoothed Data")
		#plt.legend()

		## Find index for local min and max
		ind_max = argrelmax(temp2,order=30)[0]
		ind_min = argrelmin(temp2,order=30)[0]

		## Get data points for the max and min locations
		minwav = temp1[:,2*j][ind_min]
		minabs = temp1[:,2*j+1][ind_min]
		maxwav = temp1[:,2*j][ind_max]
		maxabs = temp1[:,2*j+1][ind_max]
	
		## Ignore cases where there is only one min
		if len(ind_min) > 1:

			## Let's fit the minima!
			fit1 = fitbot(data,temp1,temp2,minwav,minabs,maxwav,maxabs,j)
			
			## Now subtract the fit
			data[:,2*j+1] -= fit1
			
			## Ignore cases where there is only one max
			if len(ind_max) > 1:
				
				## Let's fit the maxima!				
				fit2 = fittop(data,temp1,temp2,minwav,minabs,maxwav,maxabs,ind_max,j)

				scale = np.max(maxabs) / fit2
				
				## Use fit to scale data
				temp3 = np.copy(data[:,2*j+1])
				data[:,2*j+1] *= scale
		
		## Otherwise do subtraction using minumum 		
		else:
			temp = data[cut:,:]
			data[:,2*j+1] -= np.amin(temp[:,2*j+1])
	
	
	return data	


def fitbot(data,temp1,temp2,minwav,minabs,maxwav,maxabs,j):
## Function to set up the minima fitting, so that there can a baseline
## subtraction applied to the data. This is a subtractio, while the next
## step is a scaling of the data

	## Set up fits from the different models
	rayc = np.polyfit(minwav**(-4),minabs,1) ## Rayleigh
	ray = rayc[0] * (data[:,2*j]**(-4)) + rayc[1]	
	
	linc = np.polyfit(minwav,minabs,1) ## Linear
	lin = linc[0] * data[:,2*j] + linc[1]
	
	decc = np.polyfit(minwav,np.log(minabs),1) ## Decaying exponential
	dec = np.exp(decc[1])*np.exp(decc[0]*data[:,2*j])
	
	isqc = np.polyfit(minwav**(-0.5),minabs,1) ## 1/sqrt(2)
	isq = isqc[0] * (data[:,2*j]**(-0.5)) + isqc[1]
	
	## Begin figure
	fig,ax = plt.subplots()
	plt.scatter(data[:,2*j],data[:,2*j+1],label="Data",s=0.5)
	plt.scatter(minwav,minabs,c="b")
	plt.scatter(maxwav,maxabs,c="b")
	plt.legend()
	plt.title("Use number keys 1-4 to modify fit")
	plt.xlabel("Wavelength (nm) \n 1 = Rayleigh, 2 = Linear, 3 = Exponential, 4 = Inverse Square Root")
	plt.ylabel("Absorbance (au)")
	
	## Default fit (zero)
	fit = np.zeros(len(data[:,j]))
	line, = plt.plot(data[:,0],fit,linewidth=1,c="g")

	## Update on letter click	
	def on_press(event):
		if event.key == "1":
			line.set_ydata(ray)
			fit = ray
			fig.canvas.draw_idle()
		if event.key == "2":
			line.set_ydata(lin)
			fit = lin
			fig.canvas.draw_idle()
		if event.key == "3":
			line.set_ydata(dec)
			fit = dec
			fig.canvas.draw_idle()
		if event.key == "4":
			line.set_ydata(isq)
			fit = isq
			fig.canvas.draw_idle()
	
	## Track clicks on plot screen
	fig.canvas.mpl_connect("key_press_event",on_press)

	plt.show()

	print("Give number of best fit: ")
	num = inp()
	if num == "1":
		return ray
	if num == "2":
		return lin
	if num == "3":
		return dec
	if num == "4": 
		return isq


def fittop(data,temp1,temp2,minwav,minabs,maxwav,maxabs,ind_max,j):
## Function to set up the minima fitting

	## Update max values for absorbance
	maxabs = temp1[:,2*j+1][ind_max]
	
	## Fit to something that works
	irac = np.polyfit(-maxwav**(-4),maxabs,1)	## Inverse of Rayleigh
	ira = - irac[0] * (data[:,2*j]**(-4)) + irac[1]
	
	linc = np.polyfit(maxwav,maxabs,1)		## Linear
	lin = linc[0] * data[:,2*j] + linc[1]

	decc = np.polyfit(maxwav,np.log(-maxabs+1.1*np.amax(maxabs)),1)	## Inverted decaying exponential
	dec = -np.exp(-decc[1])*np.exp(decc[0]*data[:,2*j])+1.1*np.amax(maxabs)
	
	isqc = np.polyfit(-maxwav**(-0.5),maxabs,1)	## 1/sqrt(2)
	isq = - isqc[0] * (data[:,2*j]**(-0.5)) + isqc[1]
	
	## Use fit to scale data
	temp3 = np.copy(data[:,2*j+1])
	
	## Begin figure
	fig,ax = plt.subplots()
	plt.scatter(data[:,2*j],temp3,label="Data",s=0.5)
	plt.scatter(maxwav,maxabs,c="b")
	plt.legend()
	plt.title("Use number keys 1-4 to modify fit")
	plt.xlabel("Wavelength (nm) \n 1 = Rayleigh, 2 = Linear, 3 = Exponential, 4 = Inverse Square Root")
	plt.ylabel("Absorbance (au)")
	
	
	## Default fit
	fit = np.zeros(len(data[:,j]))
	line, = plt.plot(data[:,0],fit,linewidth=1,c="g")

	## Update on letter click	
	def on_press(event):
		if event.key == "1":
			line.set_ydata(ira)
			fit = ira
			fig.canvas.draw_idle()
		if event.key == "2":
			line.set_ydata(lin)
			fit = lin
			fig.canvas.draw_idle()
		if event.key == "3":
			line.set_ydata(dec)
			fit = dec
			fig.canvas.draw_idle()
		if event.key == "4":
			line.set_ydata(isq)
			fit = isq
			fig.canvas.draw_idle()
	
	## Track clicks on plot screen
	fig.canvas.mpl_connect("key_press_event",on_press)

	plt.show()

	print("Give number of best fit: ")
	num = inp()
	if num == "1":
		return ira
	if num == "2":
		return lin
	if num == "3":
		return dec
	if num == "4": 
		return isq


def mcmc(data, polymer, num_of_scans):
## This function uses a Metropolis-Hastings algorithm in order
## to optimize the fit parameters. It returns the optimal 
## parameters as well as histograms describing the parameter space

	## Data points to cut out due to noise at long wavelength
	cut = 100

	## Obtain initial guess
	print("Provide guess for thickness and scaling, with Enter in between: ")
	dguess = float(inp())
	aguess = float(inp())
	guess = dguess,aguess	

	## This value essentially dictates how risky the Markov chain is.
	## A lower value is less risky, though it can cause runtime issues
	sigy = 0.10

	## Set up relevant MCMC parameters
	#### T is the number of iterations. burn is the burn in.
	#### delta is the maximum jump size; i.e. step size. 	
	T = 30000
	burn = 5000
	delta = [100,0.15]
	t = np.zeros((2,num_of_scans))	
	
	accept = 0

	for j in range(num_of_scans):

		print("###########################")
		
		## Initialize the parameters
		theta = np.zeros((2,T))
		theta[:,0] = guess
		prob = np.zeros(T)	
		
		## Loop for the desired number of iterations
		for i in range(T-1):

			## Copy over value
			theta[:,i+1] = theta[:,i]
			
			## Get random numbers to perturb and check
			u = np.random.rand(2)
			utest = np.random.rand(1)

			## Perturb
			theta[:,i+1] += delta*(2*u-1)
			
			## Get change
			new = getP(theta[:,i+1],data[cut:,:],polymer,j,guess,sigy)
			old = getP(theta[:,i],data[cut:,:],polymer,j,guess,sigy)
			diff = new / old	
			
			## Accept condition
			if utest <= diff:	
				prob[i+1] = new
				accept += 1
			
			## Reject condition
			else:
				theta[:,i+1] = theta[:,i]
				prob[i+1] = prob[i]
		
		## Remove burn in
		theta = theta[:,burn:]
		prob = prob[burn:]
	
		## Plot 1d histograms and quality assurance
		names = ["d","a"]
		xaxes = ["Thickness (nm)","Scaling Factor (au)"]
		
		for k in range(2):
			plt.subplot(2,2,k+1)
			plt.hist(theta[k,:],bins=35)
			plt.title("MCMC Samples for {0}".format(names[k]))
			plt.xlabel(xaxes[k])
			plt.ylabel("Counts")

			plt.subplot(2,2,k+3)
			plt.plot(theta[k,:],np.arange(burn,T),linestyle='-',color='black',linewidth=1.0)
			plt.title("Quality Assurance for {0}".format(names[k]))
			plt.xlabel(names[k])
			plt.ylabel("Iteration #")

		plt.tight_layout()

		## Plot 2d histogram 
		plt.figure()
		plt.hist2d(theta[0,:],theta[1,:],bins=100)
		plt.show()
		
		## Print out best value attained
		best = np.where(prob == np.amax(prob))
		best = best[0][0]
		print("Best parameters found: d = {0}, a = {1}".format(np.around(theta[0,best],2),np.around(theta[1,best],3)))	
	
		## Print out the median values for assistance
		print("Enter bounds for stats with Enter inbetween, or m for manual: ")
		low = inp()
		if low != "m":
			low = float(low)
			high = float(inp())
			temp = theta[0,:]
			mask = np.where((temp>low) & (temp<high))
			temp = temp[mask]
			print("d: median= {0}, mean= {1} ## Caution: may not work for asymmetric output".format(np.around(np.median(temp),2),np.around(np.mean(temp),2)))
		
		if low == "m":
			print("d: median= {0}, mean= {1} ## Caution: may not work for asymmetric output".format(np.around(np.median(theta[0,:]),1),np.around(np.mean(theta[0,:]),1)))	
		print("a: median= {0}, mean= {1} ## Caution: may not work for asymmetric output".format(np.around(np.median(theta[1,:]),3),np.around(np.mean(theta[1,:]),3)))	

		## Finally, collect the output parameters
		print("Select the best fit parameters, with Enter in between: ")
		t[0,j] = float(inp())
		t[1,j] = float(inp())

	
	## A good acceptance rate is 0.25-0.4
	print("Acceptance rate: ", np.around(accept / (T*num_of_scans),4))
	
	return t	


def plot_spectra(header, data, num_of_scans, file_name, t, polymer):
## This function plots the data as a spectrum, and the fits from MCMC 
## It then asks to save the plot to a nearby directory

	## Begin figure
	plt.figure()
	ax = plt.gca()

	## Load color wheel so fit matches data in color
	color = iter(plt.cm.rainbow(np.linspace(0, 1, num_of_scans)))	

	## Plot the data and fit, together with parameter values
	for i in range(num_of_scans):
		c = next(color)
		plt.scatter(data[:,i*2],data[:,i*2+1],label=header[0,i*2],s=0.5,color=c)
		plt.plot(data[:,i*2],t[1,i]*func((data[:,0],n(polymer,data[:,0])),t[0,i]),linewidth=1,c=c)
		plt.text(0.5,0.9-i*0.05, "{0}: d = {1} nm, a = {2}".format(header[0,i*2],t[0,i],t[1,i]),transform=ax.transAxes)
	
	## Plot labels and stuff
	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Absorbance (a.u.)")
	plt.title("Absorbance vs. Wavelength")
	plt.legend()
	
	plt.tight_layout()
	
	fig = plt.gcf()

	plt.show()	

	## Save the plots to folder
	y_or_n = input("Would you like to save plot?(y/n) ")
	
	## Jumping into correct directory
	os.chdir("./Plots/MCMC")
	if y_or_n == "y":
		fig.savefig(file_name[:-4] + "_MCMC.pdf", dpi=200)


def func(X,d):
## This is the model function to fit to. X is a tuple
## of wavelength (x) and index of refraction (n)

	x,n = X
	r = ( (n - 1) / (n + 1) ) ** 2
	return -np.log10((1 - r)**2 / (1 + r**2 - 2*r*np.cos(4*np.pi*n*d/x))) 


def inp():
## Input function to determine if input is valid
	done = False
	while done == False:
		temp = input()
		test = temp.replace(".","")
		if (not test.isdigit()):
			print("Try Again")
		else:
			return temp



	
def getP(theta,data,polymer,j,guess,sigy):
## Uses inverse exponential in order to obtain a probability fron the 
## square of the residuals. This is a frequentist probability
	
	## Unpack d and a
	d,a = theta[0],theta[1]

	## Calculate frequentist probability, and scale in case of run time error
	p = np.prod(np.exp( -  res2(theta, data, polymer, j) / sigy ))
	
	## In case division is failing:
	#p *= 100000

	p *= prior(theta,guess)

	return p
	

def prior(theta,guess):
## Prior PDFs of the parameters. Gaussian about the guess, with
## a cut off at zero thickness. Then the scaling factor will 
## just be above zero. 

	## Define d and a
	d = theta[0]
	a = theta[1]

	## Probability of d according to prior
	if d <= 0.1:
		prob_d = 10e-64
	else:
		#prob_d = 1	
		z_d = (d - guess[0]) / 300
		prob_d = norm.pdf(z_d)

	## Probability of a according to prior
	if a <= 0.1:
		prob_a = 10e-64
	else: 
		#prob_a = 1
		z_a = (a - guess[1]) / 0.4
		prob_a = norm.pdf(z_a)

	return prob_d * prob_a


def res2(theta, data, polymer, i):
## The following commands return the residuals between data and guess. 
## the guess includes both a thickness and a scaling factor. Each function 
## can be modified to accept different forms of residuals.  

	## Unpack d and a	
	d, a = theta

	## Use model to get the ideal interference pattern
	interPattern = a * func((data[:,0],n(polymer,data[:,0])),d)

	## Calculate square residual at each data point
	res2 = np.square(data[:,2*i+1] - interPattern) 
	
	return res2


def n(polymer,wavelength):
## This function takes the inputted polymer and wavelength,
## and then returns the wavelength dependent index of refraction
## as an appropriately long vector

	## Use cauchy formula and data from literature to extract index of refraction
	if polymer == "PVC":
		return 1.5262 + (5593 / (wavelength ** 2))
	if polymer == "PS":
		return 1.5718 + (8412 / (wavelength ** 2)) + ((2.35e8) / (wavelength ** 4))
	if polymer == "PMMA":
		return np.ones(wavelength.shape[0]) * 1.482
	#if polymer == "PP":
	#if polymer == "PET":
	#if polymer == "PTFE":

	
	




def main():
	
	## Initialize
	polymer = init()

	## Confirm that polymer is supported by code
	if polymer == False:
		print("Unsupported Polymer")
		return
	
	## Read file
	file_name = input("Please provide file name. ")
	header, data, num_of_scans = read_file(file_name)

	## Use MCMC to find optimal parameters
	t = mcmc(data, polymer, num_of_scans)

	## Create plots
	plot_spectra(header, data, num_of_scans, file_name, t, polymer)

	## Print the calculated thickness
	for i in range(num_of_scans):
		print("To minimize residuals in {0}, [d, a] = ".format(header[0,2*i]), t[:,i])
	print("Avg. = ", np.around(np.mean(t[0,:]),2), " nm")

main()



## Extra stuff that was previosuly being used. Fairly useless. 

#def getL(data,sigy,theta,polymer,j,guess):
### Returns the frequentist probability of the model. It is 
### used on the RHS of Bayes' Theorem
#   
#	## Choose pb
#	pb = 0.1
# 
#	## Initiate L
#	L = 1 
#
#	L = getprob(data,sigy,theta,polymer,j)
#	#L *= ( (1-pb)*getprob(x[i],y[i],sigy[i],[m,b]) + pb*getprob(1,y[i],np.sqrt(vb+(sigy[i])**2),[yb,0]) )	
#	L *= prior(theta,guess)
#	return L

#def min_res(data, polymer, num_of_scans):
### This function applies scipy.optimize.least_squares to 
### minimize the residual function by adjusting the thickness
### and a scaling factor
#
#	## Obtain initial guess
#	print("Provide guess for thickness and scaling, with Enter in between: ")
#	dguess = float(inp())
#	aguess = float(inp())
#	guess = dguess,aguess
#
#	## Initiate varaible to store outputs
#	t_r = np.zeros((2,num_of_scans))
#
#	## Iterate several times to improve outputs
#	for i in range(num_of_scans):
#		#opt = least_squares(chi2, guess, bounds=([0,0],[np.inf,np.inf]), args = (data,polymer,i))
#		opt = basinhopping(chi2, guess, niter=1000, T=0.1, minimizer_kwargs = {"args":(data,polymer,i,)})
#		t_r[0,i] = opt.x[0]
#		t_r[1,i] = opt.x[1]
#
#	return t_r

#def get_thick(data, num_of_scans, polymer):
### This function applies scipy.optimize.curve_fit to fit the
### desired function
#	
#	## Produce empty array to populate with thickness measurements
#	t = np.zeros((2,num_of_scans))
#
#	## Retrieve a thickness via least squares from each trial
#	for i in range(num_of_scans):
#		popt, pcov = curve_fit(func, (data[:,0],n(polymer,data[:,0])), data[:,2*i+1], bounds=(0,[np.inf]))
#		
#		t[0,i] = popt[0]
#		t[1,i] = pcov[0,0]
#
#	return t

#def getprob(data,sigy,theta,polymer,j):
#	
#	d = theta[0]
#	a = theta[1]
#
#	prob = np.exp(-((data[:,j]-a*func((data[:,0],n(polymer,data[:,0])),d)**2)/(2*sigy**2)))
#	print(-((data[:,j]-a*func((data[:,0],n(polymer,data[:,0])),d)**2)/(2*sigy**2)))
#	prob = np.prod(prob)
#
#	return prob


