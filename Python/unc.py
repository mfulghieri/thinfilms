import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import norm
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.optimize import curve_fit
import random as r
import os

## Written by Matteo Fulghieri with help from Elizabeth Ryland, Natalia Powers-Riggs, and Trevor Wiechmann

## This code is meant to give you the uncertainty in the thickness measurement, derived from a Monte
## Carlo simulation which uses the uncertainties in the equipment and procedure. The output is first
## a set of plots to cofnirm the least squares is finding the correct fit, and then it is a set of 
## histograms that show the distribution of possible fits given the uncertainties.

def init():
## This function simply reads the polymer as an input,
## and that's it. It's the first user input when 
## code is executed. The polymer type is used for the
## index of refraction and titling purposes
	
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
		print("Not yet supported")
		return False
	if polymer == "PP":
		print("Polypropylene")
		print("Not yet supported")
		return False
	if polymer == "PTFE":
		print("Polytetrafluoroethylene")
		print("Not yet supported")
		return False
	if polymer == "PMMA":
		print("Poly(methyl methacrylate)")
		print("Not yet supported")
		return False
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


def func(X,d,a):
## This is the model function to fit to. The model comes
## from literature. NOTE: this function is written slightly
## different than in the other codes. That is so it can 
## accept variation in the independent varaibles!

	x,n,theta = X
	r = ( (n - 1) / (n + 1) ) ** 2
	return a*(-np.log10((1 - r)**2 / (1 + r**2 - 2*r*np.cos(4*np.pi*n*d*np.cos(theta)/x))))


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


def n(polymer,wavelength):
## This function takes the inputted polymer and wavelength,
## and then returns the wavelength dependent index of refraction
## as an appropriately long vector.

	## Use cauchy formula and data from literature to extract index of refraction
	if polymer == "PVC":
		return np.ones(wavelength.shape[0]) * 1.531
	if polymer == "PS":
		return 1.5718 + (8412 / (wavelength ** 2)) + ((2.35e8) / (wavelength ** 4))
	if polymer == "PMMA":
		return np.ones(wavelength.shape[0]) * 1.482
	#if polymer == "PET":
	#if polymer == "PP":
	#if polymer == "PTFE":

def uncert(header, data, num_of_scans, file_name, polymer):
## Uses Monte Carlo to find the average uncertainty. 
	
	## Here are the uncertainties***
	delabs = 0.01 ## i.e. uncertainty in absorbance of 0.01 au
	delwav = 0.1 ## i.e. uncertainty in wavelength of 0.1 nm
	delang = 0.035 ## i.e. uncertainty in angle of incidence of 0.035 radians (~2 degrees)

	## Number of simulations
	T = 1000

	## Number of points to leave out
	cut = 100

	## Setting up the variable to store the simulated fits 
	par = np.zeros((2*num_of_scans,T))
	
	## Load color wheel so fit matches data in color
	color = iter(plt.cm.rainbow(np.linspace(0, 1, num_of_scans)))	
	
	## Iterate over each scan
	for i in range(num_of_scans):
		
		## Crucial!!!! Here are the bounds. This is from the measured thickness
		print("What was measured thickness? ")
		mea = float(inp())
		low = mea - 200
		high = mea + 200
		
		## Load each simulation
		for j in range(T):
		
			## Store data as a copy
			temp = data.copy()
		
			## Perturb every data point according to uncertainty
			for k in range(len(data[:,2*i])):
				wav = temp[k,2*i]
				abs = temp[k,2*i+1]
				temp[k,2*i+1] = np.random.normal(abs,delabs)
				temp[k,2*i] = np.random.normal(wav,delwav)
			
			## Load in the independent varaibles, but we're going to leave the first __  points out
			X = (temp[:,2*i][cut:],n(polymer,temp[:,2*i][cut:]),np.ones(len(temp[:,2*i][cut:]))*np.random.normal(0,0.035))
			Y = (func(X,300,1))

			## Now to run least squares
			popt,pcov = curve_fit(func,X,temp[:,2*i+1][cut:],bounds=([low,0.5],[high,1.5]))	
			par[2*i,j] = popt[0]
			par[2*i+1,j] = popt[1]
		
		## Plot to verify accuracy
		c = next(color)
		plt.scatter(data[:,2*i],data[:,2*i+1],s=0.5,label = header[0,2*i],color=c)	
		plt.plot(data[:,2*i],func((data[:,2*i],n(polymer,temp[:,2*i]),0),np.mean(par[2*i,:]),np.mean(par[2*i+1,:])),c=c)
		plt.title("Avg. Parameters Found")
		plt.xlabel("Wavelength (nm)")
		plt.ylabel("Absorbance (au)")	
		plt.show()

		## Now to plot simulation	
		plt.subplot(2,1,1)
		plt.hist(par[2*i,:])
		plt.title("Thickness")
	
		plt.subplot(2,1,2)
		plt.hist(par[2*i+1,:])
		plt.title("Scaler")
		
		plt.tight_layout()
		plt.show()

def main():

	## Initialize
	polymer = init()

	## Confirm that polymer is supported by code
	if polymer == False:
		return
	
	## Read file
	file_name = input("Please provide file name. ")
	header, data, num_of_scans = read_file(file_name)
	
	## Enter directory to place plots	
	os.chdir("./Plots/Manual")

	## Now to do uncertatinty
	uncert(header, data, num_of_scans, file_name, polymer)
	

	
	return

main()

