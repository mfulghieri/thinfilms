import numpy as np
import matplotlib.pyplot as plt
import os

## Written by Matteo Fulghieri with help from Elizabeth Ryland, Natalia Powers-Riggs, and Trevor Wiechmann

## This code simply plots a spectrum using UV-vis data. The idea is to have a tool that quickly plots data,
## without any sort of correction or anything. 

def init():

	file_name = input("What is the file name? ")

	return file_name


def read_file(file_name):
## This function reads in the data, stores it in columns,
## and subtracts out the blank (first) scan.

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

	return header, data, num_of_scans


def plot_spectrum(header,data,num_of_scans):
## This is simply plotting each spectrum

	plt.figure()

	for j in range(num_of_scans):
		plt.scatter(data[:,2*j],data[:,2*j+1],label=header[0,2*j],s=0.5)

	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Absorbance (au)")
	plt.title("Absorbance Spectrum")
	plt.legend() 

	plt.tight_layout()
	
	plt.show()


def main():
	
	file_name = init()

	header, data, num_of_scans = read_file(file_name)

	plot_spectrum(header,data,num_of_scans)

main()
