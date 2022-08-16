import numpy as np
import matplotlib.pyplot as plt
import os

## Manually loading in the data for all of our films so that it can be plotted
data = np.zeros((2,29))
data[:,0] = [1.099,157]
data[:,1] = [1.099,388]
data[:,2] = [1.099,486]
data[:,3] = [1.099,160]
data[:,4] = [1.099,233]
data[:,5] = [1.197,272]
data[:,6] = [1.197,177]
data[:,7] = [1.197,195]
data[:,8] = [1.197,250]
data[:,9] = [1.197,410]
data[:,10] = [1.309,536]
data[:,11] = [1.309,469]
data[:,12] = [1.406,324]
data[:,13] = [1.406,662]
data[:,14] = [1.689,553]
data[:,15] = [1.689,879]
data[:,16] = [1.798,1030]
data[:,17] = [1.798,567]
data[:,18] = [1.908,590]
data[:,19] = [1.908,296]
data[:,20] = [1.908,601]
data[:,21] = [1.908,664]
data[:,22] = [1.999,621]
data[:,23] = [1.500,714]
data[:,24] = [1.500,569]
data[:,25] = [1.500,736]
data[:,26] = [1.602,736]
data[:,27] = [1.602,619]
data[:,28] = [1.602,788]

## By propagation of errors
delc = np.ones(len(data[0,:])) * 0.002

dela = [3,
	6,
	5,
	1,
	2,
	6,
	9,
	2,
	8,
	8,
	6,
	6,
	2,
	5,
	3,
	7,
	101,
	13,
	4,
	1,
	14,
	17,
	12,
	7,
	8,
	10,
	9,
	9,
	7]

plt.errorbar(data[0,:],data[1,:],dela,fmt="o",capsize=5,c="black",markersize=0.5)
plt.xlabel("Concentration (wt%)")
plt.ylabel("Thickness (nm)")
plt.title("PVC Thin Films; Slip-Coating")
fig = plt.gcf()
plt.show()


y_or_n = input("Save plot? ")
if y_or_n == "y":
	os.chdir("../Random/")
	fig.savefig("PVC Distribution", dpi=200)	






