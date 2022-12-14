import numpy as np
import matplotlib.pyplot as plt

def func(X,d):
	x,n = X
	r = ( (n - 1) / (n + 1) ) ** 2
	return -np.log10( ((1 - r)**2)  / (1 + (r**2) - (2*r*np.cos(4*np.pi*n*d/x)))) 
 
def n(polymer,wavelength):
        ## Use cauchy formula and data from literature to extract index of refraction
        if polymer == "PVC":
                return np.ones(wavelength.shape[0]) * 1.531
        if polymer == "PS":
                return 1.5718 + (8412 / (wavelength ** 2)) + ((2.35*(10**8)) / (wavelength ** 4)) 
xx = [0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1]
xx = np.array(xx)
xx *= 1000
yy = [1.56135,1.55983,1.55812,1.55625,1.55491,1.55388,1.55236,1.55145,1.5501,1.5494,1.5485,1.54761,1.54692,1.54626,1.54533,1.54493,1.54389,1.54325,1.54275,1.54238,1.54137,1.54114,1.54073,1.54004,1.53987,1.53987,1.53946,1.5388,1.53812,1.53791,1.53754,1.53727,1.53732,1.53674,1.53593,1.53544,1.53569,1.53528,1.53526,1.53468,1.53569,1.53552,1.53471,1.53519,1.53362,1.5349,1.53466,1.53446,1.53413,1.53407,1.53383,1.53361,1.53341,1.53331,1.53305,1.53287,1.53283,1.53264,1.53235,1.53245,1.53235,1.5323,1.53238,1.53013,1.53015,1.53012,1.52952,1.52961,1.53008,1.53006,1.52963]
yy = np.array(yy)

coef = np.polyfit(1/xx**2,yy,1)
print(coef)
x = np.linspace(200,1100,1000)
y = coef[1] + coef[0] / x**2
plt.scatter(xx,yy)
plt.plot(x,y,c="r")
plt.tight_layout()
plt.show()
