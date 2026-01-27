"""

Measurements and data analysis for Cosmic Ray

Angles: -90 to 180, 180 to 90

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% Carlos' Code (to compare ofc)

def norm(x):
    x = np.array(x)
    return x/np.max(x)

# [0,pi/2)
Rangles = np.deg2rad([0,10,20,30,35,50,65,85])
Rcounts = [53,51,46,35,32,21,7,3]

# (-pi/2,0)
Langles = np.deg2rad([-85,-65,-50,-35,-20,-10])
Lcounts = [3,8,18,34,44,50]

# Combined Data
angles1 = np.concatenate((Langles,Rangles))
ncounts = norm(np.concatenate((Lcounts,Rcounts)))
ncounts_err = np.sqrt(np.concatenate((Lcounts,Rcounts)))/np.max(np.concatenate((Lcounts,Rcounts)))

counts1 = np.concatenate((Lcounts,Rcounts))
counts1_err = np.sqrt(np.concatenate((Lcounts,Rcounts)))

def func(x,a):
    return a * np.cos(x)**2

popt, pcov = curve_fit(func,angles1,counts1)

# cos**2 distribution
x = np.linspace(-np.pi/2,3*np.pi/2,40)
cy = popt[0]*np.cos(x)**2

#make a normed counts vs cos to see how well it fits to the distribution
#the scale different below might be from the distance between the plates

plt.figure()
plt.errorbar(angles1,counts1,counts1_err,linestyle="",marker=".",label="data")
plt.plot(x,cy,label="Fitting")
plt.plot(x,66*np.cos(x)**2,label="Model")
plt.legend()

#%% Our Code

angles = np.deg2rad([100, 115, 130, 145, 160, 170, 180, 190, 200, 215, 230, 245, 260])
counts = np.array([1, 14, 20, 18, 51, 47, 47, 14, 0, 0, 0, 0, 0])
counts_err = np.sqrt(counts)

def func(x,a):
    return a * np.cos(x)**2

# popt, pcov = curve_fit(func,angles,counts)

plt.errorbar(angles, counts, counts_err, linestyle="", marker=".", label="Data")
plt.legend()



plt.show()