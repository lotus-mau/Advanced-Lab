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
ncounts1 = norm(np.concatenate((Lcounts,Rcounts)))
ncounts1_err = np.sqrt(np.concatenate((Lcounts,Rcounts)))/np.max(np.concatenate((Lcounts,Rcounts)))

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
plt.plot(x,66*np.cos(x)**2,label="Model") # why A = 66?
plt.legend()

#%% Our Code

angles = np.deg2rad([90, 100, 115, 130, 
                     145, 145, 
                     160, 
                     170, 170,
                     180, 180, 180,
                     190, 190, 190, 
                     200, 200, 200,
                     215, 230, 245, 260, 270])
counts = np.array([4, 1, 14, 20,        # 0-3
                   18, 39,              # 4-5
                   51,                  # 6
                   47, 53,              # 7-8
                   47, 30, 0,           # 9-11
                   14, 39, 60,          # 12-14
                   34, 36, 0,           # 15-17
                   41, 24, 8, 13, 4])   # 18-22
counts_err = np.sqrt(counts)

# cutting out bad data points
cut_idx = [4, 7, 9, 10, 11, 12, 13, 15, 16, 17, 21] 

angles_cuts = angles[cut_idx]
counts_cuts = counts[cut_idx]
counts_err_cuts = counts_err[cut_idx]

angles = np.delete(angles, cut_idx)
counts = np.delete(counts, cut_idx)
counts_err = np.delete(counts_err, cut_idx)

plt.errorbar(angles, counts, counts_err, 
             linestyle="", marker=".", label="Data")
plt.errorbar(angles_cuts, counts_cuts, counts_err_cuts,
             linestyle="", marker=".", label="Cut Data")
plt.legend()

# plotting normalized points

ncounts = counts/np.max(counts)
ncounts_err = counts_err/np.max(counts)

ncounts_cuts = counts_cuts/np.max(counts)
ncounts_err_cuts = counts_err_cuts/np.max(counts)

# fitting normalized points
popt, _ = curve_fit(func, angles, ncounts)

model = lambda x, a: a*np.cos(x)**2

plt.figure()
# -90 to 90
plt.errorbar(angles1, ncounts1, ncounts1_err, 
             linestyle="",marker=".",label="Normalized Data")
# 90 to 270
plt.errorbar(angles, ncounts, ncounts_err,
             linestyle="",marker=".",label="Normalized Data")
# Cut points
plt.errorbar(angles_cuts, ncounts_cuts, ncounts_err_cuts,
             linestyle="", marker=".", label="Normalized Cut Data")

# Model
plt.plot(x, model(x,1), label='Normalized Model')
# Fit
plt.plot(x, model(x, popt[0]), label='Fit')

plt.legend()



plt.show()