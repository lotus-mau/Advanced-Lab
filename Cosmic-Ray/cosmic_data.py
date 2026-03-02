"""

Measurements and data analysis for Cosmic Ray

Angles: -90 to 90, 90 to 270

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Helpers

def norm(x):
    x = np.array(x)
    return x/np.max(x)

def func(x,a):
    return a * np.cos(x)**2

#make a normed counts vs cos to see how well it fits to the distribution
#the scale different below might be from the distance between the plates

# Measurements of 90 to 270

angles = np.deg2rad([90, 100, 115, 130, 
                     145, 145, 
                     160, 
                     170, 170,
                     180, 180, 180, 180, 180,
                     190, 190, 190, 
                     200, 200, 200, 200,
                     215, 230, 245, 260, 270
                     ])
counts = np.array([4, 1, 14, 20,        # 0-3
                   18, 39,              # 4-5
                   51,                  # 6
                   47, 53,              # 7-8
                   47, 30, 45, 49, 42,  # 9-13      I GIVE UP.
                   14, 39, 60,          # 14-16
                   34, 36, 43, 51,      # 17-20
                   41, 24, 8, 13, 4     # 21-25
                   ])
counts_err = np.sqrt(counts)

# cutting out bad data points
cut_idx = [4, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 24] 

angles_cuts = angles[cut_idx]
counts_cuts = counts[cut_idx]
counts_err_cuts = counts_err[cut_idx]

angles = np.delete(angles, cut_idx)
counts = np.delete(counts, cut_idx)
counts_err = np.delete(counts_err, cut_idx)

popt, pcov = curve_fit(func, angles, counts)

# cos**2 distribution (90 to 270)
x2 = np.linspace(np.pi/2,3*np.pi/2,20)
cy = popt[0]*np.cos(x2)**2

plt.errorbar(angles, counts, counts_err, 
             linestyle="", marker=".", label="Data")
# plt.errorbar(angles_cuts, counts_cuts, counts_err_cuts,
#              linestyle="", marker=".", label="Cut Data")
plt.plot(x2,cy,label="Fitting")
plt.plot(x2,66*np.cos(x2)**2,label="Model") # why A = 66?

plt.xlabel(r'Angles (radians)'); plt.ylabel(r'Counts')
plt.title(r'Data Fitted to Model')

plt.legend()

# plotting normalized points [90, 270]
ncounts = counts/np.max(counts)
ncounts_err = counts_err/np.max(counts)

ncounts_cuts = counts_cuts/np.max(counts)
ncounts_err_cuts = counts_err_cuts/np.max(counts)

# fitting normalized points
popt, pcov = curve_fit(func, angles, ncounts)

model = lambda x, a: a*np.cos(x)**2

# plotting
plt.figure()

# 90 to 270
plt.errorbar(angles, ncounts, ncounts_err,
             linestyle="", marker=".", label="Normalized Data")

# Cut points

# 90 to 270
# plt.errorbar(angles_cuts, ncounts_cuts, ncounts_err_cuts,
#              linestyle="", marker=".", label="Normalized Cut Data")

# Model
plt.plot(x2, model(x2,1), label='Normalized Model')
# Fits
plt.plot(x2, model(x2, popt[0]), label='Fit2')

plt.xlabel(r'Angles (radians)'); plt.ylabel(r'Normalized Counts')
plt.title(r'Normalized Data Fitted to Model')

plt.legend()

plt.show()