"""
Millikan Oil Drop Analysis
Author: Mauricio Perez, et. al.
"""

import numpy as np
import matplotlib.pyplot as plt

V = 500.0             # plate voltage [V]
d = 5.00e-3           # plate separation [m]
eta = 1.84e-5         # viscosity of air [N s / m^2]
p = 1.0345e5          # barometric pressure [Pa]
rho = 8.800e2         # density of oil [kg / m^3]
b = 8.20e-3           # Cunningham correction constant [Pa m]
g = 9.7901            # gravitational acceleration [m / s^2]

e_accepted = 1.609e-19      # accepted elementary charge [C]
fall_distance = 0.5e-3      # reticle distance [m]

## DATA

"""
Each row corresponds to one measured trial.

For example, suppose you observe drop 1 ten times, then drop 2 ten times, etc.

drop_id tells the code which measurements belong to the same oil drop.

fall_time_s is the measured time for the drop to fall through 0.5 mm.

rise_time_s is the measured time for the drop to rise through 0.5 mm
when the electric field is turned on.

Replace the example values below with your real data.
"""

drop_id = np.array([
    1, 1, 
    2, 
    3, 3,
    4, 4, 4, 4, 4, 4,
    5, 5, 
    6, 6, 6, 6, 6, 
    7, 7, 7, 7, 7, 7, 7, 
    8, 8, 
    9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 
    11, 11, 11, 11, 11, 11, 11,
    12, 12, 12, 12, 12, 12, 12, 12
])

fall_time_s = np.array([
    17.6, 16.7,
    17.1, 
    17.6, 16.0,
    14.4, 12.1, 12.1, 12.6, 12.5, 13.2,
    12.1, 12.0, 
    11.4, 12.1, 11.6, 12.9, 12.2, 
    11.7, 11.0, 12.7, 12.1, 11.3, 12.1, 12.4,
     7.5,  8.1, 
     7.8,  7.5,  7.4,  7.5,  7.5,  7.8,  8.1, 
     7.9,  7.5,  7.9,  7.5,  7.6,  7.5,  7.4,  7.7, 
     7.7,  7.5,  7.5,  7.8,  7.9,  7.4,  8.0, 
    10.6, 11.4, 11.3, 11.7, 12.0, 12.4, 11.2, 11.6
])

rise_time_s = np.array([
    2.0, 2.7,
    1.8, 
    3.1, 2.2,
    1.5, 1.3, 1.5, 1.5, 1.3, 1.2,
    3.9, 4.0,
    6.1, 6.9, 6.7, 7.0, 6.7, 
    6.9, 6.7, 6.3, 6.1, 6.4, 6.3, 6.6,
    2.2, 2.2,
    4.9, 4.5, 4.8, 4.9, 5.0, 4.8, 5.2, 
    5.1, 5.0, 5.0, 4.9, 5.0, 5.0, 4.8, 5.0, 
    2.8, 2.9, 2.8, 2.7, 2.8, 2.3, 2.3,
    7.3, 7.8, 7.5, 7.6, 7.7, 7.2, 7.5, 7.1
])

## HELPERS

def radius_from_fall_velocity(v_fall):

    return np.sqrt((9 * eta * v_fall) / (2 * g * rho))


def millikan_charge(v_fall, v_rise):

    r = radius_from_fall_velocity(v_fall)

    cunningham_factor = 1 + b / (p * r)

    prefactor = (6 * np.pi * d) / V

    square_root_factor = np.sqrt(
        (9 * eta**3)
        / (2 * g * rho * cunningham_factor**3)
    )

    Q = prefactor * square_root_factor * (v_fall + v_rise) * np.sqrt(v_fall)

    return Q

# calcs

v_fall = fall_distance / fall_time_s
v_rise = fall_distance / rise_time_s

Q_measured_C = millikan_charge(v_fall, v_rise)

Q_measured_in_e = Q_measured_C / e_accepted


print("\n==============================")
print("Individual measurements")
print("==============================")

print(
    f"{'Drop':>6} "
    f"{'t_fall [s]':>12} "
    f"{'t_rise [s]':>12} "
    f"{'Q [C]':>18} "
    f"{'Q/e_acc':>12}"
)

for i in range(len(drop_id)):
    print(
        f"{drop_id[i]:6d} "
        f"{fall_time_s[i]:12.4f} "
        f"{rise_time_s[i]:12.4f} "
        f"{Q_measured_C[i]:18.5e} "
        f"{Q_measured_in_e[i]:12.4f}"
    )

# ANALYSIS per drop

unique_drops = np.unique(drop_id)

Q_mean_C = []
Q_std_C = []

Q_mean_in_e = []
Q_std_in_e = []

n_estimated = []
e_estimated_C = []
percent_error = []


for current_drop in unique_drops:

    mask = drop_id == current_drop

    Q_values_C = Q_measured_C[mask]
    Q_values_in_e = Q_measured_in_e[mask]

    N = len(Q_values_C)

    mean_C = np.mean(Q_values_C)
    mean_in_e = np.mean(Q_values_in_e)

    if N > 1:
        std_C = np.std(Q_values_C, ddof=1)
        std_in_e = np.std(Q_values_in_e, ddof=1)
    else:
        std_C = 0.0
        std_in_e = 0.0

    # Estimate integer charge number n by rounding Q/e_accepted
    n_guess = int(np.round(mean_in_e))

    if n_guess < 1:
        n_guess = 1

    # Estimate e from Q = n e
    e_guess = mean_C / n_guess

    error = abs(e_guess - e_accepted) / e_accepted * 100

    Q_mean_C.append(mean_C)
    Q_std_C.append(std_C)

    Q_mean_in_e.append(mean_in_e)
    Q_std_in_e.append(std_in_e)

    n_estimated.append(n_guess)
    e_estimated_C.append(e_guess)
    percent_error.append(error)


Q_mean_C = np.array(Q_mean_C)
Q_std_C = np.array(Q_std_C)

Q_mean_in_e = np.array(Q_mean_in_e)
Q_std_in_e = np.array(Q_std_in_e)

n_estimated = np.array(n_estimated)
e_estimated_C = np.array(e_estimated_C)
percent_error = np.array(percent_error)


print("\n==============================")
print("Drop-by-drop analysis")
print("==============================")

print(
    f"{'Drop':>6} "
    f"{'Q_mean/e':>12} "
    f"{'Q_std/e':>12} "
    f"{'n_est':>8} "
    f"{'e_estimated [C]':>18} "
    f"{'% error':>12}"
)

for i in range(len(unique_drops)):
    print(
        f"{unique_drops[i]:6d} "
        f"{Q_mean_in_e[i]:12.4f} "
        f"{Q_std_in_e[i]:12.4f} "
        f"{n_estimated[i]:8d} "
        f"{e_estimated_C[i]:18.5e} "
        f"{percent_error[i]:12.3f}"
    )

# RESULTS

e_mean = np.mean(e_estimated_C)
e_std = np.std(e_estimated_C, ddof=1)
e_sem = e_std / np.sqrt(len(e_estimated_C))

percent_error_final = abs(e_mean - e_accepted) / e_accepted * 100


print("\n==============================")
print("Final estimate of elementary charge")
print("==============================")
print(f"Accepted e:              {e_accepted:.5e} C")
print(f"Estimated e mean:        {e_mean:.5e} C")
print(f"Standard deviation:      {e_std:.5e} C")
print(f"Standard error of mean:  {e_sem:.5e} C")
print(f"Percent error:           {percent_error_final:.3f}%")

## PLOTTING

# measured charges per drop

plt.figure(figsize=(9, 6))

plt.scatter(
    drop_id,
    Q_measured_in_e,
    alpha=0.7,
    label="Individual measurements"
)

plt.xlabel("Drop ID")
plt.ylabel(r"Measured charge $Q/e_{\mathrm{accepted}}$")
plt.title("Measured Charge for Each Oil Drop")
plt.grid(True)
plt.legend()
plt.tight_layout()

# average charge per drop

plt.figure(figsize=(9, 6))

plt.errorbar(
    unique_drops,
    Q_mean_in_e,
    yerr=Q_std_in_e,
    fmt="o",
    capsize=5,
    label="Mean measured charge"
)

plt.scatter(
    unique_drops,
    n_estimated,
    marker="x",
    s=100,
    label="Estimated integer charge"
)

plt.xlabel("Drop ID")
plt.ylabel(r"Mean measured charge $Q/e_{\mathrm{accepted}}$")
plt.title("Average Charge of Each Oil Drop")
plt.grid(True)
plt.legend()
plt.tight_layout()

# measured charges histogram

plt.figure(figsize=(9, 6))

plt.hist(
    Q_measured_in_e,
    bins=20,
    edgecolor="black",
    alpha=0.7
)

plt.xlabel(r"Measured charge $Q/e_{\mathrm{accepted}}$")
plt.ylabel("Counts")
plt.title("Histogram of Measured Charges")
plt.grid(True)
plt.tight_layout()

# estimated e values histogram

plt.figure(figsize=(9, 6))

plt.hist(
    e_estimated_C,
    bins=10,
    edgecolor="black",
    alpha=0.7
)

plt.axvline(
    e_accepted,
    linestyle="--",
    label=rf"Accepted $e = {e_accepted:.3e}$ C"
)

plt.axvline(
    e_mean,
    linestyle="-",
    label=rf"Mean estimate $e = {e_mean:.3e}$ C"
)

plt.xlabel(r"Estimated elementary charge $e$ [C]")
plt.ylabel("Counts")
plt.title("Distribution of Estimated Elementary Charge Values")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()