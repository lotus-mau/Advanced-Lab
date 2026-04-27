"""
Millikan Oil Drop MC
Generate 100 total measurements from 10 separate oil drops.
Each drop is measured 10 times.

For each of 10 drops:
    1. Pick a random integer number of charges n from 1 to 9.
    2. Pick a reasonable fall time through 0.5 mm.
    3. Compute the fall velocity.
    4. Compute the drop radius from the fall velocity.
    5. Compute the rise velocity needed to make the total charge Q = n e.
    6. Gaussian smear the rise and fall velocities 10 times using 10% relative uncertainty.

Author: Mauricio Perez, et. al.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## CONSTANTS

V = 500.0           # Plate voltage [V]
d = 5.00e-3         # Plate separation [m]
eta = 1.84e-5       # Viscosity of air [N s / m^2
p = 1.0345e5        # Barometric pressure [Pa]
rho = 8.800e2       # Density of oil [kg / m^3] 
q_e = 1.609e-19     # Elementary charge [C]
b = 8.20e-3         # Cunningham correction constant [Pa m]
g = 9.7901          # Gravitational acceleration [m / s^2]

fall_distance = 0.5e-3  # Distance the drop falls through the reticle [m]
relative_uncertainty = 0.10     # Relative uncertainty for Gaussian smearing

num_drops = 10
measurements_per_drop = 10

## HELPERS

def radius_from_fall_velocity(v_fall):
    """
    Compute the radius of the oil drop from its terminal fall velocity.

    For a falling drop at terminal velocity, the gravitational force
    is balanced by the viscous drag force:

        mg = 6 pi eta r v_fall

    with

        m = (4/3) pi r^3 rho

    Solving gives:

        r = sqrt(9 eta v_fall / (2 g rho))

    Parameters
    ----------
    v_fall : float or array
        Fall velocity in m/s.

    Returns
    -------
    r : float or array
        Drop radius in meters.
    """
    return np.sqrt((9 * eta * v_fall) / (2 * g * rho))


def millikan_charge(v_fall, v_rise):
    """
    Compute the measured charge Q on a drop using the Millikan equation
    shown in the lab assignment:

        Q = (6 pi d / V)
            sqrt[ 9 eta^3 / (2 g rho (1 + b/(p r))^3) ]
            (v_fall + v_rise) sqrt(v_fall)

    Parameters
    ----------
    v_fall : float or array
        Fall velocity in m/s.

    v_rise : float or array
        Rise velocity in m/s.

    Returns
    -------
    Q : float or array
        Charge on the oil drop in Coulombs.
    """
    r = radius_from_fall_velocity(v_fall)

    cunningham_factor = (1 + b / (p * r))

    prefactor = (6 * np.pi * d) / V

    square_root_factor = np.sqrt(
        (9 * eta**3) / (2 * g * rho * cunningham_factor**3)
    )

    Q = prefactor * square_root_factor * (v_fall + v_rise) * np.sqrt(v_fall)

    return Q


def rise_velocity_from_charge(Q, v_fall):
    """
    Solve the Millikan equation for the rise velocity v_rise.

    Starting from:

        Q = A (v_fall + v_rise) sqrt(v_fall)

    where

        A = (6 pi d / V)
            sqrt[ 9 eta^3 / (2 g rho (1 + b/(p r))^3) ]

    Then:

        v_fall + v_rise = Q / (A sqrt\(v_fall))

    so:

        v_rise = Q / (A sqrt(v_fall)) - v_fall

    Parameters
    ----------
    Q : float
        True charge on the drop in Coulombs.

    v_fall : float
        Fall velocity in m/s.

    Returns
    -------
    v_rise : float
        Rise velocity in m/s.
    """
    r = radius_from_fall_velocity(v_fall)

    cunningham_factor = (1 + b / (p * r))

    prefactor = (6 * np.pi * d) / V

    square_root_factor = np.sqrt(
        (9 * eta**3) / (2 * g * rho * cunningham_factor**3)
    )

    A = prefactor * square_root_factor

    v_rise = Q / (A * np.sqrt(v_fall)) - v_fall

    return v_rise

## SIMULATION

# seed
np.random.seed(42)

ideal_drop_data = []

for drop_id in range(1, num_drops + 1):
    
    n_true = np.random.randint(1, 11)

    Q_true = n_true * q_e

    fall_time = np.random.uniform(9.5, 20.5)

    v_fall_true = fall_distance / fall_time

    r_true = radius_from_fall_velocity(v_fall_true)

    v_rise_true = rise_velocity_from_charge(Q_true, v_fall_true)

    # In rare cases, a randomly chosen fall time and charge can give
    # an unphysical negative rise velocity. If that happens, keep picking
    # a new fall time until the rise velocity is positive.
    while v_rise_true <= 0:
        fall_time = np.random.uniform(8.0, 25.0)
        v_fall_true = fall_distance / fall_time
        r_true = radius_from_fall_velocity(v_fall_true)
        v_rise_true = rise_velocity_from_charge(Q_true, v_fall_true)

    ideal_drop_data.append({
        "drop_id": drop_id,
        "n_true": n_true,
        "Q_true_C": Q_true,
        "fall_time_true_s": fall_time,
        "v_fall_true_m_per_s": v_fall_true,
        "v_rise_true_m_per_s": v_rise_true,
        "radius_true_m": r_true
    })


ideal_df = pd.DataFrame(ideal_drop_data)


## Gaussian smearing

simulated_measurements = []

for _, row in ideal_df.iterrows():

    drop_id = int(row["drop_id"])
    n_true = int(row["n_true"])
    Q_true = row["Q_true_C"]

    v_fall_true = row["v_fall_true_m_per_s"]
    v_rise_true = row["v_rise_true_m_per_s"]

    for measurement_id in range(1, measurements_per_drop + 1):

        # sigma = 10%
        v_fall_measured = np.random.normal(
            loc=v_fall_true,
            scale=relative_uncertainty * v_fall_true
        )

        v_rise_measured = np.random.normal(
            loc=v_rise_true,
            scale=relative_uncertainty * v_rise_true
        )

        while v_fall_measured <= 0:
            v_fall_measured = np.random.normal(
                loc=v_fall_true,
                scale=relative_uncertainty * v_fall_true
            )

        while v_rise_measured <= 0:
            v_rise_measured = np.random.normal(
                loc=v_rise_true,
                scale=relative_uncertainty * v_rise_true
            )
        
        fall_time_measured = fall_distance / v_fall_measured
        rise_time_measured = fall_distance / v_rise_measured

        Q_measured = millikan_charge(v_fall_measured, v_rise_measured)

        charge_in_e = Q_measured / q_e

        simulated_measurements.append({
            "drop_id": drop_id,
            "measurement_id": measurement_id,
            "n_true": n_true,
            "Q_true_C": Q_true,
            "v_fall_true_m_per_s": v_fall_true,
            "v_rise_true_m_per_s": v_rise_true,
            "v_fall_measured_m_per_s": v_fall_measured,
            "v_rise_measured_m_per_s": v_rise_measured,
            "fall_time_measured_s": fall_time_measured,
            "rise_time_measured_s": rise_time_measured,
            "Q_measured_C": Q_measured,
            "Q_measured_in_units_of_e": charge_in_e
        })


simulated_df = pd.DataFrame(simulated_measurements)


## RESULTS

print("\n====================")
print("IDEAL DROP DATA")
print("====================")
print(ideal_df)

print("\n====================")
print("SIMULATED MEASUREMENTS")
print("====================")
print(simulated_df)

print("\nTotal number of simulated measurements:", len(simulated_df))

ideal_df.to_csv("millikan_ideal_drop_data.csv", index=False)
simulated_df.to_csv("millikan_simulated_measurements.csv", index=False)

print("\nSaved files:")
print("  millikan_ideal_drop_data.csv")
print("  millikan_simulated_measurements.csv")

## PLOTTING

# Q/e values
plt.figure(figsize=(9, 6))
plt.scatter(
    simulated_df["drop_id"],
    simulated_df["Q_measured_in_units_of_e"],
    alpha=0.7,
    label="Smeared measurements"
)
plt.scatter(
    ideal_df["drop_id"],
    ideal_df["n_true"],
    marker="x",
    s=100,
    label="True integer charge"
)
plt.xlabel("Drop ID")
plt.ylabel(r"Measured charge $Q/e$")
plt.title("Simulated Millikan Oil Drop Data")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Q/e values histogram
plt.figure(figsize=(9, 6))
plt.hist(
    simulated_df["Q_measured_in_units_of_e"],
    bins=20,
    edgecolor="black",
    alpha=0.7
)
plt.xlabel(r"Measured charge $Q/e$")
plt.ylabel("Counts")
plt.title("Histogram of Simulated Measured Charges")
plt.grid(True)
plt.tight_layout()

# Rise v. Fall
plt.figure(figsize=(9, 6))
plt.scatter(
    simulated_df["v_fall_measured_m_per_s"],
    simulated_df["v_rise_measured_m_per_s"],
    alpha=0.7
)
plt.xlabel(r"Measured fall velocity $v_{\mathrm{fall}}$ [m/s]")
plt.ylabel(r"Measured rise velocity $v_{\mathrm{rise}}$ [m/s]")
plt.title("Rise Velocity vs Fall Velocity")
plt.grid(True)
plt.tight_layout()

plt.show()