
#Upload basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import os
import matplotlib.patches as mpatches
from scipy import constants
import pickle

#Determing server directory
current_directory = os.getcwd()
#print(current_directory)

#Importing the U(r) interpolation function for each sublevel
input_dir = current_directory + '/Interpolations/U(r)/'

# List of all sublevels 
sublevels = list(range(284, 347))
sublevels.append('TEST')

energy_interpolation = {}
for sublevel in sublevels:
    # Define the file name
    fname = f"U(r)_{sublevel}.pkl"

    # Define the full path of the input file
    input_path = os.path.join(input_dir, fname)

    # Read in the pkl file and store it in the dictionary
    with open(input_path, 'rb') as file:
        energy_interpolation[sublevel] = pickle.load(file)

#Importing the grad U interpolation function for each sublevel
input_dir = current_directory + '/Interpolations/grad_U/'

# List of all sublevels 
sublevels = list(range(284, 347))
sublevels.append('TEST')

grad_U = {}
for sublevel in sublevels:
    # Define the file name
    fname = f"grad_U_{sublevel}.pkl"

    # Define the full path of the input file
    input_path = os.path.join(input_dir, fname)

    # Read in the pkl file and store it in the dictionary
    with open(input_path, 'rb') as file:
        grad_U[sublevel] = pickle.load(file)

mRb = ((85*10**-3)/(602*10**21))

def Acceleration(sublevel, s):
    
    #interpolation was conducted for positive radial distance
    #S should always be positive as it is just a magnitude, this is just an error safeguard 
    F_x = -1 * grad_U[sublevel](abs(s))  
    
    #print(F_x)
    
    a_x = F_x / mRb
    
    return a_x

def scaled_timestep_conserve_energy(sublevel, distance_to_wire, r_max):
    # Find k from sublevel
    k = sublevel - 315

    # Define constants
    Rwire = 15e-6  # Radius of wire
    r_min = Rwire  # Minimum distance to wire

    min_time_step = 1e-10
    max_time_step = 100e-9

    # Calculate a normalized distance factor with log scale
    normalized_distance = max(0, min(np.log10(distance_to_wire - r_min + 1) / np.log10(r_max - r_min + 1), 1))

    # Define a scaling function based on normalized distance and k
    scaling_factor = min_time_step + normalized_distance * (max_time_step - min_time_step)

    # Adjust the scaling factor based on the state (k)
    scaling_factor /= abs(k - 32) + 1  # Adding 1 to avoid division by zero

    # Ensure the time step stays within the specified range
    scaled_step = max(min(scaling_factor, max_time_step), min_time_step)

    return scaled_step

def scaled_timestep(sublevel, distance_to_wire, r_max):
    # Find k from sublevel
    k = sublevel - 315
    
    # Define constants
    Rwire = 15e-6  # Radius of wire
    r_min = Rwire  # Minimum distance to wire
    
    min_time_step = 1e-10
    max_time_step = 100e-9
    
    # Calculate a normalized distance factor
    normalized_distance = max(0, min((distance_to_wire - r_min) / (r_max - r_min), 1))
    
    # Define a scaling function based on normalized distance and k
    scaling_factor = min_time_step + normalized_distance * (max_time_step - min_time_step)
    
    # Adjust the scaling factor based on the state (k) with a logarithmic scale
    scaling_factor /= np.log10(abs(k - 32) + 1)  # Adding 1 to avoid log(0)
    
    # Ensure the time step stays within the specified range
    scaled_step = max(min(scaling_factor, max_time_step), min_time_step)
    
    return scaled_step

def iterative_dynamics(t0, x0, vx0, y0, vy0, sublevel, delta_t0):
    # Define constants
    #sThresh1 = 20 * Rwire  # Distance threshold (used to change step size once we are closer to wire)
    g = - 9.81  # Gravitational acceleration
    
    s0 = ((x0**2 + y0**2)**0.5)
 
    # Calculate the time step based on s0
    delta_t = scaled_timestep(sublevel, s0, 0.001)
    
    a0 = Acceleration(sublevel, s0) + g

    # Calculate acceleration components
    ax0 = (a0 * x0) / s0
    ay0 = (a0 * y0) / s0

    # Update time and position
    t1 = t0 + delta_t
    x1 = x0 + vx0 * delta_t + (ax0 * delta_t**2) / 2
    vx1 = vx0 + ax0 * delta_t
    y1 = y0 + vy0 * delta_t + (ay0 * delta_t**2) / 2
    vy1 = vy0 + ay0 * delta_t

    # Calculate the new distance from the origin
    s1 = ((x1**2 + y1**2)**0.5)

    # Calculate the new acceleration
    a1 = Acceleration(sublevel, s1) + g        

    # Update the time step based on s1
    delta_t = scaled_timestep(sublevel, s1, 0.001)

    # Return the updated values
    return t1, x1, vx1, y1, vy1, a1, delta_t

def iterative_dynamics_for_energy(t0, x0, vx0, y0, vy0, sublevel, delta_t0):
    # Define constants
    #sThresh1 = 20 * Rwire  # Distance threshold (used to change step size once we are closer to wire)
    g = - 9.81  # Gravitational acceleration
    
    s0 = ((x0**2 + y0**2)**0.5)
 
    # Calculate the time step based on s0
    delta_t = scaled_timestep_conserve_energy(sublevel, s0, 0.001)
    
    a0 = Acceleration(sublevel, s0) + g

    # Calculate acceleration components
    ax0 = (a0 * x0) / s0
    ay0 = (a0 * y0) / s0

    # Update time and position
    t1 = t0 + delta_t
    x1 = x0 + vx0 * delta_t + (ax0 * delta_t**2) / 2
    vx1 = vx0 + ax0 * delta_t
    y1 = y0 + vy0 * delta_t + (ay0 * delta_t**2) / 2
    vy1 = vy0 + ay0 * delta_t

    # Calculate the new distance from the origin
    s1 = ((x1**2 + y1**2)**0.5)

    # Calculate the new acceleration
    a1 = Acceleration(sublevel, s1) + g        

    # Update the time step based on s1
    delta_t = scaled_timestep_conserve_energy(sublevel, s1, 0.001)

    # Return the updated values
    return t1, x1, vx1, y1, vy1, a1, delta_t

def perform_iterations_w_energy(initial_conditions):
    
     # Unpack initial conditions
    t0, x0, vx0, y0, vy0, sublevel, delta_t0 = initial_conditions
    
    t = 0   #initialize time
    Rwire = 15e-6  # (m) initialize min distance from origin
    collision_indicator = 0 # binary variable to store if atom hits wire
    
    # Create a list to store position values
    trajectory = []
    
    # Append the current position (x0, y0) to the trajectory list
    trajectory.append([x0, y0])
                       
    # Create energy DataFrame for later use
    energy_df = pd.DataFrame(columns=["Time", "Distance from Wire", "Kinetic Energy", "Potential Energy", "Total Energy", "% of Starting Energy"])
    
    #Calculate initial energy values
    s0 = ((x0**2 + y0**2)**0.5)
    initial_potential_energy = energy_interpolation[sublevel](s0)
    initial_kinetic_energy = 0.5 * mRb * (vx0**2 + vy0**2)
    initial_total_energy = initial_kinetic_energy + initial_potential_energy
    
    #Use list to store energy values
    initial_row = {"Time": 0, "Distance from Wire": s0 - Rwire, "Kinetic Energy": initial_kinetic_energy, "Potential Energy": initial_potential_energy, "Total Energy": initial_total_energy, "% of Starting Energy":1}
    row_list = [initial_row]
    
    # Calculate tLimit based on the Interaction type          #Stop when it passes plot limits
    # if sublevel < 300:
    #     tLimit = (abs(y0) / vy0) * 1.2
    # else:
      #  tLimit = (abs(y0) / vy0) * 2
        
    tLimit = (abs(y0) / vy0) * 1.2   #use shorter time for 'TEST' 
    
    while t < tLimit:
        
        t, x, vx, y, vy, a, delta_t = iterative_dynamics_for_energy(t0, x0, vx0, y0, vy0, sublevel, delta_t0)
        
        #Add new position values to list
        trajectory.append([x, y])
        
        # Calculate energy values
        kinetic_energy = 0.5 * mRb * (vx**2 + vy**2)
        s1 = ((x**2 + y**2)**0.5)
        potential_energy = energy_interpolation[sublevel](s1)
        total_energy = kinetic_energy + potential_energy
        energy_loss = total_energy/initial_total_energy
        # Store the energy values as a dictionary
        row = {"Time": t, "Distance from Wire": s1 - Rwire, "Kinetic Energy": kinetic_energy, "Potential Energy": potential_energy, "Total Energy": total_energy, "% of Starting Energy":energy_loss}
    
        # Add the row to the list
        row_list.append(row)
        
        #Set new conditions as initial conditions for next iteration
        t0, x0, vx0, y0, vy0, a0, delta_t0 = t, x, vx, y, vy, a, delta_t
        
        if s1 <= Rwire:
            row_list.pop()
            collision_indicator = 1
            t = tLimit
        
    # Extract x and y values from the trajectories
    x_values, y_values = zip(*trajectory)
    
    # Concatenate the list of rows into the DataFrame
    energy_df = pd.concat([pd.DataFrame(row_list)], ignore_index=True)
        
    return x_values, y_values, energy_df, collision_indicator

def perform_iterations_wout_energy(initial_conditions):
    
     # Unpack initial conditions
    t0, x0, vx0, y0, vy0, sublevel, delta_t0 = initial_conditions
    
    t = 0   #initialize time
    Rwire = 15e-6  # (m) initialize min distance from origin
    collision_indicator = 0 # binary variable to store if atom hits wire 
    
    # Create a list to store position values
    trajectory = []
    
    # Append the current position (x0, y0) to the trajectory list
    trajectory.append([x0, y0])
    
    #Calculate initial position
    s0 = ((x0**2 + y0**2)**0.5)
        
    tLimit = (abs(y0) / vy0) * 1.5   #use shorter time for 'TEST' 
    
    while t < tLimit:
        
        t, x, vx, y, vy, a, delta_t = iterative_dynamics(t0, x0, vx0, y0, vy0, sublevel, delta_t0)
        
        #Add new position values to list
        trajectory.append([x, y])
        
        # Calculate new position
        s1 = ((x**2 + y**2)**0.5)
        
        #Set new conditions as initial conditions for next iteration
        t0, x0, vx0, y0, vy0, a0, delta_t0 = t, x, vx, y, vy, a, delta_t
        
        if s1 <= Rwire:
            collision_indicator = 1
            t = tLimit
        
    # Extract x and y values from the trajectories
    x_values, y_values = zip(*trajectory)
        
    return x_values, y_values, collision_indicator
