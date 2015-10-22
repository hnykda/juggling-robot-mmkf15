
# coding: utf-8

# # Simulation

# In[1]:

# some necessary imports
import pandas as pd
import numpy as np

# or functions are stored in utils module
from utils import init_veloc, trajectory, move_from_catch_to_new_origin, reverse_trajectory


# # Parameters

# In[2]:

X_0 = 0 # initial x position in centimetres
Y_0 = 50 # initial height in centimetres
ALPHA = -np.pi/5 # fire angle (in radians) from x axis - MUST BE NEGATIVE
ENERGY_LOSS_RATIO = 0.7 # how much energy is lost because of bounce
J = 6.045 # http://scitation.aip.org/content/aapt/journal/tpt/50/5/10.1119/1.3703546
MASS = 2.456 
R = 1.970

# initial speed from bottle
LENGTH_OF_BOTTLE = 10 # cm
h = LENGTH_OF_BOTTLE * np.sin(-ALPHA)
# get initial speed of ball "poured" from bottle
V_0 = init_veloc(h, J, MASS, R)

# specify velocity angle under which we want to catch the ball
CATCH_ANGLE = -np.pi/3
VELOCITY_OF_ROBOT = 100  # speed of robot in cm/s


# # Computation

# In[3]:

# select time step and length
time_discret = np.arange(0,10,0.01)

# get the result for the bounce
bounce_traj, info = trajectory(time_discret, X_0, Y_0, V_0, ALPHA, ENERGY_LOSS_RATIO)
forward_traj = move_from_catch_to_new_origin(bounce_traj, CATCH_ANGLE, info["x_impact_point"], Y_0, VELOCITY_OF_ROBOT)

backward_traj = reverse_trajectory(forward_traj)
backward_traj["velocity_angle"].iloc[0] = backward_traj["velocity_angle"].iloc[0] - np.pi/2
backward_traj["flag"].iloc[0] = 1


backward_traj = backward_traj.append(forward_traj.iloc[0])
backward_traj["flag"].iloc[-3:] = 3
backward_traj.iloc[-3:].velocity_angle = backward_traj.iloc[-3:].velocity_angle - np.pi
backward_traj.iloc[-1].velocity_angle = backward_traj.iloc[-1].velocity_angle + np.pi/2


result_df = forward_traj.append(backward_traj)
result_df["time_diff"] = result_df.time.diff()
result_df["time_diff"].iloc[0] = 0
result_df = result_df.applymap(lambda x: round(x,4))
result_df.velocity_angle = np.degrees(result_df.velocity_angle)
result_df[["time_diff", "x","y", "velocity_angle", "flag"]].to_csv("trajectory.csv", header=False, index=False)

result_df.x = result_df.x - (result_df.x.max() / 2)
result_df[["time_diff", "x","y", "velocity_angle", "flag"]].to_csv("trajectory_m.csv", header=False, index=False)
