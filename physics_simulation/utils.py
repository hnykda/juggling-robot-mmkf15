g = 98.1
import numpy as np
import pandas as pd
from plotly.graph_objs import *


def x_proj(t, x_0, v_0, alpha):
    # get x for given time
    return x_0 + v_0 * t * np.cos(alpha)

def y_proj(t, y_0, v_0, alpha):
    # get y for given time
    return y_0 + v_0 * t * np.sin(alpha) - 0.5*g*(t**2)

def get_impact_angle(x, v_0, alpha):
    # get angle of impact
    return np.arctan(np.tan(alpha) - ((g*x)/((v_0*np.cos(alpha))**2)))

def get_angle_of_velocity(v_x, v_y):
    return np.arctan(v_y/v_x)

def v_by_x(x, v_0, alpha):
    # get magnitude of velocity at x
    return np.sqrt(v_0**2 - 2*g*x*np.tan(alpha) + ((g*x)/(v_0 * np.cos(alpha)))**2)

def get_v_y(x, v_0, alpha):
    return (v_0 * np.sin(alpha)) - ((g*x)/(v_0 * np.cos(alpha)))

def get_v_x(v_0, alpha):
    # v_x is trivialy equal to v_0*cos(alpha)
    return v_0 * np.cos(alpha)

def get_velocity_info(x, v_0, alpha):
    # wrapper for information about velocity
    v_x = get_v_x(v_0, alpha)
    v_y = get_v_y(x, v_0, alpha)
    return pd.Series({"velocity_size" : v_by_x(x, v_0, alpha),
                     "v_x" : v_x,
                     "v_y" : v_y,
                     "velocity_angle" : get_angle_of_velocity(v_x, v_y),
                     "velocity_size_check" : np.sqrt(v_x**2 + v_y**2)}) 

def trajectory(t, x_0, y_0, v_0, alpha, energy_loss_ratio):
    # wrapper for a one ball exchange
    
    df = pd.DataFrame({"time":t, "x":0, "y":0})
    
    # get trajectories for x and y
    df["x"] = x_proj(df.time, x_0, v_0, alpha)
    df["y"] = y_proj(df.time, y_0, v_0, alpha)
    
    # this is the first part of the move (until ball hit the table - impact)
    df_f = df[df["y"]>0].copy()
    df_f = df_f.join(df_f.x.apply(lambda numb: get_velocity_info(numb, v_0, alpha)))
    
    # now take the rest - where the ball would go indifinetely "down"
    df_sec = df[df["y"]<=0].copy()
    
    impact_point = df_sec["x"].iloc[0] # this is x coord of impact
    impact_angle = get_impact_angle(impact_point, v_0, alpha) # get angle of impact (velocity)
    # get the size of velocity at impact point and reduced it by energy loss ratio
    impact_velocity = v_by_x(impact_point, v_0, alpha)
    impact_velocity_reduced = impact_velocity * energy_loss_ratio
    # under this angle we are going to fire another shot
    second_angle = abs(impact_angle)

    # preparation for another shot
    second_time = df_sec.time - df_sec.time.iloc[0]
    # compute another shot
    df_sec["x"] = x_proj(second_time, impact_point, impact_velocity_reduced, second_angle)
    df_sec["y"] = y_proj(second_time, 0, impact_velocity_reduced, second_angle)
    ms = (df_sec.x-df_sec.x.iloc[0]).copy() # we have to start from beginning just for right v_y values
    get_second_veloc = ms.apply(lambda numb: get_velocity_info(numb, impact_velocity_reduced, second_angle))
    df_sec = df_sec.join(get_second_veloc)
    
    # just some summary for debugging 
    info = pd.Series(
         {"x_impact_point" : impact_point,
          "initial_velocity" : v_0,
          "impact_velocity_reduced" : impact_velocity_reduced,
          "impact_velocity" : impact_velocity,
          "impact_time" : df_sec.time.iloc[0],
          "abs(impact_angle) [rad]" : second_angle,
          "abs(impact_angle) [deg]" : np.degrees(second_angle),
          "time" : df_sec[df_sec.y >= 0].time.iloc[-1],
          })

    # return only until the second hit of a ball
    res = df_f.append(df_sec[df_sec.y >= 0]).applymap(lambda x: np.round(x, 4))
    return res, info.map(lambda x: np.round(x, 4))

def init_veloc(h, J, mass, R):
    # rolling on inclined plane
    #veloc = np.sqrt((2*g*h)/(1+(2/3))) # the old one solution
    veloc = np.sqrt( (2*g*h) / ( ( J / ( mass * R**2 ) ) + 1 ) )
    return veloc

def move_from_catch_to_new_origin(res, catch_angle, impact_point, y_0, velocity_of_robot):
    # locate where the original trajectory of bounce 
    # ends, i.e where velocity has CATCH_ANGLE?
    idx_of_catch = np.argmin(np.abs(res[res.x >= impact_point].velocity_angle - catch_angle))
    catch_point = res.loc[idx_of_catch]

    comp = res.loc[:idx_of_catch] # comp will hold only what we want to

    # now we want to move to the Y_0 and X_02 is where would be the impact point
    # of the second bounce.
    catch_coordinates = np.array([catch_point["x"], catch_point["y"]])
    wished_coordinates = np.array([res.iloc[-1]["x"], y_0])
    movement_vector = abs(catch_coordinates-wished_coordinates)
    dist = np.linalg.norm(movement_vector) # distance to move in centimeters
    time_needed_for_transition = dist/velocity_of_robot # in seconds

    v_x_move = movement_vector[0]/time_needed_for_transition
    v_y_move = movement_vector[1]/time_needed_for_transition

    comp = comp.append({"x": res.iloc[-1]["x"], "y": y_0, 
                        "time": res.loc[idx_of_catch].time + time_needed_for_transition,
                        "velocity_size" : velocity_of_robot, 
                        "velocity_size_check" : np.sqrt(v_x_move**2 + v_y_move**2), 
                        "v_x" : v_x_move,
                        "v_y" : v_y_move,
                        "velocity_angle" : get_angle_of_velocity(v_x_move, v_y_move)}, 
                        ignore_index=True)
    
    return comp

def reverse_trajectory(comp):
    
    plgn = comp.copy()
    plgn.x = -plgn.x + comp.x.iloc[-1]
    plgn.time = plgn.time + comp.time.iloc[-1]
    plgn.v_x = -plgn.v_x
    plgn.velocity_angle = np.pi - plgn.velocity_angle # could be -np.pi
    
    return plgn

def df_pl(df, val, name):
    return dict(
        x=df['time'], 
        y=df[val], 
        name=name)
        #text=[str(float(x)) for x in df.time.tolist()])

def for_all(res, ls):
    
    scats = []
    for i in ls:
        scats.append(Scatter(df_pl(res, *i)))
    return scats

def plot_all(df, title):
    # wrapper
    ls = (
          ("x", "X [cm]"),
          ("y", "Y [cm]"),
          ("v_y", "Y velocity [cm/s]"),
          #("v_x", "X velocity [cm/s]"),
          ("velocity_size", "Velocity magnitude [cm]"),
          ("velocity_angle", "Velocity angle [rad]"),
        )

    layout = Layout(
        title=title,
        yaxis=YAxis(title='Various!'),
        xaxis=XAxis(title='Time [s]')
    )

    return Figure(data=for_all(df, ls), layout=layout)

def plot_traj(forward_traj, backward_traj):

    data = [
        Scatter(
            x=forward_traj['x'], # assign x as the dataframe column 'x'
            y=forward_traj['y'],
            text=[str(round(float(x), 5)) + " s" for x in forward_traj.time.tolist()],
            name="Forward"
        ),
        Scatter(
            x=backward_traj['x'], # assign x as the dataframe column 'x'
            y=backward_traj['y'],
            text=[str(round(float(x), 5)) + " s" for x in backward_traj.time.tolist()],
            name="Backward"
        )
    ]

    layout = Layout(
            title="Trajectory",
            yaxis=YAxis(title='Y [cm]'),
            xaxis=XAxis(title='X [cm]')
        )
    
    return Figure(data=data, layout=layout)

#%matplotlib inline
#import matplotlib.pylab as plt
#fig = plt.figure()
#ax = fig.gca()
#rev_traj.plot(x="x", y="y", ax=ax)
#ax.invert_xaxis()
#comp.plot(x="x", y="y",ax=ax)
    