# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:30:38 2021

@author: xaver
"""


import os
import importlib
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from sklearn.utils import Bunch
import os
from scipy import signal
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from dynTimeWarp import dynTimeWarp
np.random.seed(42)

# define a few helpful functions

def data_loader(vp_name, root = os.getcwd()):#'C:/Users/xaver/Desktop/wng-data-analysis/'):
    
    """takes vp_name and root folder and returns list of blocks as dataframes and scenario dictionary"""
    
    #root = os.getcwd()
    # add vp name to root
    vp_directory = root + vp_name
    os.chdir(vp_directory)
   
    # get all blocks as dataframes andput them in list
    blocks = [pd.read_csv(file, delimiter =';') for file in os.listdir(vp_directory) if fnmatch.fnmatch(file, '*_log.csv')]
    
    # load the scenario 
    _temp = __import__('Versuchsszenario_Pilot' + vp_name[-1], globals(), locals(), ['vardict'], 0)
    scenario = _temp.vardict

    return blocks, scenario


def get_conditions(scenario, num_blocks = 8):
    """takes a Versuchsszenario and returns the relevant conditions per block as dictionary"""
    
    # init dictionary
    conditions = {}
    
    # iterate over blocks and add unique conditions to dictionary 
    for block in ['Block' + str(i) for i in range(1, num_blocks+1)]:

        s = set()
        for trial, val in scenario[block]['Trials'].items():

            s.add((val['Cube']['Placement'], val['Cube']['Rack']))
            
        conditions[block] = list(s)
    
    return conditions


def get_coi(block, add_cois = None):
    """Takes a block and extracts its columns of interest (coi)"""
    
    columns = [' \t\tRighthand_x', ' Righthand_y', ' Righthand_z',
           ' \t\tLefthand_x', ' Lefthand_y', ' Lefthand_z',
           ' Trial', ' TrialType', '\t\tZeit', ' TrailDuration', '\t\tSuccess']

    block = block[columns]
    
    new_columns = ['x_right', 'y_right', 'z_right',
           'x_left', 'y_left', 'z_left',
           'trial', 'type', 'time', 'duration', 'is_success']
    
    block.columns = new_columns

    return block

def s2f(block, numerical_coi = None):
    """Takes a block and converts numerial columns from strings to floats"""
    
    
    if numerical_coi is None:
        numerical_coi = ['x_right', 'y_right', 'z_right',
           'x_left', 'y_left', 'z_left', 'duration']
    
    
    for column in numerical_coi:    
        block[column] = block[column].str.replace(",", ".").astype(float)

    return block

def trialsplit(block):
    """
    splits a block df into a list of several trial dfs
    also removes trailing -1's and buggy ends, 
    """
    
    trials = []
    for i in range(1, max(block.trial)):
        trial = block.iloc[np.where(block.trial == i)]#[6:]  # [6:] removes leading -1s
        
        # remove leading 0s
        drop_rows = [row for row in trial.index if -1.0 in trial.loc[row,:].values]
        trial.drop(drop_rows, inplace= True)

        trials.append(trial[:-1]) # cut off buggy ends

    return trials


def plot_trajectories(trajectories):
    """
    plotting endeffector trajectories in 3D
    """
    
    # single DF
    try:
        plt.clf()
        ax = plt.axes(projection='3d')
    
        #trial = trial[6:] 
    
        ax.plot3D(trajectories.x_right, trajectories.y_right, trajectories.z_right, 'gray')
        ax.plot3D(trajectories.x_left, trajectories.y_left, trajectories.z_left, 'red')
        plt.show()
    
    # multiple DFs
    except AttributeError:
        for trajectory in trajectories:
            
            #print(len(trial))
            plt.clf()
            ax = plt.axes(projection='3d')
    
            #trial = trial[6:] 
    
            ax.plot3D(trajectory.x_right, trajectory.y_right, trajectory.z_right, 'gray')
            ax.plot3D(trajectory.x_left, trajectory.y_left, trajectory.z_left, 'red')
            plt.show()


def plot_all_trajectories(trajectories):
    """
    plotting endeffector trajectories in 3D on the same plot
    """
     
    # multiple DFs
    
    plt.clf()
    ax = plt.axes(projection='3d')

    for trajectory in trajectories:
        

        ax.plot3D(trajectory.x_right, trajectory.y_right, trajectory.z_right, 'gray')
        ax.plot3D(trajectory.x_left, trajectory.y_left, trajectory.z_left, 'red')
    
    plt.show()
            
def interpolate_trajectories(trajectories, length=1500, t=None):
    
    """
    trajectories is iterable of 3D trajectories as pd.dfs
    length is the desired length of the trajectories after interpolation
    
    assumes even smapling of data if t is not provided
    """

    # initialize newt
    #newt = np.linspace(0,length,length)
    #newt = np.arange(0,length)

    # initialize list to store trajectories
    interpolated_trajectories = []
    
    # columns of interest
    coi = ['x_right', 'y_right', 'z_right',
           'x_left', 'y_left', 'z_left']
    
    # un-elegantly allow for singular inputs
    if isinstance(trajectories, pd.DataFrame):
        trajectories = [trajectories]
    
    for trajectory in trajectories:

        # initialize empty array to store trajectory
        #traj = np.zeros((length, 3))
        traj = pd.DataFrame(columns = coi)

        #for i in range(trajectory.shape[1]):

        #    f = interp1d(np.linspace(0,1,trajectory.shape[0]), data2[i], kind = 'cubic')
        #    traj[:,i] = f(x)

        for column in trajectory[coi]:
            
            # check if eunevenly spaced time is present
            if 'right' in column:
                try:
                    t =  trajectory.t_right
                except AttributeError:
                    t = None
                
            if 'left' in column:
                try:
                    t =  trajectory.t_left
                except AttributeError:
                    t = None

            if t is None:
                #assuming evenly spaced
                t = np.linspace(0,1500,trajectory.shape[0])
                #t = np.arange(0,trajectory.shape[0])
            

            newt = np.linspace(t.min(), t.max(), length)

            f = interp1d(t, trajectory[column], kind = 'cubic', bounds_error=None)
            traj[column] = f(newt)

        interpolated_trajectories.append(traj)
    
    return interpolated_trajectories

def resample_trajectories(trajectories, length):
    
    coi = ['x_right', 'y_right', 'z_right',
           'x_left', 'y_left', 'z_left']
    
    # columns of hands
    right = ['x_right', 'y_right', 'z_right']
    left = ['x_left', 'y_left', 'z_left']
        
    # initialize list to store trajectories
    resampled_trajectories = []
    
    
    for trajectory in trajectories:

        # initialize empty array to store trajectory
        traj_resampled = pd.DataFrame(columns = coi)
        
        
        for hand in (right, left):
            
            # get hand
            hand_array = trajectory[hand].to_numpy()
            
            # get difference
            diff = np.diff(hand_array, axis = 0)
        
            # get distance
            dist = np.linalg.norm(diff, axis = 1)
        
            # get cumulative sum
            cumsum = np.hstack([[0],np.cumsum(dist)])
            
            # make x
            x = np.linspace(0,cumsum.max(),length)
            
            # resample equidistantly in space and add to traj_resampled
            #for i, coordinate in enumerate(hand):
                
            traj_resampled[hand] = np.interp2d(np.tile(x, (3,1)).T, np.tile(cumsum, (3,1)).T, hand_array)
                            
            x_hand = np.interp(x, cumsum, hand_array[:,0])
            y_hand = np.interp(x, cumsum, hand_array[:,1])
            z_hand = np.interp(x, cumsum, hand_array[:,2])
            
                
        resampled_trajectories.append(traj_resampled)
    
    
    return resampled_trajectories
    

def get_mean_trajectory(trajectories):
    "trajectories is iterable of trajectories as pd.dfs"

    # check if all trials are of the same shape    
    assert all(trajectory.shape[0]==trajectories[0].shape[0] for trajectory in trajectories), \
        'trajectories do not seem to be of same shape, please interpolate first'
    
    mean = np.zeros(trajectories[0].shape)
    
    for trajectory in trajectories:
        
        mean += trajectory
        
    mean = mean/len(trajectories)
    
    return mean


def check_successes(vps, root = 'C:/Users/xaver/Desktop/repos/wng-data-analysis/'):
    """
    iterates over vps and prints out the indeces of successful trials
    """
    
    for vp in vps:
        blocks, scenario = data_loader(vp, root)
        #blocks, scenario = data_loader(vp_name, root = 'C:/Users/gffun/OneDrive/Desktop/spyder-py3XF/Hiwi/wng-data-analysis/')

        # get the different conditions as referenced in scenario
        #conditions = get_conditions(scenario, 8)
 
        # split blocks into trials
        all_trials = []
        print('Successes of {}:'.format(vp))

        for block in blocks:
            
            block = s2f(get_coi(block))
            all_trials.append(trialsplit(block))
            
            # idxs of successful trials            
            print(np.unique(block[['trial']].iloc[np.where(block['is_success']==True)[0]])-1)

# =============================================================================
# def get_successes_all(vps, root = 'C:/Users/xaver/Desktop/repos/wng-data-analysis/'):
#     """
#     iterates over vps and returns the indeces of successful trials
#     """
#     
#     for vp in vps:
#         blocks, scenario = data_loader(vp, root)
#         #blocks, scenario = data_loader(vp_name, root = 'C:/Users/gffun/OneDrive/Desktop/spyder-py3XF/Hiwi/wng-data-analysis/')
# 
#         # get the different conditions as referenced in scenario
#         #conditions = get_conditions(scenario, 8)
#  
#         # split blocks into trials
#         all_trials = []
#         print('Successes of {}:'.format(vp))
# 
#         for block in blocks:
#             
#             block = s2f(get_coi(block))
#             all_trials.append(trialsplit(block))
#             
#             # idxs of successful trials            
#             print(np.unique(block[['trial']].iloc[np.where(block['is_success']==True)[0]])-1)
#             
# =============================================================================

def get_successes_blockwise(blocks):
    """
    iterates over vps and returns the indeces of successful trials
    """
    successes = []
    
    for block in blocks:
        
        block = s2f(get_coi(block))
        #all_trials.append(trialsplit(block))
        
        # idxs of successful trials            
        successes.append(np.unique(block[['trial']].iloc[np.where(block['is_success']==True)[0]])-1)
    
    return successes 

    
def sort_trials(all_trials, scenario, conditions, successful_trials):
    
    """
    takes as arguments the all_trials list and condtitions dict
    returns a dict matching successful trials with their condition
    TODO
    returns a dict grouping successful trials with same condition together?
    """
                   
    blockwise_trials_to_conditions = {}
    blockwise_conditions_to_trials = {}
    
    for block in ['Block' + str(i) for i in range(1, len(all_trials)+1)]:
        
        trials_to_conditions = {}
        blockwise_conditions_to_trials[block] = {}
        
        for condition in conditions[block]:
            blockwise_conditions_to_trials[block][str(condition)] = [] 
            
            
        for trial, val in scenario[block]['Trials'].items():
                
                if int(trial) in successful_trials[int(block[-1])-1]:
                    #print(block, trial, (val['Cube']['Placement'], val['Cube']['Rack']))
                    trials_to_conditions[str(trial)] = (val['Cube']['Placement'], val['Cube']['Rack'])
                    
                    # blockwise append successful trial-indeces to list of trials pertaining to one condition
                    blockwise_conditions_to_trials[block][str((val['Cube']['Placement'], val['Cube']['Rack']))].append(int(trial)-1)

        blockwise_trials_to_conditions[block] = trials_to_conditions

    return blockwise_trials_to_conditions, blockwise_conditions_to_trials # block : trialnumber : condition, block : conditon : [trialidx]

            
#__name__ = 'notmain'
if __name__ == "__main__":
    
    vps = ['pilot' + str(i) for i in range(4, 9)]
    #check_successes(vps)
    
    vp_name = 'pilot7'
    
    # load data
    blocks, scenario = data_loader(vp_name, root = 'C:/Users/xaver/Desktop/repos/wng-data-analysis/')
    #blocks, scenario = data_loader(vp_name, root = 'C:/Users/gffun/OneDrive/Desktop/spyder-py3XF/Hiwi/wng-data-analysis/')

    # some manual data janitoring
    blocks.pop(-1)
        
    # get the different conditions as referenced in scenario
    conditions = get_conditions(scenario, 8)
    
    # get successful trials
    successful_trials = get_successes_blockwise(blocks)

    # split blocks into trials
    all_trials = []
    
    for block in blocks:
        
        block = s2f(get_coi(block))
        all_trials.append(trialsplit(block))
        
        # idxs of successful trials
        #print(np.unique(block[['trial']].iloc[np.where(block['is_success']==True)[0]])-1)
        
        #durations of successful trials
        #print(np.unique(block['duration'].iloc[np.where(block['is_success']==True)[0]])-1)

    
    _, sorted_trials = sort_trials(all_trials, scenario, conditions, successful_trials)
    
    # sanity check: how do the trials look like
#    for block in all_trials:
#        for trial in block:
#            plot_trajectories([trial])
#            #print(type(trial))#.columns)
    
    for block, blockname in zip(all_trials, ['Block' + str(i) for i in range(1, len(all_trials)+1)]):
        
        for condition in conditions[blockname]:
            
            trials_of_interest = [block[i] for i in sorted_trials[blockname][str(condition)]]
            interpolated_trials = interpolate_trajectories(trials_of_interest, 1500)
            mean_trajectory = get_mean_trajectory(interpolated_trials)

            convergence = False
            
            # set warped_trajectories to start
            warped_trajectories = trials_of_interest 
            # total distance for convergence check
            tdist = 0
            
            loop = 0

            # randomly choose a trajectory to warp onto
            chosen_one = np.random.randint(0, len(trials_of_interest))
                
            while not convergence:
                                
                # interpolate trajectories
                interpolated_trials = interpolate_trajectories(warped_trajectories, 1500)
                # get mean
                mean_trajectory = get_mean_trajectory(interpolated_trials)
                # time warp
                warped_trajectories = [] # store warped trials
                # total distance for convergence check
                tdist_old = tdist
                tdist = 0
                
                # plot
                plt.clf()
                ax = plt.axes(projection='3d')
                #ax2 = plt.axes()
                for trajectory in interpolated_trials:
                    
            
                    ax.plot3D(trajectory.x_right, trajectory.y_right, trajectory.z_right, 'gray')
                    ax.plot3D(trajectory.x_left, trajectory.y_left, trajectory.z_left, 'red')
                    #ax2.plot(trajectory.x_right)
                
                ax.plot3D(mean_trajectory.x_right, mean_trajectory.y_right, mean_trajectory.z_right, 'black')
                ax.plot3D(mean_trajectory.x_left, mean_trajectory.y_left, mean_trajectory.z_left, 'purple')
                plt.title('iteration {}, condition {}, {}'.format(str(loop), str(condition), str(blockname)))
                plt.show()
                

                
                for trial in tqdm(trials_of_interest):
                    # right hand
                    x_r = trial[['x_right','y_right','z_right']].to_numpy()
                    #l_r = mean_trajectory[['x_right','y_right','z_right']].to_numpy()
                    l_r = trials_of_interest[chosen_one][['x_right','y_right','z_right']].to_numpy()

                    
                    dis, optPath = dynTimeWarp(x_r,l_r,latentStddev=None,wndlen=1000,transitions=[(1,1,np.log(2.0)),(0,1,np.log(2.0))])
                    
                    tdist += dis
                    
                    wx_r = np.array([x_r[i[1],0] for i in optPath])
                    wy_r = np.array([x_r[i[1],1] for i in optPath])
                    wz_r = np.array([x_r[i[1],2] for i in optPath])
                    wt_r = [np.arange(1500)[i[0]] for i in optPath]
                    
                    # left hand
                    x_l = trial[['x_left','y_left','z_left']].to_numpy()
                    #l_l = mean_trajectory[['x_left','y_left','z_left']].to_numpy()
                    l_l = trials_of_interest[chosen_one][['x_left','y_left','z_left']].to_numpy()

                    
                    dis, optPath = dynTimeWarp(x_l,l_l,latentStddev=None,wndlen=1000,transitions=[(1,1,np.log(2.0)),(0,1,np.log(2.0))])
                    
                    tdist += dis
                    
                    wx_l = np.array([x_l[i[1],0] for i in optPath])
                    wy_l = np.array([x_l[i[1],1] for i in optPath])
                    wz_l = np.array([x_l[i[1],2] for i in optPath])
                    wt_l = [np.arange(1500)[i[0]] for i in optPath]
                    
                    # get warped trajectories
                    warped_trajectories.append(pd.DataFrame(np.vstack((wx_r,wy_r,wz_r,wt_r,wx_l,wy_l,wz_l,wt_l)).T, columns = ['x_right','y_right','z_right','t_right','x_left','y_left','z_left','t_left']))
                
                loop += 1
                print(tdist)
                # check for convergence
                if loop == 10:
                    break
            
            
            # plot
            plt.clf()
            ax = plt.axes(projection='3d')
            #ax2 = plt.axes()
            for trajectory in trials_of_interest:
                
        
                ax.plot3D(trajectory.x_right, trajectory.y_right, trajectory.z_right, 'gray')
                ax.plot3D(trajectory.x_left, trajectory.y_left, trajectory.z_left, 'red')
                #ax2.plot(trajectory.x_right)
            
            ax.plot3D(mean_trajectory.x_right, mean_trajectory.y_right, mean_trajectory.z_right, 'black')
            ax.plot3D(mean_trajectory.x_left, mean_trajectory.y_left, mean_trajectory.z_left, 'purple')
            plt.title('resulting means and original traj., condition {}, {}'.format(str(condition), str(blockname)))
            plt.show()            
            
    
    
    
    
    
    
    # select successful trials
    
    # check how many there are
    
    # get mean trajectories
    
    # sanity check: how do the mean trajectories look like

    # get stds
    
    # time
    
    # space
    
    # plot stds (see legacy version)

    # plot across trials
    
    # # time warping
    # for trial in trials_of_interest:
    #     #x = trial.x_right.to_numpy()
    #     x = trial[['x_right','y_right','z_right']].to_numpy()
    #     #l = mean_trajectory.x_right.to_numpy()
    #     l = mean_trajectory[['x_right','y_right','z_right']].to_numpy()
        
        
    #     dis, optPath = dynTimeWarp(x,l,latentStddev=None,wndlen=1000,transitions=[(1,1,np.log(2.0)),(0,1,np.log(2.0))])
        
        
    #     plt.clf()
    #     ax = plt.axes(projection='3d')
    #     ax.plot3D(l[:,0],l[:,1],l[:,2], linewidth=3,label="target")
    #     ax.plot3D(x[:,0],x[:,1],x[:,2], label="signal")
        
    #     wx = np.array([x[i[1],0] for i in optPath])
    #     wy = np.array([x[i[1],1] for i in optPath])
    #     wz = np.array([x[i[1],2] for i in optPath])
    #     wt=[np.arange(1500)[i[0]] for i in optPath]

    #     #ax.plot3D(wx,wy,wz,color="red",label="opt. warp")
    #     plt.legend()
    #     plt.show()
                      
    #     plt.clf()
        
    #     plt.plot(l[:,1],linewidth=3,label="target")
    #     plt.plot(x[:,1],linewidth=2.0,marker="o",linestyle="--",label="signal")


    #     # wy=np.array([x[i[1]] for i in optPath])
    #     # wt=[np.arange(1500)[i[0]] for i in optPath]
    #     plt.plot(wt,wy,color="red",marker="o",label="opt. warp")
    #     plt.legend()
    #     plt.show()
    
    
    