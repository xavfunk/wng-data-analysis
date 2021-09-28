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

import warnings
warnings.filterwarnings('ignore')

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


def mean_trajectory(trajectories):
    "trajectories is iterable of trajectories as pd.dfs"

    # check if all trials are of the same shape    
    assert all(trajectory.shape[0]==trajectories[0].shape[0] for trajectory in trajectories), 'trajectories do not seem to be of same shape, please interpolate first'
    
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
    
    for block in ['Block' + str(i) for i in range(1, len(all_trials)+1)]:
        
        trials_to_conditions = {}
        
        for trial, val in scenario[block]['Trials'].items():
                
                if int(trial) in successful_trials[int(block[-1])-1]:
                    #print(block, trial, (val['Cube']['Placement'], val['Cube']['Rack']))
                    trials_to_conditions[str(trial)] = (val['Cube']['Placement'], val['Cube']['Rack'])

        blockwise_trials_to_conditions[block] = trials_to_conditions

    return blockwise_trials_to_conditions  # block : trialnumber : condition

            
        
if __name__ == "__main__":
    
    vps = ['pilot' + str(i) for i in range(4, 9)]
    #check_successes(vps)
    
    
    vp_name = 'pilot7'
    
    # load data
    blocks, scenario = data_loader(vp_name, root = 'C:/Users/xaver/Desktop/repos/wng-data-analysis/')
    #blocks, scenario = data_loader(vp_name, root = 'C:/Users/xaver/Desktop/wng-data-analysis/')
    #blocks, scenario = data_loader(vp_name, root = 'C:/Users/gffun/OneDrive/Desktop/spyder-py3XF/Hiwi/wng-data-analysis/')

    # some manual data janitoring
    #if vp_name == 'pilot7':
    blocks.pop(-1)
        
    # get the different conditions as referenced in scenario
    conditions = get_conditions(scenario, 8)
    
    # get all successful trials
#    for block in blocks:
#        #print(np.unique(block[[' Trial']].iloc[np.where(block['\t\tSuccess']==True)[0]])-1)
#        print(np.unique(block[' TrailDuration'].iloc[np.where(block['\t\tSuccess']==True)[0]])-1)
#

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

    #for block in all_trials:
        
    #    sorted_trials = sort_trials(block, conditions)
    
    sort_trials(all_trials, scenario, conditions, successful_trials)
    
    # sanity check: how do the trials look like
#    for block in all_trials:
#        for trial in block:
#            plot_trajectories([trial])
#            #print(type(trial))#.columns)
    
    # select successful trials
    
    # check how many there are
    
    # get mean trajectories
    
    # sanity check: how do the mean trajectories look like

    # get stds
    
    # time
    
    # space
    
    # plot stds (see legacy version)

    # plot across trials
    
    
    
    