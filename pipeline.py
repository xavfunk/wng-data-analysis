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

import warnings
warnings.filterwarnings('ignore')

# define a few helpful functions

def data_loader(vp_name, root = 'C:/Users/xaver/Desktop/wng-data-analysis/'):
    
    """takes vp_name and root folder and returns list of blocks as dataframes and scenario dictionary"""
    
    # add vp name to root
    root = 'C:/Users/xaver/Desktop/wng-data-analysis/' + vp_name
    os.chdir(root)
   
    # get all blocks as dataframes andput them in list
    blocks = [pd.read_csv(file, delimiter =';') for file in os.listdir(root) if fnmatch.fnmatch(file, '*_log.csv')]
    
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
           ' Trial', ' TrialType', '\t\tZeit', ' TrailDuration']

    block = block[columns]
    
    new_columns = ['x_right', 'y_right', 'z_right',
           'x_left', 'y_left', 'z_left',
           'trial', 'type', 'time', 'duration']
    
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
    also removes trailing -1's, 
    TODO could be done more eleganty
    """
    
    trials = []
    for i in range(1, max(block.trial)):
        trial = block.iloc[np.where(block.trial == i)][6:]  # [6:] removes leading -1s
        trials.append(trial)

    return trials


def plot_trajectories(trajectories):
    """plotting trajectories
    TODO still need to make this work for singular noniterable input"""
    
    for trajectory in trajectories:
        
        #print(len(trial))
        plt.clf()
        ax = plt.axes(projection='3d')

        #trial = trial[6:] 

        ax.plot3D(trajectory.x_right, trajectory.y_right, trajectory.z_right, 'gray')
        ax.plot3D(trajectory.x_left, trajectory.y_left, trajectory.z_left, 'red')
        plt.show()


if __name__ == "__main__":
    
    vp_name = 'pilot7'
    
    # load data
    blocks, scenario = data_loader(vp_name, root = 'C:/Users/xaver/Desktop/wng-data-analysis/')
    
    # get the different conditions as referenced in scenario
    conditions = get_conditions(scenario, 8)
    
    # split blocks into trials
    all_trials = []
    for block in blocks:
        
        block = s2f(get_coi(block))
        all_trials.append(trialsplit(block))
    
    # sanity check: how do the trials look like
    for block in all_trials:
        for trial in block:
            plot_trajectories([trial])
            #print(type(trial))#.columns)
    
    # select successful trials
    
    # check how many there are
    
    # get mean trajectories
    
    # sanity check: how do the mean trajectories look like

    # get stds
    
    # time
    
    # space
    
    # plot stds (see legacy version)

    # plot across trials
    
    
    
    