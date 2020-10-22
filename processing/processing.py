from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from itertools import chain, combinations
import os
import numpy as np

SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60

def getSheet(name):
    """Read a csv into a Pandas DataFrame"""
    
    csv = pd.read_csv(name)
    return pd.DataFrame(csv)

# locations returns, given the number of sheets, 
# the lower and upper bounds for each trial to 
# select from. 
def locations(name):
    csv = pd.read_csv(name)
    df = pd.DataFrame(csv)[['filename', 'start_stamp', 'frames_after', 'height']]
    i = list(np.where(df['filename'] == 'Subject_02_P2_Zone12_T1')[0]) + list(np.where(df['filename'] == 'Subject_02_P2_Sit_T1')[0])
    df = df.drop(i)
    return df.loc[:, :'frames_after'], df.loc[:, 'height']

# seconds_in_day returns, given an epoch,
# the number of seconds since the start
# of that day.
def seconds_in_day(epoch):
    d = datetime.utcfromtimestamp(epoch)
    return d.hour * SECONDS_IN_HOUR + d.minute * SECONDS_IN_MINUTE + d.second

def find_start_index_old(trial, start):
    stamps = trial.loc[:, 'Time_stp'].map(lambda x: seconds_in_day(x))
    index = stamps.eq(start.loc['start_stamp']).idxmin()
    index += start.loc['frames_after']
    return index


def find_start_index(trial, start):
    stamps = trial.loc[:, 'Time_stp'].map(lambda x: seconds_in_day(x))
    index = (stamps >= start.loc['start_stamp']).idxmax()
    if stamps[0] <= start.loc['start_stamp']: # only add frames after if the lift started during the sequence
        index += start.loc['frames_after']
    return index

sheet_count = 360 # number of total trials

# process of converting the trials into vectors to train on
def preprocess(directory, start, offset, low_butter, hi_butter, do_filter=False, allowed_sensors=slice(None, None, None)):
    sheets = []
    slices, height = locations('./metadata/lift_times_untrimmed.csv')
    max_size = 0
    starts = []

    for i, trial in slices.iterrows():
        # extract columns
        df = getSheet(os.path.join(directory, trial.loc['filename'] + '.csv'))
        starts.append(find_start_index(df, trial))
        if start == 'min' and offset == 'max':
            df = df.iloc[:, :-1]
        else:
            # start of the lift
            #print(trial.loc['filename'])
            lobound = int(max(find_start_index(df, trial) + start, 0))
            #lobound = max(lobound - 10, 0)
            # end of the lift
            hibound = int(lobound + offset)

            if hibound >= len(df):
                lobound -= hibound - len(df) + 1
                hibound -= hibound - len(df) + 1
        
            # Remove timestamps (always the last column)
            df = df.iloc[lobound:hibound, :-1]
        df = df[allowed_sensors]
        #df = df[['UArm_A_x', 'UArm_A_y', 'UArm_A_z', 'UArm_G_x', 'UArm_G_y', 'UArm_G_z', 'Side_A_x', 'Side_A_y', 'Side_A_z', 'Side_G_x', 'Side_G_y', 'Side_G_z']]
        #df = df[['UArm_A_x', 'UArm_A_y', 'UArm_A_z', 'UArm_G_x', 'UArm_G_y', 'UArm_G_z']]



        #df = df[['UArm_A_x', 'UArm_A_y', 'UArm_A_z', 'UArm_G_x', 'UArm_G_y', 'UArm_G_z', 'Side_A_x', 'Side_A_y', 'Side_A_z', 'Side_G_x', 'Side_G_y', 'Side_G_z', 'Back_A_x', 'Back_A_y', 'Back_A_z', 'Back_G_x', 'Back_G_y', 'Back_G_z']]
        #df = remove_outliers(df)       
        if df.shape[0] > max_size:
            max_size=df.shape[0]

        # create a dataframe of zeros with 10 rows to pad front of lift
        #zeros = pd.DataFrame(np.zeros((10, df.shape[1])))
        #zeros.columns = df.columns

        #df = zeros.append(df)
        # apply low pass filter to reduce low-frequency noise
        # REMOVING FILTER FOR NOW
        
        # try removing outliers
        #df = remove_outliers(df)
        sheets.append(df)
    return (sheets, slices.loc[:, 'filename'], max_size, starts)