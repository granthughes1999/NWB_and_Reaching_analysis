from pynwb import NWBHDF5IO
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class nwb_loader:

    #  ----------------- Load data -----------------------------
    def __init__(self,nwb_path):
        self.nwb_path = nwb_path
        self.load_nwb()

    def nwb(self):
        if self.nwb is None:
            self.load_nwb()
        return self.nwb

    def load_nwb(self):
        nwb_path = self.nwb_path
        io = NWBHDF5IO(nwb_path, 'r')
        nwb = io.read()
        self.nwb = nwb
        print(f'loaded NWB from: {nwb_path}')
        return self.nwb

    # prints the metaData associated with nwb
    def view_nwb(self):
        nwb = self.nwb
        print(nwb)

    # creates a df of the trails associated with nwb. the Data frame contains data about each trail and its structure 
    def trials(self):
        nwb = self.nwb
        df_stim = nwb.trials.to_dataframe()
        # df_stim.loc[2100:2699,'contacts'] = '10r'  #specific to this recording, fixes an error in dataframe
        self.df_stim = df_stim
        return self.df_stim

    # creates the Units Data frame associated with nwb. which contains all the data about the sorted spike units from the recording
    def units(self):
        nwb = self.nwb
        df_units = nwb.units.to_dataframe()
        self.df_units = df_units
        return self.df_units

    # creates the optogenetics_states Data frame associated with nwb. which contains all the data about the optogenetics states 
    def optogenetics_states(self):
        nwb = self.nwb
        optogenetics_states_df = nwb.intervals['optogenetics_states'].to_dataframe()
        self.optogenetics_states_df = optogenetics_states_df
        return optogenetics_states_df

    # creates the epochs Data frame associated with nwb. which contains just the entire recording length. 
    def epochs(self):
        nwb = self.nwb
        epochs_df = nwb.intervals['epochs'].to_dataframe()
        self.epochs_df = epochs_df
        return epochs_df
    
    def verify_nwb_data(self):
        # Extract data from nwb
        df_units = self.units()
        df_stim = self.trials()
        df_units.probe.unique(),df_stim.stimulus.unique()

        # Change ID to cluster_id
        df_units = df_units.reset_index()                     # New Code
        df_units = df_units.rename(columns={'id': 'cluster_id'})   # New Code

        # print unique names inside stimulus
        unit_probes = np.array(df_units.probe.unique())
        print('\n===== Total units Per Probe ====')
        for probe in unit_probes:
            df = df_units[df_units['probe']==probe]
            print(f'{probe}: ', len(df))

        # Check unique stimulus counts in df_stim
        event_names = np.array(df_stim.stimulus.unique())
        print('\n ======= Unique stimulus types ==========   : \n', event_names[0:])

        print('\n===== Total Timestamps Per Event ====')
        for event in event_names:
            df = df_stim[df_stim['stimulus']==event]
            print(f'{event}: ', len(df))

        print('\n')

        return df_stim, df_units

