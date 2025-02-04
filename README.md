Date: 02-04-2025

this repository is designed to create NWBs, and also analyze reaching data from deeplab cut based behavioral videos

-----------------------------------------------
order of main files to run for Jason Christie full anaylsis (NWB + Behavior)

1. event_plots.ipynb --> allows you to make plots to see if Tones, optical pulses, and all events are correctly aligned. uses digital inputs from open-ephys to create plots

2. pre_proccess_data_forNWB.ipynb --> This will create two .json files, df_stim.json with all of the timestamps for every event(T6000, T5000, stimROI...) aligned to neuropixel data. and the df_units.json, which holds all the information for every unit from every probe. both of these files gets saved to the intermediates folder here \\Record Node 103\\experiment1\\recording1\\continuous\\intermediates

3. makeNWB.ipynb --> this takes the df_stim.json and df_units.json, and creates the final NWB

4. path_raster_NWB.ipynb --> this allows you to load in the NWB and make raster and PSTH plots for all the units and all the events

----------------------------------------------
Extra Code Packages

1.  manually_align_timestamps.ipynb --> this is for the instance when the timestamps.npy file is coruputed from open-ephys, this notebook allows you to re-create the timestamps.npy files.

2. viewNWB.ipynb --> this allows you to look at the NWB in-depth if you want

-----------------------------------------------
order of files to run to just make a general NWB

1. pre_proccess_data_forNWB.ipynb --> This will create two .json files, df_stim.json with all of the timestamps for every event(T6000, T5000, stimROI...) aligned to neuropixel data. and the df_units.json, which holds all the information for every unit from every probe. both of these files gets saved to the intermediates folder here \\Record Node 103\\experiment1\\recording1\\continuous\\intermediates

2. makeNWB.ipynb --> this takes the df_stim.json and df_units.json, and creates the final NWB
