import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as Path

def seperate_good_mua_units(df_units):

    dfA_total = df_units[df_units.probe == 'A']
    dfB_total = df_units[df_units.probe == 'B']
    dfC_total = df_units[df_units.probe == 'C']
    dfD_total = df_units[df_units.probe == 'D']
    dfE_total = df_units[df_units.probe == 'E']
    dfF_total = df_units[df_units.probe == 'F']

    # New code
    dfA_good = df_units[df_units.probe=='A'][df_units.KSlabel==2] # these are the "good" labelled units from phy before curation
    dfA_good = dfA_good.reset_index()                     # New Code
    dfA_good = dfA_good.rename(columns={'id': 'cluster_id'})   # New Code
    dfA_good

    dfA_SIM_good = dfA_good[dfA_good.Brain_Region=='SIM'] # these are the "good" labelled units from phy before curation
    dfA_IP_good = dfA_good[dfA_good.Brain_Region=='IP'] # these are the "good" labelled units from phy before curation

    dfB_good = df_units[(df_units.probe == 'B') & (df_units.KSlabel == 2)]
    dfC_good = df_units[(df_units.probe == 'C') & (df_units.KSlabel == 2)]
    dfD_good = df_units[(df_units.probe == 'D') & (df_units.KSlabel == 2)]
    dfE_good = df_units[(df_units.probe == 'E') & (df_units.KSlabel == 2)]
    dfF_good = df_units[(df_units.probe == 'F') & (df_units.KSlabel == 2)]

    dfA_mua = df_units[(df_units.probe == 'A') & (df_units.KSlabel == 1)]
    dfB_mua = df_units[(df_units.probe == 'B') & (df_units.KSlabel == 1)]
    dfC_mua = df_units[(df_units.probe == 'C') & (df_units.KSlabel == 1)]
    dfD_mua = df_units[(df_units.probe == 'D') & (df_units.KSlabel == 1)]
    dfE_mua = df_units[(df_units.probe == 'E') & (df_units.KSlabel == 1)]
    dfF_mua = df_units[(df_units.probe == 'F') & (df_units.KSlabel == 1)]

    print(f'Total units in probe A: {len(dfA_total)}')
    print(f'Good units in probe A: {len(dfA_good)} --> SIM & IP')
    print(f'MUA units in probe A: {len(dfA_mua)} --> SIM & IP\n')

    print(f'Total units in probe A SIM: {len(dfA_SIM_good)}')
    print(f'Total units in probe A IP: {len(dfA_IP_good)}\n')

    print(f'Total units in probe B: {len(dfB_total)}')
    print(f'Good units in probe B: {len(dfB_good)} --> PG')
    print(f'MUA units in probe B: {len(dfB_mua)} --> PG\n')

    print(f'Total units in probe C: {len(dfC_total)}')
    print(f'Good units in probe C: {len(dfC_good)} --> Mop')
    print(f'MUA units in probe C: {len(dfC_mua)} --> Mop\n')

    print(f'Total units in probe D: {len(dfD_total)}')
    print(f'Good units in probe D: {len(dfD_good)} --> VaL')
    print(f'MUA units in probe D: {len(dfD_mua)} --> VaL\n')

    print(f'Total units in probe E: {len(dfE_total)}')
    print(f'Good units in probe E: {len(dfE_good)} --> SnR')
    print(f'MUA units in probe E: {len(dfE_mua)} --> SnR\n')

    print(f'Total units in probe F: {len(dfF_total)}')
    print(f'Good units in probe F: {len(dfF_good)} --> RN')
    print(f'MUA units in probe F: {len(dfF_mua)} --> RN')

    return (
        dfA_good, dfA_SIM_good, dfA_IP_good, dfB_good, dfC_good, dfD_good, dfE_good, dfF_good,
        dfA_mua, dfB_mua, dfC_mua, dfD_mua, dfE_mua, dfF_mua
    )

def extract_start_times(df_stim):

    frame_events_df = df_stim[df_stim['stimulus'] == 'frame_events_timestamp']
    tone1_df = df_stim[df_stim['stimulus'] == 'tone1_timestamps']
    tone2_df = df_stim[df_stim['stimulus'] == 'tone2_timestamps']
    optical_df = df_stim[df_stim['stimulus'] == 'optical_timestamps']
    stim_ROI_df = df_stim[df_stim['stimulus'] == 'stimROI_timestamps']
    all_stimROI_triggers = df_stim[df_stim['stimulus'] == 'reachInit_stimROI_timestamps']

    tone1_start_times = tone1_df['start_time'].values
    tone2_start_times = tone2_df['start_time'].values
    frame_events_start_times = frame_events_df['start_time'].values
    stimROI_start_times = stim_ROI_df['start_time'].values
    optical_start_times = optical_df['start_time'].values
    all_stimROI_triggers_start_times = all_stimROI_triggers['start_time'].values

    print(f"Total Tone1 start_times: {len(tone1_start_times)}")
    print(f"Total Tone2 start_times: {len(tone2_start_times)}")
    print(f"Total Frame Events start_times: {len(frame_events_start_times)}")
    print(f"Total stimROI start_times: {len(stimROI_start_times)}")
    print(f"Total Optical start_times: {len(optical_start_times)}")
    print(f"Total reachInit_stimROI start_times: {len(all_stimROI_triggers_start_times)}")

    return (
        tone1_start_times,
        tone2_start_times,
        frame_events_start_times,
        stimROI_start_times,
        optical_start_times,
        all_stimROI_triggers_start_times,
    )

def seperate_closedLoop_optoTagging(
    optical_start_times,
    tone2_start_times,
    frame_events_start_times,
    total_opto_tagging_events=60,
    pulses_per_event=10
):

    total_opto_tagging_pulses = pulses_per_event

    first_opto_tagging_timestamp = optical_start_times[-600]
    end_opto_tagging_index = optical_start_times[-1]
    last_tone2_time = tone2_start_times[-1]

    start_of_opto_tagging_index = np.where(
        optical_start_times == first_opto_tagging_timestamp
    )[0][0]

    behavioral_video_duration = len(frame_events_start_times) / 150 / 60
    final_behavioral_video_time = frame_events_start_times[-1]

    final_closed_loop_optical_index = start_of_opto_tagging_index

    opto_closed_loop_start_timestamps = optical_start_times[
        0:final_closed_loop_optical_index
    ]

    opto_tag_start_timestamps = optical_start_times[
        start_of_opto_tagging_index:
    ]

    total_optical_timestamps = len(optical_start_times)
    total_optoTagging_timestamps = len(opto_tag_start_timestamps)

    last_closed_loop_start_time = opto_closed_loop_start_timestamps[-1]
    first_opto_tagging_timestamp = opto_tag_start_timestamps[0]

    first_opto_tagging_timestamp_per_trial = opto_tag_start_timestamps[::pulses_per_event]
    first_optical_pulse_per_closed_loop = opto_closed_loop_start_timestamps[::pulses_per_event]

    print('Total optical start times:', total_optical_timestamps)
    print(f'Total opto-tagging events: {total_opto_tagging_events}')
    print(f'Total opto-tagging pulses per event: {total_opto_tagging_pulses}')
    print(f'Last tone2 time: {last_tone2_time}')
    print(f'Opto-tagging start time: {first_opto_tagging_timestamp}, end time: {end_opto_tagging_index}')
    print(f'Start of opto-tagging index: {start_of_opto_tagging_index}')

    print('\nEstimated Behavioral Video duration (min):',
          round(behavioral_video_duration, 2))
    print('Final behavioral Video Frame start_time:',
          round(final_behavioral_video_time, 2))

    print('\nTotal Closed Loop optical pulses:',
          len(opto_closed_loop_start_timestamps))
    print('Total Opto-tagging optical pulses:',
          total_optoTagging_timestamps)

    if first_opto_tagging_timestamp < last_closed_loop_start_time:
        print("WARNING: Data separation issue")
    else:
        print("\n\n ------- SUCCESS -------\n Optical Pulses have been seperated into closed loop and opto-tagging groups. And it has been verified that the first opto-tagging timestamp occurs after the last closed loop timestamp. \n\n")

    return (
        total_optical_timestamps,
        total_optoTagging_timestamps,
        total_opto_tagging_events,
        total_opto_tagging_pulses,
        first_opto_tagging_timestamp,
        end_opto_tagging_index,
        last_tone2_time,
        start_of_opto_tagging_index,
        behavioral_video_duration,
        final_behavioral_video_time,
        opto_closed_loop_start_timestamps,
        opto_tag_start_timestamps,
        last_closed_loop_start_time,
        first_opto_tagging_timestamp_per_trial,
        first_optical_pulse_per_closed_loop
    )

