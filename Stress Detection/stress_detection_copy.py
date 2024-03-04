##Outdated check the Final Scripts folder


import biobss
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Standardizing the features
# Dropping non-numeric columns and columns with missing values for simplicity
numeric_data = data.select_dtypes(include=[np.number]).dropna(axis=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Applying PCA
pca = PCA()
pca.fit(scaled_data)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') # for each component
plt.title('Explained Variance by PCA Components')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=12)
plt.show()


def deviation_above_mean(unit, mean_unit, std_unit):
        '''
        Function takes 3 arguments 
        unit : number of Standard deviations above the mean
        mean_unit : mean value of each signal
        std_unit : standard deviation of each signal
        
        '''
        if unit == 0:
            return (mean_unit)
        else:
            return (mean_unit + (unit*std_unit))

def feature_extract(file_path):
    data = pd.read_csv(file_path)
    sig_bvp = np.asarray(data['.bvp'])
    sig_eda = np.asarray(data['.gsr'])
    fs = 64  # Sample rate

    #Signal Preprosessing 
    #Filter PPG signal by defining the filter parameters  implement Butterworth filter by defining the filter parameters (filter type, filter order, cutoff frequencies)
    f_sigbvp= biobss.preprocess.filter_signal(sig_bvp,sampling_rate=fs,filter_type='bandpass',N=2,f_lower=0.5,f_upper=5)
    f_sigeda=biobss.edatools.filter_eda(sig_eda, fs, method='neurokit')

    #Normalize Signal
    n_sigbvp= biobss.preprocess.normalize_signal(f_sigbvp)
    n_sigeda= biobss.preprocess.normalize_signal(f_sigeda)

    #EDA features

    # Define window parameters
    window_size = 5  # in seconds
    step_size = 2.5  # in seconds

    # Segment the EDA signal
    eda_windows = biobss.preprocess.segment_signal(n_sigeda, fs, window_size, step_size)
    dec_eda = biobss.edatools.eda_decompose(n_sigeda, sampling_rate=fs, method='highpass')
    eda_ton, eda_phasic = dec_eda['EDA_Tonic'], dec_eda['EDA_Phasic']
    phasic_windows = biobss.preprocess.segment_signal(eda_phasic, fs, window_size, step_size)
    tonic_windows = biobss.preprocess.segment_signal(eda_ton, fs, window_size, step_size)
    #eda_featurespt =biobss.edatools.from_decomposed_windows(phasic_windows, tonic_windows, sampling_rate=fs, parallel=False, n_jobs=6)

    #eda_features =biobss.edatools.from_windows(eda_windows, sampling_rate=fs, parallel=False, n_jobs=6)
    eda_freq=biobss.edatools.eda_freq_features(eda_windows, prefix='eda')

    """ eda_features_df = from_edawindow(eda_windows, fs)
    eda_statfeat = []
    for w in eda_windows:
        eda_statfeat.append(biobss.edatools.eda_stat_features(w, prefix='eda'))
    eda_statfeat = pd.DataFrame(eda_statfeat)
    eda_freqfeat = []
    for w in eda_windows:
        eda_freqfeat.append(biobss.edatools.eda_freq_features(w, prefix='eda'))
    eda_freqfeat = pd.DataFrame(eda_freqfeat)
    eda_decomfeat = []
    for w in phasic_windows:
        eda_decomfeat.append(biobss.edatools.from_scr(w))
    eda_decomfeat = pd.DataFrame(eda_decomfeat)
    eda_decomtfeat = []
    for w in tonic_windows:
        eda_decomtfeat.append(biobss.edatools.from_scl(w))
    eda_decomtfeat = pd.DataFrame(eda_decomtfeat) """

    # Extract features from segmented windows
    def extract_features(windows, feature_function, prefix=None):
        if prefix:
            features = [feature_function(w, prefix=prefix) for w in windows]
        else:
            features = [feature_function(w) for w in windows]
        return pd.DataFrame(features)
    
    len(eda_windows)


    # Calculate different types of EDA features
    eda_stat_features = extract_features(eda_windows, biobss.edatools.eda_stat_features, 'eda')
    eda_freq_features = extract_features(eda_windows, biobss.edatools.eda_freq_features, 'eda')
    eda_decom_phasic_features = extract_features(phasic_windows, biobss.edatools.from_scr)
    eda_decom_tonic_features = extract_features(tonic_windows, biobss.edatools.from_scl)
    
    #### For HRV feaures
    """ info=biobss.ppgtools.ppg_detectpeaks(sig=n_sigbvp, sampling_rate=fs, method='peakdet', delta=0.01, correct_peaks=True)

    locs_peaks=info['Peak_locs']
    peaks=sig_bvp[locs_peaks]
    locs_onsets=info['Trough_locs']
    onsets=sig_bvp[locs_onsets]
    signals={'Raw': sig_bvp, 'Filtered': f_sigbvp}
    peaks={'Raw':{'Peaks': locs_peaks, 'Onsets': locs_onsets} , 'Filtered': {'Peaks': locs_peaks, 'Onsets':locs_onsets}}
    
    #Calculate HRV parameters using ppi intervals
    ppi = 1000*np.diff(locs_peaks)/fs 
    ppi_windows = biobss.preprocess.segment_signal(ppi, fs, window_size, step_size)
    len(ppi_windows)
    
    ppg_hrv = biobss.hrvtools.get_hrv_features(sampling_rate=fs, signal_length=L, input_type='ppi',ppi=ppi)
    #Calculate HRV parameters using peak locations
    ppg_hrv = biobss.hrvtools.get_hrv_features(sampling_rate=fs, signal_length=L, input_type='peaks',peaks_locs=locs_peaks)
    #Calculate time-domain HRV parameters 
    ppg_hrv_time = biobss.hrvtools.hrv_time_features(ppi=ppi_windows, sampling_rate=fs)
    #Calculate frequency-domain HRV parameters 
    ppg_hrv_freq = biobss.hrvtools.hrv_freq_features(ppi=ppi, sampling_rate=fs)
    #Calculate nonlinear HRV parameters 
    ppg_hrv_nl = biobss.hrvtools.hrv_nl_features(ppi=ppi, sampling_rate=fs) """
    
    ppg_windows = biobss.preprocess.segment_signal(n_sigbvp, fs, window_size, step_size)

    # Placeholder for HRV features for each window
    hrv_features_all_windows = []

    # Process each window to calculate HRV features
    for window in ppg_windows:
        # Detect peaks within this window (this is an example function call, replace with actual peak detection)
        info = biobss.ppgtools.ppg_detectpeaks(sig=window, sampling_rate=fs, method='peakdet', delta=0.01, correct_peaks=True)
        
        # Extract locations of peaks (and onsets) within the window
        locs_peaks = info['Peak_locs']
        # locs_onsets = info['Trough_locs']  # Uncomment if trough locations are also needed

        # Calculate interbeat intervals (IBIs) or peak-to-peak intervals in milliseconds
        ppi = 1000 * np.diff(locs_peaks) / fs
        
        # Ensure there are enough peaks to calculate HRV features
        if len(ppi) > 1:
            # Calculate HRV features for the window
            hrv_features = {
                'time': biobss.hrvtools.hrv_time_features(ppi, fs),
                #'freq': biobss.hrvtools.hrv_freq_features(ppi, fs),
                'nonlinear': biobss.hrvtools.hrv_nl_features(ppi, fs)
            }
            # Combine all HRV features into a single dictionary for the window
            hrv_features_combined = {**hrv_features['time'],  **hrv_features['nonlinear']}
            
            # Append the features of this window to the list
            hrv_features_all_windows.append(hrv_features_combined)

    # Convert the list of dictionaries to a DataFrame
    hrv_features_df = pd.DataFrame(hrv_features_all_windows)
    

    # Combine features
    part_data = data.iloc[:len(eda_stat_features), :2]
    e_combined_features = pd.concat([part_data, eda_stat_features, eda_freq_features, eda_decom_phasic_features, eda_decom_tonic_features], axis=1)
    all_combined_features = pd.concat([e_combined_features,hrv_features_df], axis=1)
    return all_combined_features

def process_participant_data(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    participant_results = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        processed_data = feature_extract(file_path)
        participant_results.append(processed_data)

    # Concatenate all results for the participant
    participant_data = pd.concat(participant_results, ignore_index=True)

    # Calculate participant-specific stress thresholds
    mean_scl = participant_data['scl_mean'].mean()
    std_scl = participant_data['scl_mean'].std()
    stress_threshold_scl = deviation_above_mean(1, mean_scl, std_scl)

    mean_scr = participant_data['scr_mean'].mean()
    std_scr = participant_data['scr_mean'].std()
    stress_threshold_scr = deviation_above_mean(1, mean_scr, std_scr)
    
    mean_hrv_rmssd = participant_data['hrv_rmssd'].mean()
    std_hrv_rmssd = participant_data['hrv_rmssd'].std()
    stress_threshold_hrv_rmssd = deviation_above_mean(1, mean_hrv_rmssd, std_hrv_rmssd)

    # Apply stress labeling
    participant_data['stress_label_scl'] = participant_data['scl_mean'].apply(lambda x: 1 if x > stress_threshold_scl else 0)
    participant_data['stress_label_scr'] = participant_data['scr_mean'].apply(lambda x: 1 if x > stress_threshold_scr else 0)
    participant_data['stress_label_hrv_rmssd'] = participant_data['hrv_rmssd'].apply(lambda x: 1 if x > stress_threshold_hrv_rmssd else 0)

    return participant_data

# Process data for each participant
all_participant_data = []

for participant_id in range(5, 20):  # Folders 5 to 19
    folder_path = f'C:\Thesis-script\\biobss\data\\{participant_id}'
    participant_data = process_participant_data(folder_path)
    all_participant_data.append(participant_data)

    # Optionally save each participant's data to a CSV file
    participant_data.to_csv(f'C:\Thesis-script\\biobss\data\\{participant_id}\\final_combined_data.csv', index=False)

# Concatenate data from all participants
final_data = pd.concat(all_participant_data, ignore_index=True)

# Save the combined data for all participants
final_data.to_csv('C:\\Thesis-script\\Data\\final_combined_data_all_participants.csv', index=False)

#Adding TLX Questionnaire data and labelling anythong above 1 z score as stress
TLX = pd.read_csv("C:/Thesis-script/Data/Nasa-Tlx-zscores.csv")
TLX.rename(columns={'ParticipantID': 'Participant ID'}, inplace=True)
count_weighted_rating_gt_50 = (TLX['WeightedRating'] > 50).sum()

count_z_scores_gt_1 = (TLX['Z_Score'] > 1).sum()
count_z_scores_gt_2=((TLX['Z_Score'] < -1) | (TLX['Z_Score'] > 1)).sum()

merged_data = final_data.merge(TLX[['Participant ID', 'Experiment', 'WeightedRating', 'Z_Score']], on=['Participant ID', 'Experiment'],how='left')
merged_data.rename(columns={'WeightedRating': 'tlx_weighted_rating', 'Z_Score': 'tlx_z_score'}, inplace=True)
merged_data['stress_label_tlx'] = merged_data['tlx_z_score'].apply(lambda x: 1 if x > 1 else 0)

# Concatenate the stress labels to form a unique identifier for each combination
merged_data['stress_label_combination'] = (merged_data['stress_label_scl'].astype(str) + merged_data['stress_label_scr'].astype(str) + merged_data['stress_label_hrv_rmssd'].astype(str) + merged_data['stress_label_tlx'].astype(str))
label_combinations_counts = merged_data['stress_label_combination'].value_counts()


# Bar plot of label combinations
label_combinations_counts.plot(kind='bar')
plt.xlabel('Stress Label Combinations (SCL_SCR_TLX)')
plt.ylabel('Count')
plt.title('Distribution of Stress Label Combinations')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()

def final_stress_label(row):
    # Concatenate the stress labels into a single string
    label_combination = f"{row['stress_label_scl']}{row['stress_label_scr']}{row['stress_label_hrv_rmssd']}{row['stress_label_tlx']}"
    
    # Check the combination and assign the final stress label
    if label_combination in ['0000', '0001']:
        return 0
    else:
        return 1

# Apply the function to each row in the DataFrame to create the final stress label
merged_data['final_stress_label'] = merged_data.apply(final_stress_label, axis=1)

merged_data.to_pickle('C:/Thesis-script/Data/biodata.pkl')
merged_data.to_csv('C:/Thesis-script/Data/biodata.csv', index=False)


#################################################


import os

# Deleting 'final_combined_data.csv' file from each participant's folder
for participant_id in range(5, 20):  # Folders 5 to 19
    folder_path = f'C:\\Thesis-script\\biobss\\data\\{participant_id}'
    file_path = os.path.join(folder_path, 'final_combined_data.csv')

    # Check if the file exists before attempting to delete
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted 'final_combined_data.csv' from {folder_path}")
    else:
        print(f"No file found: 'final_combined_data.csv' in {folder_path}")


############################################################################

# Concatenate the stress labels to form a unique identifier for each combination
merged_data['stress_label_combination'] = (merged_data['stress_label_scl'].astype(str) + merged_data['stress_label_scr'].astype(str) + merged_data['stress_label_hrv_rmssd'].astype(str) + merged_data['stress_label_tlx'].astype(str))
label_combinations_counts = merged_data['stress_label_combination'].value_counts()


# Bar plot of label combinations
label_combinations_counts.plot(kind='bar')
plt.xlabel('Stress Label Combinations (SCL_SCR_HRV_RMSSD_TLX)')
plt.ylabel('Count')
plt.title('Distribution of Stress Label Combinations')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()
















TLX = pd.read_csv("C:/Thesis-script/Data/Nasa-Tlx-zscores.csv")
TLX.rename(columns={'ParticipantID': 'Participant ID'}, inplace=True)
count_weighted_rating_gt_50 = (TLX['WeightedRating'] > 50).sum()

count_z_scores_gt_1 = (TLX['Z_Score'] > 1).sum()
count_z_scores_gt_2=((TLX['Z_Score'] < -1) | (TLX['Z_Score'] > 1)).sum()

merged_data = final_data.merge(TLX[['Participant ID', 'Experiment', 'WeightedRating', 'Z_Score']], on=['Participant ID', 'Experiment'],how='left')
merged_data.rename(columns={'WeightedRating': 'tlx_weighted_rating', 'Z_Score': 'tlx_z_score'}, inplace=True)




# Concatenate the stress labels to form a unique identifier for each combination
merged_data['stress_label_combination'] = (merged_data['stress_label_scl'].astype(str) + merged_data['stress_label_scr'].astype(str) + merged_data['stress_label_hrv_rmssd'].astype(str) + merged_data['stress_label_tlx'].astype(str))
label_combinations_counts = merged_data['stress_label_combination'].value_counts()


# Bar plot of label combinations
label_combinations_counts.plot(kind='bar')
plt.xlabel('Stress Label Combinations (SCL_SCR_TLX)')
plt.ylabel('Count')
plt.title('Distribution of Stress Label Combinations')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()

def final_stress_label(row):
    # Concatenate the stress labels into a single string
    label_combination = f"{row['stress_label_scl']}{row['stress_label_scr']}{row['stress_label_hrv_rmssd']}{row['stress_label_tlx']}"
    
    # Check the combination and assign the final stress label
    if label_combination in ['0000', '0001']:
        return 0
    else:
        return 1

# Apply the function to each row in the DataFrame to create the final stress label
merged_data['final_stress_label'] = merged_data.apply(final_stress_label, axis=1)



######################################################################################


#Load the sample data ( the whole participants to be done later)
#data, info = biobss.utils.load_sample_data(data_type='PPG_SHORT')
csv_file_path = 'C:\Thesis-script\Stress_Detection\mydata\modified_15AX-aggregated_data.csv'
data = pd.read_csv(csv_file_path)
sig_bvp = np.asarray(data['.bvp'])
sig_eda = np.asarray(data['.gsr'])
#fs = info['sampling_rate']
fs = 64
#L = info['signal_length']
L_bvp = len(sig_bvp)

#Signal Preprosessing 
#Filter PPG signal by defining the filter parameters  implement Butterworth filter by defining the filter parameters (filter type, filter order, cutoff frequencies)
f_sigbvp= biobss.preprocess.filter_signal(sig_bvp,sampling_rate=fs,filter_type='bandpass',N=2,f_lower=0.5,f_upper=5)
f_sigeda=biobss.edatools.filter_eda(sig_eda, fs, method='neurokit')

#Normalize Signal
n_sigbvp= biobss.preprocess.normalize_signal(f_sigbvp)
n_sigeda= biobss.preprocess.normalize_signal(f_sigeda)

#EDA features

# Define window parameters
window_size = 5  # in seconds
step_size = 2.5  # in seconds

# Segment the EDA signal
eda_windows = biobss.preprocess.segment_signal(n_sigeda, fs, window_size, step_size)
dec_eda = biobss.edatools.eda_decompose(n_sigeda, sampling_rate=fs, method='highpass')
eda_ton, eda_phasic = dec_eda['EDA_Tonic'], dec_eda['EDA_Phasic']
phasic_windows = biobss.preprocess.segment_signal(eda_phasic, fs, window_size, step_size)
tonic_windows = biobss.preprocess.segment_signal(eda_ton, fs, window_size, step_size)
#eda_featurespt =biobss.edatools.from_decomposed_windows(phasic_windows, tonic_windows, sampling_rate=fs, parallel=False, n_jobs=6)

#eda_features =biobss.edatools.from_windows(eda_windows, sampling_rate=fs, parallel=False, n_jobs=6)
eda_freq=biobss.edatools.eda_freq_features(eda_windows, prefix='eda')

""" eda_features_df = from_edawindow(eda_windows, fs)
eda_statfeat = []
for w in eda_windows:
    eda_statfeat.append(biobss.edatools.eda_stat_features(w, prefix='eda'))
eda_statfeat = pd.DataFrame(eda_statfeat)
eda_freqfeat = []
for w in eda_windows:
    eda_freqfeat.append(biobss.edatools.eda_freq_features(w, prefix='eda'))
eda_freqfeat = pd.DataFrame(eda_freqfeat)
eda_decomfeat = []
for w in phasic_windows:
    eda_decomfeat.append(biobss.edatools.from_scr(w))
eda_decomfeat = pd.DataFrame(eda_decomfeat)
eda_decomtfeat = []
for w in tonic_windows:
    eda_decomtfeat.append(biobss.edatools.from_scl(w))
eda_decomtfeat = pd.DataFrame(eda_decomtfeat) """

# Extract features from segmented windows
def extract_features(windows, feature_function, prefix=None):
    if prefix:
        features = [feature_function(w, prefix=prefix) for w in windows]
    else:
        features = [feature_function(w) for w in windows]
    return pd.DataFrame(features)

# Calculate different types of EDA features
eda_stat_features = extract_features(eda_windows, biobss.edatools.eda_stat_features, 'eda')
eda_freq_features = extract_features(eda_windows, biobss.edatools.eda_freq_features, 'eda')
eda_decom_phasic_features = extract_features(phasic_windows, biobss.edatools.from_scr)
eda_decom_tonic_features = extract_features(tonic_windows, biobss.edatools.from_scl)

# Combine all features into one DataFrame
part_data = data.iloc[:len(eda_stat_features), :2] # retaining only the participant ID and the experiment
eda_combined_features = pd.concat([part_data,eda_stat_features, eda_freq_features, eda_decom_phasic_features, eda_decom_tonic_features], axis=1)

participant_experiment_data = data.iloc[:, :2]





""" for segment in segments:
    dec_eda = biobss.edatools.eda_decompose(n_sigeda, sampling_rate=fs, method='highpass')
    eda_ton, eda_phasic = dec_eda['EDA_Tonic'], dec_eda['EDA_Phasic']

    fig, ax = plt.subplots(figsize=(10, 4))
    biobss.plottools.create_signal_plot_matplotlib(ax, signal=eda_ton, plot_title='EDA Tonic', signal_name='BVP', x_label='Samples')
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 4))
    biobss.plottools.create_signal_plot_matplotlib(ax, signal=eda_phasic, plot_title='EDA Phasic', signal_name='BVP', x_label='Samples')
    plt.show()

    #edapeaks=eda_detectpeaks(eda_phasic, sampling_rate=fs)

    eda_feat=biobss.edatools.from_decomposed(signal_phasic=eda_phasic, signal_tonic=eda_ton, sampling_rate=fs)

    eda_featsig=biobss.edatools.from_signal(signal=n_sigeda, sampling_rate=fs)

    eda_freq=biobss.edatools.eda_freq_features(n_sigeda, prefix='eda')
    eda_stat=biobss.edatools.eda_stat_features(n_sigeda, prefix='eda') """









#plot
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=sig_bvp, plot_title='Original BVP Signal', signal_name='BVP', x_label='Samples')
plt.show()
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=sig_eda, plot_title='Original EDA Signal', signal_name='BVP', x_label='Samples')
plt.show()



#plot
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=f_sigbvp, plot_title='Filtered BVP Signal', signal_name='BVP', x_label='Samples')
plt.show()
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=f_sigeda, plot_title='Filtered EDA Signal', signal_name='BVP', x_label='Samples')
plt.show()



#plot
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=n_sigbvp, plot_title='Filtered BVP Signal', signal_name='BVP', x_label='Samples')
plt.show()
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=n_sigeda, plot_title='Normalized EDA Signal', signal_name='BVP', x_label='Samples')
plt.show()


#Detect peaks using 'peakdet' method (delta=0.01). Delta parameter should be adjusted related to the amplitude of the signal.

info=biobss.ppgtools.ppg_detectpeaks(sig=n_sigbvp, sampling_rate=fs, method='peakdet', delta=0.01, correct_peaks=True)

locs_peaks=info['Peak_locs']
peaks=sig_bvp[locs_peaks]
locs_onsets=info['Trough_locs']
onsets=sig_bvp[locs_onsets]
signals={'Raw': sig_bvp, 'Filtered': f_sigbvp}
peaks={'Raw':{'Peaks': locs_peaks, 'Onsets': locs_onsets} , 'Filtered': {'Peaks': locs_peaks, 'Onsets':locs_onsets}}
biobss.ppgtools.plot_ppg(signals=signals, peaks=peaks, sampling_rate=fs, show_peaks=True, rescale=True, figsize=(10,5))

""" locs_peaks=biobss.ppgtools.ppg_detectbeats(sig, sampling_rate=fs, method='peakdet', delta=0.005)
peaks=sig[locs_peaks]
locs_onsets=info['Trough_locs']
onsets=sig[locs_onsets]

#info=biobss.ppgtools.ppg_detectpeaks(sig=n_sigbvp, sampling_rate=fs, method='peakdet', delta=0.005, type='beat', correct_peaks=True)

locs_peaks=info['Peak_locs']
peaks=sig[locs_peaks]
locs_onsets=info['Trough_locs']
onsets=sig[locs_onsets] """
#PPG signal waveform includes two peaks which are systolic and diastolic peaks however the diastolic peak may not be observable in some conditions. From the diastolic peak locations, some extra features may be calculated.
#In order to detect the location of diastolic peak, generally the first and second derivatives of PPG signal are needed. First, fiducial points on the first derivative (Velocity Plethysmogram, VPG) and the second derivative (Acceleration Plethysmogram, APG) should be detected. Fiducials can also be used to calculate VPG and APG features which may be helpful in some analysis, e.g. blood pressure estimation from PPG signal.
#Calculate first and second derivatives of the PPG signal
vpg_sig = np.gradient(n_sigbvp) / (1/fs)
apg_sig = np.gradient(vpg_sig) / (1/fs)

vpg_fiducials = biobss.ppgtools.vpg_delineate(vpg_sig, sampling_rate=fs)

apg_fiducials = biobss.ppgtools.apg_delineate(apg_sig, vpg_sig, vpg_fiducials, sampling_rate=fs)

ppg_fiducials = biobss.ppgtools.ppg_delineate(n_sigbvp, vpg_sig, vpg_fiducials, apg_sig, apg_fiducials, sampling_rate=fs)

ppg_waves=biobss.ppgtools.ppg_waves(sig=n_sigbvp, locs_onsets= locs_onsets, sampling_rate=fs)

fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=apg_sig, plot_title='VPG signal', signal_name='BVP', x_label='Samples')
plt.show()


#Plotting
#Generate inputs as dictionaries
signals={'Raw': sig_bvp, 'Filtered': f_sigbvp}
peaks={'Raw':{'Peaks': locs_peaks, 'Onsets': locs_onsets} , 'Filtered': {'Peaks': locs_peaks, 'Onsets':locs_onsets}}

#Plot PPG Signal using Matplotlib
biobss.ppgtools.plot_ppg(signals=signals, peaks=peaks, sampling_rate=fs, show_peaks=True, rescale=True, figsize=(10,5))

#Plot PPG signal using Plotly
biobss.ppgtools.plot_ppg(signals=signals, peaks=peaks, sampling_rate=fs, method='plotly', show_peaks=True, rescale=True)


#Check for physiological and morphological limits 
info_phys=biobss.sqatools.check_phys(locs_peaks,fs)
info_morph=biobss.sqatools.check_morph(sig,locs_peaks,locs_onsets,fs)

#PPG Feature extraction
#Calculate cycle-based time-domain features
""" a_S: Mean amplitude of the systolic peaks
t_S: Mean systolic peak duration
t_C: Mean cycle duration
DW: Mean diastolic peak duration
SW_10: The systolic peak duration at 10% amplitude of systolic amplitude
SW_25: The systolic peak duration at 25% amplitude of systolic amplitude
SW_33: The systolic peak duration at 33% amplitude of systolic amplitude
SW_50: The systolic peak duration at 50% amplitude of systolic amplitude
SW_66: The systolic peak duration at 66% amplitude of systolic amplitude
SW_75: The systolic peak duration at 75% amplitude of systolic amplitude
DW_10: The diastolic peak duration at 10% amplitude of systolic amplitude
DW_25: The diastolic peak duration at 25% amplitude of systolic amplitude
DW_33: The diastolic peak duration at 33% amplitude of systolic amplitude
DW_50: The diastolic peak duration at 50% amplitude of systolic amplitude
DW_66: The diastolic peak duration at 66% amplitude of systolic amplitude
DW_75: The diastolic peak duration at 75% amplitude of systolic amplitude
DW_SW_10: The ratio of diastolic peak duration to systolic peak duration at 10% amplitude of systolic amplitude
DW_SW_25: The ratio of diastolic peak duration to systolic peak duration at 25% amplitude of systolic amplitude
DW_SW_33: The ratio of diastolic peak duration to systolic peak duration at 33% amplitude of systolic amplitude
DW_SW_50: The ratio of diastolic peak duration to systolic peak duration at 50% amplitude of systolic amplitude
DW_SW_66: The ratio of diastolic peak duration to systolic peak duration at 66% amplitude of systolic amplitude
DW_SW_75: The ratio of diastolic peak duration to systolic peak duration at 75% amplitude of systolic amplitude
PR_mean: Mean pulse rate
a_D: Mean amplitude of the diastolic peaks
t_D: Mean difference between diastolic peak and onset
r_D: Mean ratio of the diastolic peak amplitude to diastolic peak duration
a_N: Mean amplitude of the dicrotic notchs
t_N: Mean dicrotic notch duration
r_N: Mean ratio of the dicrotic notch amplitude to dicrotic notch duration
dT: Mean duration from systolic to diastolic peaks
r_D_NC: Mean ratio of diastolic peak amplitudes to difference between ppg wave duration and dictoric notch duration
r_N_NC: Mean ratio of dicrotic notch amplitudes to difference between ppg wave duration and dictoric notch duration
a_N_S: Mean ratio of dicrotic notch amplitudes to systolic peak amplitudes
AI: Mean ratio of diastolic peak amplitudes to systolic peak amplitudes
AI_2: Mean ratio of difference between systolic and diastolic peak amplitudes to systolic peak amplitudes """
ppg_time = biobss.ppgtools.ppg_features.ppg_time_features(n_sigbvp, sampling_rate=fs, input_types=['cycle','segment'], peaks_locs=locs_peaks, troughs_locs=locs_onsets)
ppg_time
#ppg_time = biobss.ppgtools.ppg_features.ppg_time_features(n_sigbvp, sampling_rate=fs, input_types=['cycle','segment'], fiducials=ppg_fiducials) 
#Calculate frequency-domain features
ppg_freq = biobss.ppgtools.ppg_features.ppg_freq_features(sig_bvp, sampling_rate=fs, input_types=['segment'])
ppg_freq
#Calculate cycle-based statistical features
""" mean_peaks: Mean of the peak amplitudes
std_peaks: Standard deviation of the peak amplitudes
Segment-based features:

mean: Mean value of the signal
median: Median value of the signal
std: Standard deviation of the signal
pct_25: 25th percentile of the signal
pct_75 75th percentile of the signal
mad: Mean absolute deviation of the signal
skewness: Skewness of the signal
kurtosis: Kurtosis of the signal
entropy: Entropy of the signal """
ppg_stat = biobss.ppgtools.ppg_features.ppg_stat_features(sig_bvp, sampling_rate=fs, input_types=['segment'], peaks_amp=peaks, peaks_locs=locs_peaks, troughs_locs=locs_onsets)
ppg_stat
features_all = biobss.ppgtools.get_ppg_features(sig_bvp, sampling_rate=fs, input_types=['cycle','segment'], feature_domain={'cycle':['Time'],'segment':['time','freq','stat']}, peaks_locs=locs_peaks, peaks_amp=peaks, troughs_locs=locs_onsets, troughs_amp=onsets)
features_all
#features_vpg = biobss.ppgtools.get_vpg_features(vpg_sig=vpg_sig, locs_O=locs_onsets, fiducials=vpg_fiducials, sampling_rate=fs)

#features_apg = biobss.ppgtools.get_apg_features(apg_sig=vpg_sig, locs_O=locs_onsets, fiducials=apg_fiducials, sampling_rate=fs)

#Calculate HRV parameters using ppi intervals
ppi = 1000*np.diff(locs_peaks)/fs 
ppg_hrv = biobss.hrvtools.get_hrv_features(sampling_rate=fs, signal_length=L, input_type='ppi',ppi=ppi)
#Calculate HRV parameters using peak locations
ppg_hrv = biobss.hrvtools.get_hrv_features(sampling_rate=fs, signal_length=L, input_type='peaks',peaks_locs=locs_peaks)
#Calculate time-domain HRV parameters 
ppg_hrv_time = biobss.hrvtools.hrv_time_features(ppi=ppi, sampling_rate=fs)
#Calculate frequency-domain HRV parameters 
ppg_hrv_freq = biobss.hrvtools.hrv_freq_features(ppi=ppi, sampling_rate=fs)
#Calculate nonlinear HRV parameters 
ppg_hrv_nl = biobss.hrvtools.hrv_nl_features(ppi=ppi, sampling_rate=fs)


fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=n_sigeda, plot_title='EDA Tonic', signal_name='BVP', x_label='Samples')
plt.show()


