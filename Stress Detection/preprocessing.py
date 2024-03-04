##Outdated check the Final Scripts folder

import biobss
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load the sample data
#data, info = biobss.utils.load_sample_data(data_type='PPG_SHORT')
csv_file_path = 'C:\Thesis-script\Stress_Detection\mydata\modified_15AX-aggregated_data.csv'
data = pd.read_csv(csv_file_path)
sig_bvp = np.asarray(data['.bvp'])
sig_eda = np.asarray(data['.gsr'])
#fs = info['sampling_rate']
fs = 64
#L = info['signal_length']
L_bvp = len(sig_bvp)

#plot
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=sig_bvp, plot_title='Original BVP Signal', signal_name='BVP', x_label='Samples')
plt.show()
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=sig_eda, plot_title='Original EDA Signal', signal_name='BVP', x_label='Samples')
plt.show()
#Filter PPG signal by defining the filter parameters  implement Butterworth filter by defining the filter parameters (filter type, filter order, cutoff frequencies)
f_sigbvp= biobss.preprocess.filter_signal(sig_bvp,sampling_rate=fs,filter_type='bandpass',N=2,f_lower=0.5,f_upper=5)
f_sigeda=biobss.edatools.filter_eda(sig_eda, fs, method='neurokit')


#plot
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=f_sigbvp, plot_title='Filtered BVP Signal', signal_name='BVP', x_label='Samples')
plt.show()
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=f_sigeda, plot_title='Filtered EDA Signal', signal_name='BVP', x_label='Samples')
plt.show()

#Normalize Signal
n_sigbvp= biobss.preprocess.normalize_signal(f_sigbvp)
n_sigeda= biobss.preprocess.normalize_signal(f_sigeda)

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

#EDA features
dec_eda = biobss.edatools.eda_decompose(n_sigeda, sampling_rate=fs, method='highpass')
eda_ton, eda_phasic = dec_eda['EDA_Tonic'], dec_eda['EDA_Phasic']

fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=eda_ton, plot_title='EDA Tonic', signal_name='BVP', x_label='Samples')
plt.show()
fig, ax = plt.subplots(figsize=(10, 4))
biobss.plottools.create_signal_plot_matplotlib(ax, signal=eda_phasic, plot_title='EDA Phasic', signal_name='BVP', x_label='Samples')
plt.show()

edapeaks=eda_detectpeaks(eda_phasic, sampling_rate=fs)

eda_feat=biobss.edatools.from_decomposed(signal_phasic=eda_phasic, signal_tonic=eda_ton, sampling_rate=fs)

eda_featsig=biobss.edatools.from_signal(signal=n_sigeda, sampling_rate=fs)

eda_freq=biobss.edatools.eda_freq_features(n_sigeda, prefix='eda')
eda_stat=biobss.edatools.eda_stat_features(n_sigeda, prefix='eda')

