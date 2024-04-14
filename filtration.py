import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi

# Similar to preprocessing but for real-time data

def standardize(data):
    data = data.astype(float)
    for i in range(data.shape[1]):
        mean = data[:, i].mean()
        std = data[:, i].std()
        data[:, i] = (data[:, i]- mean) / std
        print(data[:,i])
    return data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a) * data[0]
    y, _ = lfilter(b, a, data, zi=zi)
    return y

def butter_notch(notch_freq, fs, Q=30):
    nyq = 0.5 * fs
    notch = notch_freq / nyq
    b, a = butter(1, [notch - 1.0/Q, notch + 1.0/Q], btype='bandstop')
    return b, a

def butter_notch_filter(data, notch_freq, fs, Q=30):
    b, a = butter_notch(notch_freq, fs, Q)
    y = lfilter(b, a, data)
    return y

def butter_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a

def butter_highpass_filter(data, highcut, fs, order=5):
    b, a = butter_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_electrode_data(eeg_data, electrode_idx):
    num_samples = len(eeg_data)
    electrode_data = eeg_data[:, electrode_idx]
    labels = eeg_data[:, -1]  # Labels are in the last column

    plt.figure(figsize=(8, 4))
    plt.plot(range(num_samples), electrode_data, label=f'Electrode {electrode_idx+1}')
    plt.scatter(range(num_samples), electrode_data, c=labels, cmap='viridis', marker='o')
    plt.xlabel('Sample Index')
    plt.ylabel(f'Electrode {electrode_idx+1}')
    plt.title(f'EEG Data for Electrode {electrode_idx+1}')
    plt.legend()
    
    # Define a function for interactive zooming
    def on_xlims_change(axes):
        xlim = axes.get_xlim()
        #print(f"Current x-axis limits: {xlim}")
    
    plt.gca().callbacks.connect('xlim_changed', on_xlims_change)  # Connect the function to xlim_changed event
    
    plt.show()

# The same as in preprocessing.py
def filtrate(eeg_data):
    dataset_1 = eeg_data

    # Seperate labels and data
    data = dataset_1[:, :-1]
    labels = dataset_1[:, -1]
    dataset_1 = data

    processed_data = []  # feature after filtering

    # Add notch filter
    notch_freq = 50  # Hz
    dataset_1 = butter_notch_filter(dataset_1, notch_freq, 125)

    # Get wanted range of data
    for i in range(dataset_1.shape[1]):
        x = dataset_1[:, i]
        fs = 125
        lowcut = 5
        highcut = 50
        y = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
        #y = butter_highpass_filter(y, lowcut, fs, order=3)
        
        processed_data.append(y) 
        
    processed_data=np.array(processed_data).T

    processed_data = standardize(processed_data)

    filtered_data_with_labels = np.column_stack((processed_data, labels))

    filtered_data_with_labels = filtered_data_with_labels[100:]

    return filtered_data_with_labels