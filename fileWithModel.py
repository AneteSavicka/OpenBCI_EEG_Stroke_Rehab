# import packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# LSTM algorithm created by adapting algorithm published: https://github.com/xiangzhang1015/Deep-Learning-for-BCI/tree/38ecb0645cf861504a637a556fd4de74e106dabf

n_class = 2  
no_feature = 16 # The number of the electrodes
segment_length = 30  # Selected time window
n_hidden = 128  # Number of neurons in hidden layer
no_longfeature = no_feature*segment_length

def extract(input, n_classes, n_fea, time_window, moving):
    xx = input[:, :n_fea]
    yy = input[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        segment = xx[int(i * moving):int(i * moving + time_window), :]
        label = yy[int(i * moving):int(i * moving + time_window)]
        #print("Segment length:", segment.shape[0], "Label length:", label.shape[0])
        
        if segment.shape[0] == time_window and label.shape[0] == time_window:
            ave_y = int(np.round(np.average(label)))
            if ave_y in range(n_classes + 1):
                new_x.append(segment)
                new_y.append(ave_y)
            else:
                new_x.append(segment)
                new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    data = np.vstack((data, data[-1]))  # add the last sample again, to make the sample number round
    return data

# Load new data
new_data = np.load('laba/data_for_right_hand/1709304273_equalSize_filtered.npy')

plt.plot(new_data)
plt.show()

# Preprocess the new data
new_data_seg = extract(new_data, n_classes=n_class, n_fea=no_feature, time_window=segment_length, moving=(segment_length/2))
new_data_feature = new_data_seg[:, :no_longfeature]
new_data_seg_label = new_data_seg[:, no_longfeature:no_longfeature+1]
new_data_feature_2d = new_data_feature.reshape([-1, no_feature])

# Normalize the new data using the same scaler
scaler1 = StandardScaler().fit(new_data_feature_2d)
new_data_feature_norm = scaler1.transform(new_data_feature_2d)

#from sklearn.preprocessing import MinMaxScaler
#scaler1 = MinMaxScaler().fit(new_data_feature_2d)
#new_data_feature_norm = scaler1.transform(new_data_feature_2d)

# Reshape the normalized data for the LSTM model
new_data_feature_norm_3d = new_data_feature_norm.reshape([-1, segment_length, no_feature])

# Convert to PyTorch tensor
new_data_tensor = torch.tensor(new_data_feature_norm_3d).to(device)

# LSTM classifier
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm_layer = nn.LSTM(
            input_size=no_feature,
            hidden_size=n_hidden,         # LSTM hidden unit
            num_layers=2,           # number of LSTM layer
            bias=True,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
            #dropout=0.5  # Adjust the dropout rate as needed
        )

        self.out = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm_layer(x.float(), None)
        r_out = F.dropout(r_out, 0.3)

        test_output = self.out(r_out[:, -1, :]) # choose r_out at the last time step
        return test_output

# Set the model to evaluation mode and perform inference
lstm = LSTM()
model_weights_path = 'model/best_model_LSTM_laba1.pth' # Load trained model
lstm.load_state_dict(torch.load(model_weights_path))
lstm.eval()
with torch.no_grad():
    new_data_output = lstm(new_data_tensor)

# Interpret the output (e.g., calculate probabilities or make predictions)
new_data_probabilities = F.softmax(new_data_output, dim=1)
new_data_predictions = torch.argmax(new_data_output, dim=1).cpu().numpy()

total_elements = len(new_data_seg_label)
different_elements = sum(1 for a, b in zip(new_data_seg_label, new_data_predictions) if a != b)
accuracy = 1 - (different_elements / total_elements)

# Display the results
print(f"Total elements: {total_elements}")
print(f"Different elements: {different_elements}")
print(f"Accuracy: {accuracy * 100:.2f}%")

reality = np.array(list(new_data_seg_label), dtype=int)
predictions = np.array(list(new_data_predictions), dtype=int)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(range(len(reality)), reality, label='Reality', linestyle='-', marker='o', color='b')
ax.plot(range(len(predictions)), predictions, label='Predictions', linestyle='-', marker='x', color='r')

ax.set_title('Reality vs. Predictions')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.legend()

plt.show()