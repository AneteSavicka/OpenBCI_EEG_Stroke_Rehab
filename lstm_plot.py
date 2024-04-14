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

# Check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

# Load data
dataset_1 = np.load('laba/data_for_right_hand/1709300764_equalSize_filtered.npy')

# Extract the relevant channels
#channels_to_keep = [2,3,4,5,6,7,12,11,14,15,16] #[3,4,11,12,13,14,15,16,17] remove 1, 2, 9, 10, 11, 12 -> 0,1,8,9,10,11
#dataset_1 = dataset_1[:, channels_to_keep]

n_class = 2  
no_feature = 16 # The number of the electrodes
segment_length = 30  # Selected time window
LR = 0.005  # Learning rate
EPOCH = 201
n_hidden = 128  # Number of neurons in hidden layer
l2 = 0.001  # The coefficient of l2-norm regularization

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

def one_hot(y_):
    y_ = y_.reshape(len(y_))
    y_ = [int(xx) for xx in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


# Divide data in segments of 30 frames with 50% overlap
data_seg = extract(dataset_1, n_classes=n_class, n_fea=no_feature, time_window=segment_length, moving=(segment_length / 2))

# Split training and test data where test is 20%
no_longfeature = no_feature*segment_length
data_seg_feature = data_seg[:, :no_longfeature]
data_seg_label = data_seg[:, no_longfeature:no_longfeature+1]
train_feature, test_feature, train_label, test_label = train_test_split(data_seg_feature, data_seg_label,test_size=0.2, shuffle=True)

# Before normalize reshape data back to raw data shape
train_feature_2d = train_feature.reshape([-1, no_feature])
test_feature_2d = test_feature.reshape([-1, no_feature])

#from sklearn.preprocessing import MinMaxScaler
#scaler1 = MinMaxScaler().fit(train_feature_2d)
#train_fea_norm1 = scaler1.transform(train_feature_2d)
#test_fea_norm1 = scaler1.transform(test_feature_2d)

# Normalization
scaler1 = StandardScaler().fit(train_feature_2d)
train_fea_norm1 = scaler1.transform(train_feature_2d) # normalize the training data
test_fea_norm1 = scaler1.transform(test_feature_2d) # normalize the test data

# After normalization, reshape data to 3d in order to feed in to LSTM
train_fea_norm1 = train_fea_norm1.reshape([-1, segment_length, no_feature])
test_fea_norm1 = test_fea_norm1.reshape([-1, segment_length, no_feature])

BATCH_size = test_fea_norm1.shape[0]

# Feed data into dataloader
train_fea_norm1 = torch.tensor(train_fea_norm1).to(device)
train_label = torch.tensor(train_label.flatten()).to(device)
train_data = Data.TensorDataset(train_fea_norm1, train_label)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_size, shuffle=False)

test_fea_norm1 = torch.tensor(test_fea_norm1).to(device)
test_label = torch.tensor(test_label.flatten()).to(device)

############################
# Load second dataset for validation
dataset_2 = np.load('laba/data_for_right_hand/1709302461_equalSize_filtered.npy')

# Perform the same preprocessing
data_seg_2 = extract(dataset_2, n_classes=n_class, n_fea=no_feature, time_window=segment_length, moving=(segment_length)/2)

data_seg_feature_2 = data_seg_2[:, :no_longfeature]
data_seg_label_2 = data_seg_2[:, no_longfeature:no_longfeature+1]

data_seg_feature_2_2d = data_seg_feature_2.reshape([-1, no_feature])
data_seg_feature_2_norm = scaler1.transform(data_seg_feature_2_2d)  # using the same scaler as for training data
data_seg_feature_2_norm_3d = data_seg_feature_2_norm.reshape([-1, segment_length, no_feature])

validation_feature = torch.tensor(data_seg_feature_2_norm_3d).to(device)
validation_label = torch.tensor(data_seg_label_2.flatten()).to(device)
validation_data = Data.TensorDataset(validation_feature, validation_label)
validation_loader = Data.DataLoader(dataset=validation_data, batch_size=BATCH_size, shuffle=False)
#####################################################

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

        test_output = self.out(r_out[:, -1, :]) # hoose r_out at the last time step
        return test_output

lstm = LSTM()
lstm.to(device)
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr=LR, weight_decay=l2)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()

best_acc = []
best_auc = []
previous = 0

# Training and testing
start_time = time.perf_counter()
for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(train_loader):

        output = lstm(train_x)  # LSTM output of training data
        loss = loss_func(output, train_y.long())  # Cross entropy loss
        optimizer.zero_grad()  # Clear gradients for this training step
        loss.backward()  # Backpropagation, compute gradients
        optimizer.step()  # Apply gradients

        if epoch % 10 == 0 and step==2:
            test_output = lstm(test_fea_norm1)  # LSTM output of test data
            test_loss = loss_func(test_output, test_label.long())

            test_y_score = one_hot(test_label.data.cpu().numpy())
            pred_score = F.softmax(test_output, dim=1).data.cpu().numpy()  # Normalize the output
            auc_score = roc_auc_score(test_y_score, pred_score)

            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            pred_train = torch.max(output, 1)[1].data.cpu().numpy()

            test_acc = accuracy_score(test_label.data.cpu().numpy(), pred_y)
            train_acc = accuracy_score(train_y.data.cpu().numpy(), pred_train)


            print('Epoch: ', epoch, '|train loss: %.4f' % loss.item(),
                  ' train ACC: %.4f' % train_acc, '| test loss: %.4f' % test_loss.item(),
                  'test ACC: %.4f' % test_acc, '| AUC: %.4f' % auc_score)
            best_acc.append(test_acc)
            best_auc.append(auc_score)

            with torch.no_grad():
                # Validation with second file added
                validation_output = lstm(validation_feature)
                validation_loss = loss_func(validation_output, validation_label.long())

                validation_y_score = one_hot(validation_label.data.cpu().numpy())
                pred_validation_score = F.softmax(validation_output, dim=1).data.cpu().numpy()
                validation_auc_score = roc_auc_score(validation_y_score, pred_validation_score)

                pred_validation_y = torch.max(validation_output, 1)[1].data.cpu().numpy()
                validation_acc = accuracy_score(validation_label.data.cpu().numpy(), pred_validation_y)

                print('Epoch: ', epoch, '| Validation loss: %.4f' % validation_loss.item(),
                    'Validation ACC: %.4f' % validation_acc, '| Validation AUC: %.4f' % validation_auc_score)
                
                # Save the model with the best validation accuracy
                print(previous)
                print(validation_acc)
                if validation_acc > previous:
                    previous = validation_acc
                    model_weights_path = 'model/best_model_LSTM_laba1.pth'
                    torch.save(lstm.state_dict(), model_weights_path)
                    print("Best model saved at:", model_weights_path)

current_time = time.perf_counter()
running_time = current_time - start_time
print(classification_report(test_label.data.cpu().numpy(), pred_y))
print('BEST TEST ACC: {}, AUC: {}'.format(max(best_acc), max(best_auc)))
print("Total Running Time: {} seconds".format(round(running_time, 2)))

# Create confusion matrix
mapping = {'Rest':0,'Right':1}

cm = confusion_matrix(test_label, pred_y)
cir = classification_report(test_label, pred_y, target_names=mapping.keys())

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(2) + 0.5, label=mapping.keys())
plt.yticks(np.arange(2) + 0.5, label=mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
