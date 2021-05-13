##IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import sklearn as sk
import time
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History

##READING IN THE DATASET AND FORMATTING IT FOR DATA PREPROCESSING
#Edit to ensure that the current path is shown
dataset = pd.read_csv('C:\\Users\\abiga\\Documents\\University Of Lincoln\\Third Year\\Project\\Phishing Classification\\dataset_small.csv').values
#dataset = pd.read_csv('dataset_small.csv').values
data = dataset
data = np.delete(data, 111, 1)
data = np.array(data)
#print(data)

#Initializing the lists that will be used for preprocessing
means, medians, mins, maxs, variances, stds = [], [], [], [], [], []
phishing, legitimate = [], []

#Calculating the summary statistics for each column
for i in range(0,8):
    means.append(np.mean(data[:,i]))
    medians.append(np.median(data[:,i]))
    mins.append(np.min(data[:,i]))
    maxs.append(np.max(data[:,i]))
    variances.append(np.var(data[:,i]))
    stds.append(np.std(data[:,i]))
#Printing each of the statistic arrays
#print ("Means: ", means, "\nMedians: ", medians, "\nMins: ", mins, "\nMaxs: ", maxs, "\nVariances: ", variances, "\nStandard Deviations: ", stds)

#Getting the size of the dataset
length = len(dataset)
#Finding the number of features in the dataset
features = len(dataset[1,:])

#Splitting the dataset into legitimate and phishing
for i in range(len(dataset)):
    if dataset[i,9] == 0:
        legitimate.append(dataset[i,:])
    elif dataset[i,9] == 1:
        phishing.append(dataset[i,:])

#Turning the legitimate and phishing arrays into lists
legitimate = np.array(legitimate).tolist()
phishing = np.array(phishing).tolist()

##### SPLITTING THE DATA #####
#Using the library to split the data into 70% training and 30% testing
training, testing = train_test_split(dataset, test_size=0.3)
x_train, y_train, x_test, y_test = [], [], [], []
#Using the split data to split it futher into x and y - the X holds the feature data and the Y holds the status column
x_train, x_test = training, testing
x_train, x_test = np.delete(x_train, 111, 1), np.delete(x_test, 111 , 1)
y_train, y_test = training[:,111],testing[:,111]
#Using the sklearn library to scale and normalise the feature data
scaler = MinMaxScaler()
x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)
#Making sure all the arrays are in a format that both models can understand
x_train, x_test, y_train, y_test = np.asarray(x_train).astype(np.float32), np.asarray(x_test).astype(np.float32), np.asarray(y_train).astype(np.float32),np.asarray(y_test).astype(np.float32)

##CLASS USED TO TIME EACH OF THE EPOCH IN THE NEURAL NETWORK
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

##FUNCTION USED TO CALCULATE THE AVERAGE OF THE EPOCHS
def average(array):
    sum = 0
    for i in range(len(array)):
        sum += array[i]
    average = sum/len(array)
    return average

##### PERFORMANCE METRICS #####
#Using the sklearn to calculate the accuracy
#Status = actual values (y_test) and pred = the models prediction

##FUNCTION USED TO CALCULATE THE ACCURACY OF THE PREDICTED ARRAYS
def accuracy(status, pred): #how often is the classifier correct
    x = 0
    for i in range(len(status)):
        if status[i] == pred[i]:
            x += 1
    accuracy = x/len(status) * 100
    return accuracy

def error_rate(status, pred): #how often is the classifier wrong
    x = 0
    for i in range(len(status)):
        if status[i] == pred[i]:
            x += 1
    error = len(status) - x
    error_rate = error/len(status) * 100
    return error_rate

def sensitivity(status, pred): #when yes, how often does it predict yes
    TP, FN = 0 ,0
    for i in range(len(status)):
        if status[i] == 1:
            if pred[i] == 1:
                TP += 1
            else:
                FN += 1
    sensitivity = TP / (TP+FN)
    return sensitivity

def specificity(status, pred): #when no, how often does it predict no
    TN, FP = 0, 0
    for i in range(len(status)):
        if status[i] == 0:
            if pred[i] == 0:
                TN += 1
            else:
                FP += 1
    specificity = TN / (TN+FP)
    return specificity

##FUNCTION TO CALL THE ABOVE FUNCTIONS
def performance(status, pred):
    accuracy1 = accuracy(status, pred)
    error_rate1 = error_rate(status, pred)
    sensitivity1 = sensitivity(status, pred)
    specificity1 = specificity(status, pred)
    return accuracy1, error_rate1, sensitivity1, specificity1

##write a function for converting the y_pred into 1s and 0s to be passed into my performance metrics
def convert(pred):
    for i in range(len(pred)):
        if pred[i] >= 0.5:
            pred[i] =1 
        elif pred[i] < 0.5:
            pred[i] = 0
    return pred

########## MACHINE LEARNING STUFF #########

##### RANDOM TREE CLASSIFIER STUFF #####
#Using the function to train the RFC against the training data with 100 trees and node size of 5
RF_5 = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, verbose=1)
RF_5.fit(x_train, y_train)
#Using the predict function to apply the testing data to the trained model
RF_5_pred = RF_5.predict(x_test)
#Using the accuracy score function to find the accuracy of the trained model against the actual values for the test set
# RF_5_accuracy = accuracy_score(y_test ,RF_5_pred)
[RF_5_accuracy, RF_5_error, RF_5_sensitivity, RF_5_specificity] = performance(y_test, RF_5_pred)

# #Using the function to train the RFC against the training data with 1000 trees and node size of 5
# RF_5_2 = RandomForestClassifier(n_estimators=1000, min_samples_leaf=5, verbose=1)
# RF_5_2.fit(x_train, y_train)
# #Using the predict function to apply the testing data to the trained model
# RF_5_2_pred = RF_5_2.predict(x_test)
# #Using the accuracy score function to find the accuracy of the trained model against the actual values for the test set
# # RF_5_accuracy = accuracy_score(y_test ,RF_5_pred)
# [RF_5_2_accuracy, RF_5_2_error, RF_5_2_sensitivity, RF_5_2_specificity] = performance(y_test, RF_5_2_pred)

# #Using the function to train the RFC against the training data with 100 trees and node size of 50
# RF_50 = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, verbose=1)
# RF_50.fit(x_train, y_train)
# #Using the predict function to apply the testing data to the trained model
# RF_50_pred = RF_50.predict(x_test)
# #Using the accuracy score function to find the accuracy of the trained model against the actual values for the test set
# # RF_5_accuracy = accuracy_score(y_test ,RF_5_pred)
# [RF_50_accuracy, RF_50_error, RF_50_sensitivity, RF_50_specificity] = performance(y_test, RF_50_pred)

# #Using the function to train the RFC against the training data with 1000 trees and node size of 50
# RF_50_2 = RandomForestClassifier(n_estimators=1000, min_samples_leaf=50, verbose=1)
# RF_50_2.fit(x_train, y_train)
# #Using the predict function to apply the testing data to the trained model
# RF_50_2_pred = RF_50_2.predict(x_test)
# #Using the accuracy score function to find the accuracy of the trained model against the actual values for the test set
# # RF_5_accuracy = accuracy_score(y_test ,RF_5_pred)
# [RF_50_2_accuracy, RF_50_2_error, RF_50_2_sensitivity, RF_50_2_specificity] = performance(y_test, RF_50_2_pred)

# ##### LOGISTIC REGRESSION STUFF #####

#Using the libraries function to initialise the LR algorithm with the wanted paramaters
LR_100 = LogisticRegression(max_iter=100, random_state=0, solver='lbfgs', multi_class='ovr', verbose=1)
LR_100.fit(x_train, y_train)
#Using the predict function to apply the testing data to the trained model
LR_100_pred = LR_100.predict(x_test)
# LR_accuracy = accuracy_score(y_test ,LR_pred)
[LR_100_accuracy, LR_100_error, LR_100_sensitivity, LR_100_specificity] = performance(y_test, LR_100_pred)

# LR_300 = LogisticRegression(max_iter=300, random_state=0, solver='lbfgs', multi_class='ovr', verbose=1)
# LR_300.fit(x_train, y_train)
# LR_300_pred = LR_300.predict(x_test)
# # LR_accuracy = accuracy_score(y_test ,LR_pred)
# [LR_300_accuracy, LR_300_error, LR_300_sensitivity, LR_300_specificity] = performance(y_test, LR_300_pred)

# LR_500 = LogisticRegression(max_iter=500, random_state=0, solver='lbfgs', multi_class='ovr', verbose=1)
# LR_500.fit(x_train, y_train)
# LR_500_pred = LR_500.predict(x_test)
# # LR_accuracy = accuracy_score(y_test ,LR_pred)
# [LR_500_accuracy, LR_500_error, LR_500_sensitivity, LR_500_specificity] = performance(y_test, LR_500_pred)

# LR_1000 = LogisticRegression(max_iter=1000, random_state=0, solver='lbfgs', multi_class='ovr', verbose=1)
# LR_1000.fit(x_train, y_train)
# LR_1000_pred = LR_1000.predict(x_test)
# # LR_accuracy = accuracy_score(y_test ,LR_pred)
# [LR_1000_accuracy, LR_1000_error, LR_1000_sensitivity, LR_1000_specificity] = performance(y_test, LR_1000_pred)

# ##### SUPPORT MACHINE VECTOR STUFF #####
#Using the library's function to initialise the algorithm
SVM_100 = svm.LinearSVC(max_iter=100, verbose=1)
SVM_100.fit(x_train, y_train)
#Using the predict function to apply the testing data to the trained model
SVM_100_pred = SVM_100.predict(x_test)
#SVM_100_accuracy = accuracy_score(y_test ,SVM_100_pred)
[SVM_100_accuracy, SVM_100_error, SVM_100_sensitivity, SVM_100_specificity] = performance(y_test, SVM_100_pred)

# SVM_300 = svm.LinearSVC(max_iter=300,verbose=1)
# SVM_300.fit(x_train, y_train)
# SVM_300_pred = SVM_300.predict(x_test)
# #SVM_100_accuracy = accuracy_score(y_test ,SVM_100_pred)
# [SVM_300_accuracy, SVM_300_error, SVM_300_sensitivity, SVM_300_specificity] = performance(y_test, SVM_300_pred)

# SVM_500 = svm.LinearSVC(max_iter=500,verbose=1)
# SVM_500.fit(x_train, y_train)
# SVM_500_pred = SVM_500.predict(x_test)
# #SVM_100_accuracy = accuracy_score(y_test ,SVM_100_pred)
# [SVM_500_accuracy, SVM_500_error, SVM_500_sensitivity, SVM_500_specificity] = performance(y_test, SVM_500_pred)

# SVM_1000 = svm.LinearSVC(max_iter=1000,verbose=1)
# SVM_1000.fit(x_train, y_train)
# SVM_1000_pred = SVM_1000.predict(x_test)
# #SVM_100_accuracy = accuracy_score(y_test ,SVM_100_pred)
# [SVM_1000_accuracy, SVM_1000_error, SVM_1000_sensitivity, SVM_1000_specificity] = performance(y_test, SVM_1000_pred)

##### ARTIFICIAL NEURAL NETWORK STUFF #####

# Initialize the constructor
model = Sequential()
# Input Layer and first hidden layer
model.add(Dense(500, activation='relu', input_shape=(111,)))
# Add one hidden layer
model.add(Dense(500, activation='relu'))
# Add an output layer
model.add(Dense(1, activation='sigmoid'))
# #Setting the number of epochs
num_epochs = 10
#Compiling the model with the loss function and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Making the history function a variable to call
history = History()
#Training the model on the training data
clf = model.fit(x_train, y_train, epochs=num_epochs,  callbacks=[history], verbose=1)
#Using the predict function to test the trained model against the test set
y_pred = model.predict(x_test)
#Using the evaulate model the calculate the loss function and the accuracy of the test set in the trained model
#ann1_loss, ann1_accuracy = model.evaluate(x_test, y_test)
y_pred1 = convert(y_pred)
[ann_accuracy, ann_error, ann_sensitivity, ann_specificity] = performance(y_test, y_pred1)

# num_epochs = 100
# model = Sequential()
# model.add(Dense(100, activation='sigmoid', input_shape=(111,)))
# model.add(Dense(100, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = History()
# time_callback = TimeHistory()
# clf = model.fit(x_train, y_train, epochs=num_epochs,  callbacks=[history, time_callback], verbose=1)
# y_pred_100 = model.predict(x_test)
# y_pred_100_1 = convert(y_pred_100)
# [ann_accuracy_100, ann_error_100, ann_sensitivity_100, ann_specificity_100] = performance(y_test, y_pred_100_1)

# model2 = Sequential()
# model2.add(Dense(300, activation='sigmoid', input_shape=(111,)))
# model2.add(Dense(300, activation='sigmoid'))
# model2.add(Dense(1, activation='sigmoid'))
# model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #num_epochs = 300
# history2 = History()
# time_callback2 = TimeHistory()
# clf = model2.fit(x_train, y_train, epochs=num_epochs,  callbacks=[history2, time_callback2], verbose=1)
# y_pred_300 = model2.predict(x_test)
# y_pred_300_1 = convert(y_pred_300)
# [ann_accuracy_300, ann_error_300, ann_sensitivity_300, ann_specificity_300] = performance(y_test, y_pred_300_1)
# print("Done 300")

# model3 = Sequential()
# model3.add(Dense(500, activation='sigmoid', input_shape=(111,)))
# model3.add(Dense(500, activation='sigmoid'))
# model3.add(Dense(1, activation='sigmoid'))
# model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #num_epochs = 300
# history3 = History()
# time_callback3 = TimeHistory()
# clf = model3.fit(x_train, y_train, epochs=num_epochs,  callbacks=[history3, time_callback3], verbose=1)
# y_pred_500 = model2.predict(x_test)
# y_pred_500_1 = convert(y_pred_500)
# [ann_accuracy_500, ann_error_500, ann_sensitivity_500, ann_specificity_500] = performance(y_test, y_pred_500_1)
# print("Done 500")

# model4 = Sequential()
# model4.add(Dense(1000, activation='sigmoid', input_shape=(111,)))
# model4.add(Dense(1000, activation='sigmoid'))
# model4.add(Dense(1, activation='sigmoid'))
# model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #num_epochs = 300
# history4 = History()
# time_callback4 = TimeHistory()
# clf = model4.fit(x_train, y_train, epochs=num_epochs,  callbacks=[history4, time_callback4], verbose=1)
# y_pred_1000 = model.predict(x_test)
# y_pred_1000_1 = convert(y_pred_1000)
# [ann_accuracy_1000, ann_error_1000, ann_sensitivity_1000, ann_specificity_1000] = performance(y_test, y_pred_1000_1)
# print("Done 1000")

##DURING THE COMPARISON OF THE MODELS A FILE WAS USED TO APPEND THE RESULTS TOO
# Initialising a text file to paste the metric results
# file = open("Results.txt", "a")
# file.write('Test 3 \n')
# file.write('Artificial Neural Network \n')
# file.write('500 Epochs\n')
# file.write('activation = sigmoid and sigmoid, loss = binary_crossentropy, optimizer = adam \n')
# # file.write(f'100, RF_5 accuracy: {RF_5_accuracy}\n')
# # file.write(f'100, RF_5 error: {RF_5_error}\n')
# # file.write(f'100, RF_5 sensitivity: {RF_5_sensitivity}\n')
# # file.write(f'100, RF_5 specificity: {RF_5_specificity}\n')
# # file.write(f'1000, RF_5 accuracy: {RF_5_2_accuracy}\n')
# # file.write(f'1000, RF_5 error: {RF_5_2_error}\n')
# # file.write(f'1000, RF_5 sensitivity: {RF_5_2_sensitivity}\n')
# # file.write(f'1000, RF_5 specificity: {RF_5_2_specificity}\n')
# # file.write(f'100, RF_50 accuracy: {RF_50_accuracy}\n')
# # file.write(f'100, RF_50 error: {RF_50_error}\n')
# # file.write(f'100, RF_50 sensitivity: {RF_50_sensitivity}\n')
# # file.write(f'100, RF_50 specificity: {RF_50_specificity}\n')
# # file.write(f'1000, RF_50 accuracy: {RF_50_2_accuracy}\n')
# # file.write(f'1000, RF_50 error: {RF_50_2_error}\n')
# # file.write(f'1000, RF_50 sensitivity: {RF_50_2_sensitivity}\n')
# # file.write(f'1000, RF_50 specificity: {RF_50_2_specificity}\n')
# # file.write(f'LR 100 accuracy: {LR_100_accuracy}\n')
# # file.write(f'LR 100 error_rate: {LR_100_error}\n')
# # file.write(f'LR 100 sensitivity: {LR_100_sensitivity}\n')
# # file.write(f'LR 100 specificity: {LR_100_specificity}\n')
# # file.write(f'LR 300 accuracy: {LR_300_accuracy}\n')
# # file.write(f'LR 300 error_rate: {LR_300_error}\n')
# # file.write(f'LR 300 sensitivity: {LR_300_sensitivity}\n')
# # file.write(f'LR 300 specificity: {LR_300_specificity}\n')
# # file.write(f'LR 500 accuracy: {LR_500_accuracy}\n')
# # file.write(f'LR 500 error_rate: {LR_500_error}\n')
# # file.write(f'LR 500 sensitivity: {LR_500_sensitivity}\n')
# # file.write(f'LR 500 specificity: {LR_500_specificity}\n')
# # file.write(f'LR 1000 accuracy: {LR_1000_accuracy}\n')
# # file.write(f'LR 1000 error_rate: {LR_1000_error}\n')
# # file.write(f'LR 1000 sensitivity: {LR_1000_sensitivity}\n')
# # file.write(f'LR 1000 specificity: {LR_1000_specificity}\n')
# # file.write(f'SVM 100 accuracy: {SVM_100_accuracy}\n')
# # file.write(f'SVM 100 error_rate: {SVM_100_error}\n')
# # file.write(f'SVM 100 sensitivity: {SVM_100_sensitivity}\n')
# # file.write(f'SVM 100 specificity: {SVM_100_specificity}\n')
# # file.write(f'SVM 300 accuracy: {SVM_300_accuracy}\n')
# # file.write(f'SVM 300 error_rate: {SVM_300_error}\n')
# # file.write(f'SVM 300 sensitivity: {SVM_300_sensitivity}\n')
# # file.write(f'SVM 300 specificity: {SVM_300_specificity}\n')
# # file.write(f'SVM 500 accuracy: {SVM_500_accuracy}\n')
# # file.write(f'SVM 500 error_rate: {SVM_500_error}\n')
# # file.write(f'SVM 500 sensitivity: {SVM_500_sensitivity}\n')
# # file.write(f'SVM 500 specificity: {SVM_500_specificity}\n')
# # file.write(f'SVM 1000 accuracy: {SVM_1000_accuracy}\n')
# # file.write(f'SVM 1000 error_rate: {SVM_1000_error}\n')
# # file.write(f'SVM 1000 sensitivity: {SVM_1000_sensitivity}\n')
# # file.write(f'SVM 1000 specificity: {SVM_1000_specificity}\n')
# # print('NN accuracy: ', NN_accuracy)
# # print('ANN 100 accuracy: ', ann_accuracy_100)
# # print('ANN 100 error_rate: ', ann_error_100)
# # print('ANN 100 sensitivity: ', ann_sensitivity_100)
# # print('ANN 100 specificity: ', ann_specificity_100)
# # print('ANN 100 accuracy: ', ann_accuracy_300)
# # print('ANN 100 error_rate: ', ann_error_100)
# # print('ANN 100 sensitivity: ', ann_sensitivity_100)
# # print('ANN 100 specificity: ', ann_specificity_100)
# file.write(f'ANN 100 accuracy: {ann_accuracy_100}\n')
# file.write(f'ANN 100 error_rate: {ann_error_100}\n')
# file.write(f'ANN 100 sensitivity: {ann_sensitivity_100}\n')
# file.write(f'ANN 100 specificity: {ann_specificity_100}\n')
# file.write(f'Times for each of the epochs: {time_callback.times}\n')
# file.write(f'Average: {average(time_callback.times)}\n\n')
# file.write(f'ANN 300 accuracy: {ann_accuracy_300}\n')
# file.write(f'ANN 300 error_rate: {ann_error_300}\n')
# file.write(f'ANN 300 sensitivity: {ann_sensitivity_300}\n')
# file.write(f'ANN 300 specificity: {ann_specificity_300}\n')
# file.write(f'Times for each of the epochs: {time_callback2.times}\n')
# file.write(f'Average: {average(time_callback2.times)}\n\n')
# file.write(f'ANN 500 accuracy: {ann_accuracy_500}\n')
# file.write(f'ANN 500 error_rate: {ann_error_500}\n')
# file.write(f'ANN 500 sensitivity: {ann_sensitivity_500}\n')
# file.write(f'ANN 500 specificity: {ann_specificity_500}\n')
# file.write(f'Times for each of the epochs: {time_callback3.times}\n')
# file.write(f'Average: {average(time_callback3.times)}\n\n')
# file.write(f'ANN 1000 accuracy: {ann_accuracy_1000}\n')
# file.write(f'ANN 1000 error_rate: {ann_error_1000}\n')
# file.write(f'ANN 1000 sensitivity: {ann_sensitivity_1000}\n')
# file.write(f'ANN 1000 specificity: {ann_specificity_1000}\n')
# file.write(f'Times for each of the epochs: {time_callback4.times}\n')
# file.write(f'Average: {average(time_callback4.times)}\n\n')

##CROSS VALIDATION USED AS A FINAL TEST TO ENUSURE THAT THE MODELS WERE WORKING AS REQUIRED
## Using the KFold function to set the cross validation variable
kf = KFold(n_splits=10, shuffle=True)
#Using the function to train the RFC against the training data with 1000 trees and node size of 5
RF_5_2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, verbose=1)
#Using the function of LR with a max iterations of 100
LR_100 = LogisticRegression(max_iter=100, random_state=0, solver='saga', multi_class='ovr', verbose=1)
#Using the function of SVM with a max iteration of 100
SVM_100 = svm.LinearSVC(max_iter=100, verbose=1)
#Initialisng the ANN with the activation functions of relu and sigmoid, with 100 layers thorughout
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(111,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = History()
time_callback = TimeHistory()

RFC_accuracy, LR_accuracy, SVM_accuracy, ANN_accuracy = [], [], [], []

##Using the KFold function to split the data based on the variable above
for train_index, test_index in kf.split(dataset):
    ##Splitting the data into the training and testing, and further into x and y
    X_Train, X_Test = dataset[train_index], dataset[test_index]
    X_Train, X_Test = np.delete(X_Train, 9 , 1), np.delete(X_Test, 9 , 1)
    Y_Train, Y_Test = dataset[train_index][:,9], dataset[test_index][:,9]
    ##Shuffing the data within the x and y
    X_Train, Y_Train = shuffle(X_Train, Y_Train)
    X_Test, Y_Test = shuffle(X_Test, Y_Test)
    ##Scaling and normalizing the feature data
    X_Train, X_Test = scaler.fit_transform(X_Train), scaler.fit_transform(X_Test)
    X_Train, X_Test, Y_Train, Y_Test = np.asarray(X_Train).astype(np.float32), np.asarray(X_Test).astype(np.float32),np.asarray(Y_Train).astype(np.float32), np.asarray(Y_Test).astype(np.float32)
    
    ##Appling the x and y training variables against each of the constructed models
    RF_5_2.fit(X_Train, Y_Train)
    #Using the predict function to apply the testing data to the trained model
    RF_5_2_pred = RF_5_2.predict(X_Test)
    #[RF_5_2_accuracy, RF_5_2_error, RF_5_2_sensitivity, RF_5_2_specificity] = performance(y_test, RF_5_2_pred)
    RFC_accuracy.append(accuracy_score(Y_Test, RF_5_2_pred.round()))

    LR_100.fit(X_Train, Y_Train)
    LR_100_pred = LR_100.predict(X_Test)
    # [LR_100_accuracy, LR_100_error, LR_100_sensitivity, LR_100_specificity] = performance(y_test, LR_100_pred)
    LR_accuracy.append(accuracy_score(Y_Test, LR_100_pred.round()))

    SVM_100.fit(X_Train, Y_Train)
    SVM_100_pred = SVM_100.predict(X_Test)
    # [SVM_100_accuracy, SVM_100_error, SVM_100_sensitivity, SVM_100_specificity] = performance(y_test, SVM_100_pred)
    SVM_accuracy.append(accuracy_score(Y_Test, SVM_100_pred.round()))

    clf = model.fit(X_Train, Y_Train, epochs=100,  callbacks=[history, time_callback], verbose=1)
    y_pred_100 = model.predict(X_Test)
    y_pred_100_1 = convert(y_pred_100)
    # [ann_accuracy_300, ann_error_300, ann_sensitivity_300, ann_specificity_300] = performance(y_test, y_pred_300_1)
    ANN_accuracy.append(accuracy_score(Y_Test, y_pred_100.round()))

    
# ##Initialising a text file to paste the metric results
# file = open("Cross_results.txt", "a")

# file.write('Test 1 \n')
# file.write(f'RFC_accuracy: {RFC_accuracy}\n')
# file.write(f'LR_accuracy: {LR_accuracy}\n')
# file.write(f'SVM_accuracy: {SVM_accuracy}\n')
# file.write(f'ANN_accuracy: {ANN_accuracy}\n')

##Finding the mean of each of the accuracy arrays and outputting it
#[RF_5_2_accuracy, RF_5_2_error, RF_5_2_sensitivity, RF_5_2_specificity] = performance(y_test, RF_5_2_pred)
print('RF_1000_5: ', np.mean(RFC_accuracy))
#file.write(f'RFC_mean: {np.mean(RFC_accuracy)}\n')
#[LR_100_accuracy, LR_100_error, LR_100_sensitivity, LR_100_specificity] = performance(y_test, LR_100_pred)
print('LR_100: ', np.mean(LR_accuracy))
#file.write(f'LR_mean: {np.mean(LR_accuracy)}\n')
#[SVM_100_accuracy, SVM_100_error, SVM_100_sensitivity, SVM_100_specificity] = performance(y_test, SVM_100_pred)
print('SVM_100: ', np.mean(SVM_accuracy))
#file.write(f'SVM_mean: {np.mean(SVM_accuracy)}\n')
#[ann_accuracy_300, ann_error_300, ann_sensitivity_300, ann_specificity_300] = performance(y_test, y_pred_300_1)
print('ANN_300: ', np.mean(ANN_accuracy))
#file.write(f'ANN_mean: {np.mean(ANN_accuracy)}\n')
# file.close()
