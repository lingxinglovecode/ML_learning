import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import pickle
import os
from os import path
from sklearn.preprocessing import scale
current_dir = os.getcwd()
train_data = pd.read_csv("D://personal_file//Github//ML_learning//data//MNIST//train.csv//train.csv")
train_X = train_data.iloc[:,1:].to_numpy()
train_Y = train_data.iloc[:,0].to_numpy()
test_data = pd.read_csv("D://personal_file//Github//ML_learning//data//MNIST//train.csv//test.csv")
test_X = test_data.iloc[:,1:].to_numpy()
test_Y = test_data.iloc[:,0].to_numpy()
# preprocess the data
train_X = train_X / 255.0
test_X = test_X / 255.0

train_X = scale(train_X)
test_X = scale(test_X)

sample_x = train_data.iloc[1,1:].to_numpy().reshape(1,-1)



#Check if there exist a model trained
if os.path.exists(os.path.join(current_dir,"linear_svm.p")):
    linear_svm = pickle.load(open("linear_svm.p","rb"))
else:
# IF no trained model found, train and save
    linear_svm = LinearSVC()
    linear_svm.fit(train_X,train_Y)
    fp = open(os.path.join(current_dir,"linear_svm.p"),'wb')
    pickle.dump(linear_svm,fp)


#sample from train data and test
print("从测试集中获取一个sample用来测试模型：")
sample_x = train_X[2].reshape(1,-1)
sample_y = train_Y[2]
print("预测分类："+str(linear_svm.predict(sample_x)))
print("实际类别："+str(sample_y))




# # Save the trained model as a pickle string.
# saved_model = pickle.dumps(knn)
#
# # Load the pickled model
# knn_from_pickle = pickle.loads(saved_model)
#
# # Use the loaded pickled model to make predictions
# knn_from_pickle.predict(X_test)


