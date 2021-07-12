import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import pickle
import os
from os import path
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


class My_SVM:
    def __init__(self,data_dir=None,use_trained=False):
        self.data_dir = data_dir
        self.use_trained = use_trained
        self.linear_svm = None

    def read_data(self):

        train_data = pd.read_csv("D://personal_file//Github//ML_learning//data//MNIST//train//train.csv")
        X = train_data.iloc[:,1:].to_numpy()
        Y = train_data.iloc[:,0].to_numpy()
        train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3,random_state=1,shuffle=True)
        return train_X,test_X,train_Y,test_Y

    def preprocess(self,train_X,test_X):

        # preprocess the data
        train_X = train_X / 255.0
        test_X = test_X / 255.0

        train_X = scale(train_X)
        test_X = scale(test_X)
        return train_X,test_X

    def train_model(self,train_X,train_Y):
        current_dir = os.getcwd()
        #Check if there exist a model trained
        if self.use_trained:
            if os.path.exists(os.path.join(current_dir,"linear_svm.p")):
                self.linear_svm = pickle.load(open("linear_svm.p","rb"))
                return
            else:
                print("No trained model found!")
        # IF no trained model found, train and save
        self.linear_svm = LinearSVC()
        self.linear_svm.fit(train_X,train_Y)
        fp = open(os.path.join(current_dir,"linear_svm.p"),'wb')
        pickle.dump(self.linear_svm,fp)
        return

    def hyperparam_search(self,X,Y,space):

        # hyper-param search
        cross_val = RepeatedStratifiedKFold(n_splits=3,n_repeats=1,random_state=1)
        search = RandomizedSearchCV(self.linear_svm,space,n_iter=2,scoring='accuracy',cv=cross_val)
        result = search.fit(X,Y)
        print("最好的分类成绩：" + str(result.best_score_))
        print("最好的超参数：" + str(result.best_params_))

    def compute_acc(self,test_X,test_Y):

        # predict on test set // compute accuracy
        y_predict_test = self.linear_svm.predict(test_X)
        accuracy = accuracy_score(test_Y,y_predict_test)
        print("在测试集上的准确率:"+str(accuracy))
        con_matrix = confusion_matrix(test_Y,y_predict_test)
        print("混淆矩阵：")
        print(con_matrix)



if __name__ == '__main__':
    #data
    data_dir = 'D://personal_file//Github//ML_learning//data//MNIST//train//train.csv'
    svm = My_SVM(data_dir,use_trained=True)
    train_X,test_X,train_Y,test_Y = svm.read_data()
    train_X, test_X= svm.preprocess(train_X,test_X)
    space = dict()
    space['C'] = loguniform(1e-2, 100)
    svm.train_model(train_X,train_Y)
    svm.hyperparam_search(train_X,train_Y,space)
    svm.compute_acc(test_X,test_Y)

