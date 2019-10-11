# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:10:34 2018

@author: Rutuja
"""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns",None)

network_data=pd.read_csv("OBS_Network_data.csv",header=None,delimiter=" *, *",engine='python')

network_data.columns=["Node","Utilised Bandwith Rate","Packet Drop Rate","Full_Bandwidth","Average_Delay_Time_Per_Sec",
"Percentage_Of_Lost_Pcaket_Rate","Percentage_Of_Lost_Byte_Rate","Packet Received Rate","of Used_Bandwidth",
"Lost_Bandwidth","Packet Size_Byte","Packet_Transmitted","Packet_Received","Packet_lost","Transmitted_Byte",
"Received_Byte","10-Run-AVG-Drop-Rate","10-Run-AVG-Bandwith-Use","10-Run-Delay","Node Status","Flood Status","Class"]

network_data.head()
network_data.info()
network_data.isnull().sum()

#create copy of data
network_data_rev=pd.DataFrame.copy(network_data)
network_data_rev.head()

#drop Packet Size_Byte as it contains same value
network_data_rev=network_data_rev.drop('Packet Size_Byte',axis=1)
network_data_rev.shape

#label encoding
colname=['Node','Full_Bandwidth','Node Status','Class']

from sklearn import preprocessing
le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    network_data_rev[x]=le[x].fit_transform(network_data_rev.__getattr__(x))

network_data_rev.head()

#create X and Y variable
Y=network_data_rev.values[:,-1]
X=network_data_rev.values[:,:-1]

#standardize 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(X)

X=scaler.transform(X)

print(X)

Y=Y.astype(int)

#splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=0.3,random_state=10)
#random state= Seed

#%%
#Running Decision Tree model
#predicting using Decision_Tree_classifier
from sklearn.tree import DecisionTreeClassifier

model_dt=DecisionTreeClassifier()
model_dt.fit(X_train,Y_train)

#fit the model on the data and predict the values
Y_pred=model_dt.predict(X_test)
#print(Y_pred)
print(list(zip(Y_test,Y_pred)))

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#cross validation

classifier=DecisionTreeClassifier()

from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%% Logistic
from sklearn.linear_model import LogisticRegression
classifier=(LogisticRegression())
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

#evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


#%%  SVM
from sklearn import svm
#from sklearn import LogisticRegression
#svc_model=LogisticRegression()
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
svc_model.fit(X_train, Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))

Y_pred_col=list(Y_pred)

#evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%% export decsion trees
from sklearn import tree
with open("model_DecisionTree.txt","w") as f:
    f=tree.export_graphviz(model_dt,out_file=f) #export_graphviz saves all the parameters
#copy all the text from that txt file and go to webgraphwiz.com to view the tree
#%% Extra tree classifier
from sklearn.ensemble import ExtraTreesClassifier
model=(ExtraTreesClassifier(21)) #21=no of estimator
#fit the mvaluesodel on data and predict the 
model=model.fit(X_train,Y_train)

#fit the model on the data and predict the values
Y_pred=model.predict(X_test)

#evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%% random forest 
from sklearn.ensemble import RandomForestClassifier
model=(RandomForestClassifier(501)) #501=no of trees
#fit the model on data and predict the 
model=model.fit(X_train,Y_train)

#fit the model on the data and predict the values
Y_pred=model.predict(X_test)

#evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%% Running adaboost
from sklearn.ensemble import AdaBoostClassifier
model_AdaBoost=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100))
#fit the model on data and predict the 
model_AdaBoost.fit(X_train,Y_train)

#fit the model on the data and predict the values
Y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report
#accuracy
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%% Gradient boosting

from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
#fit the model on data and predict the values 
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score ,classification_report
#accuracy
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)
#%% Ensemble modelling
from sklearn.ensemble import VotingClassifier
#create sub models
estimators=[]
model1=LogisticRegression()
estimators.append(('log',model1))
model2=DecisionTreeClassifier()
estimators.append(('cart',model2))
model3=svm.SVC()
estimators.append(('svm',model3))
#print(estimators)
#%%create ensemble models
ensemble=VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)

#accuracy
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


