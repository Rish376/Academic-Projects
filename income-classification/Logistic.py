
import pandas as pd
import numpy as np

adult_df=pd.read_csv('adult_data.csv',header=None,delimiter=' *, *',engine='python')

adult_df.head()
adult_df.shape

adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

adult_df.isnull().sum()#detects only Nan values.Cannot detect categorical null values for example "?"
adult_df.info

#to find missing values in characterial variable
for value in ['workclass','education',
              'marital_status','occupation','relationship',
              'race','sex','native_country','income']:
       print(value, sum(adult_df[value] == '?'))
       
#create a copy of dataframe
adult_df_rev = pd.DataFrame.copy(adult_df)        

pd.set_option('display.max_columns',None)#view all columns
pd.set_option('display.max_rows',None)#view all rows
ad=adult_df_rev.describe(include='all')  #for categorical summary     

#for loop to fill categorical missing values

for value in['workclass','occupation','native_country']:
    adult_df_rev[value].replace(['?'],
                adult_df_rev[value].mode()[0],inplace=True)
    
#check again for missing values in categorical variable
    for value in ['workclass','education',
              'marital_status','occupation','relationship',
              'race','sex','native_country','income']:
       print(value, sum(adult_df_rev[value] == '?'))
       
adult_df_rev.workclass.value_counts()
adult_df_rev.education.value_counts()   

colname=['workclass','education','marital_status',
         'occupation','relationship','race',
         'sex','native_country','income']
colname

#for preprocessing the data
from sklearn import preprocessing

le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    adult_df_rev[x]=le[x].fit_transform(adult_df_rev.__getattr__(x))

adult_df_rev.head()
#0<- <=50k
#1<- >50k

#create x and y variable
X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]

#standardizing the data: StandardScaler()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print(X)

#convert Y into int.Precautionary steps
Y=Y.astype(int) 

#Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=0.3,random_state=10)

#Building the model
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


#store the predicted probabilities
Y_pred_prob=classifier.predict_proba(X_test)
print(Y_pred_prob)

Y_pred_class=[]
for value in Y_pred_prob[:,0]:
    if value < 0.5:
        Y_pred_class.append(1)
    else:
        Y_pred_class.append(0)
#print(Y_pred_class)
        
#convert y test to list so that both the inputs have same data structure
cfm=confusion_matrix(Y_test.tolist(),Y_pred_class)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test.tolist(),Y_pred_class))

acc=accuracy_score(Y_test.tolist(),Y_pred_class)
print("Accuracy of the model: ",acc)

for a in np.arange(0,1,0.05):
    predict_mine = np.where(Y_pred_prob[:,0] < a,1,0)
    cfm=confusion_matrix(Y_test.tolist(),predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Error at threshold",a, ":" ,total_err,\
          "type 2 error:",cfm[1,0], "type 1 error", cfm[0,1])
# 
"""
AUC
accuracy of model
50-60=Bad model
60-70=fair 
70:80=good
80:90=v good
90:100=excellent

"""
#plot auc and roc
from sklearn import metrics
#preds=classifier.pred_proba(X_test)[:,0]
fpr,tpr,threshold=metrics.roc_curve(Y_test.tolist(),Y_pred_class)
auc=metrics.auc(fpr,tpr)
print(auc)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label=auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
#
#Using cross validation

classifier=(LogisticRegression())

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

#if results are not good then 
for train_value, test_value in kfold_cv:
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

# feature Selection
#create x and y variable
X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]

#standardizing the data: StandardScaler()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()#creating object
scaler.fit(X) #standardized
X=scaler.transform(X)#replace
print(X)

#Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=0.3,random_state=10)

colname=adult_df_rev.columns[:]
# RFE
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 7)
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") 
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_)
Y_pred=model_rfe.predict(X_test)
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#univariate
X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=10)
fit1 = test.fit(X, Y)

print(fit1.scores_)
print(list(zip(colname,fit1.get_support())))
features = fit1.transform(X)

print(features)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()#creating object
scaler.fit(features) #standardized
X=scaler.transform(features)#replace

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X, Y,test_size=0.3,random_state=10)

#Building the model
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


#store the predicted probabilities
Y_pred_prob=classifier.predict_proba(X_test)
print(Y_pred_prob)

Y_pred_class=[]
for value in Y_pred_prob[:,0]:
    if value < 0.5:
        Y_pred_class.append(1)
    else:
        Y_pred_class.append(0)
#print(Y_pred_class)
        
#convert y test to list so that both the inputs have same data structure
cfm=confusion_matrix(Y_test.tolist(),Y_pred_class)
print(cfm)

print("Classification Report: ")
print(classification_report(Y_test.tolist(),Y_pred_class))

acc=accuracy_score(Y_test.tolist(),Y_pred_class)
print("Accuracy of the model: ",acc)

#boxplot
ax=plt.figure()
box=ax.add_subplot(1,1,1)
box.boxplot(adult_df['fnlwgt'])
plt.show()

#barchart
var = df.groupby('Gender').Sales.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Sales')
ax1.set_title("Gender wise Sum of Sales")

