import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
import seaborn as sns

bank=pd.read_csv("E://excelr data//assignments//logistic regression//bank-full.csv")
bank.columns

sns.countplot(x='job',data=bank)
sns.countplot(x='marital',data=bank)
sns.countplot(x='education',data=bank)
sns.countplot(x='default',data=bank)
sns.countplot(x='housing',data=bank)
sns.countplot(x='loan',data=bank)
sns.countplot(x='contact',data=bank)
sns.countplot(x='month',data=bank)
bank.isnull().sum()

#data wrangling
bank_2=bank.drop(['job','education','contact','month','poutcome'],axis=1)
#getting the dummy vraibles
x=pd.get_dummies(bank_2['marital'],drop_first=True)
y=pd.get_dummies(bank_2['default'],drop_first=True)
z=pd.get_dummies(bank_2['housing'],drop_first=True)
a=pd.get_dummies(bank_2['loan'],drop_first=True)

y.rename(columns={'yes':'default'},inplace=True)
z.rename(columns={'yes':'housing'},inplace=True)
a.rename(columns={'yes':'loan'},inplace=True)

new=bank_2.drop(['marital','default','housing','loan'],axis=1)
new_log= pd.concat([new,x,y,z,a],axis=1)

#train and test data
x=new_log.drop('y',axis=1)
y=new_log['y']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)

from sklearn.metrics import classification_report
classification_report(y_test,predictions)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
(12924+301)/(12924+301+263+1432)
