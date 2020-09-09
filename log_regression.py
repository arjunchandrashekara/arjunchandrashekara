import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
import seaborn as sns

credit_card=pd.read_csv("E://excelr data//assignments//logistic regression//creditcard.csv")
credit_card.columns
credit_card=credit_card.drop('Unnamed: 0',axis=1)
sns.countplot(x='card',data=credit_card)
sns.countplot(x='owner',data=credit_card)
sns.countplot(x='selfemp',data=credit_card)
credit_card.isnull().sum()
sns.countplot(x='card',hue='owner',data=credit_card)
sns.countplot(x='owner',hue='selfemp',data=credit_card)
sns.countplot(x='selfemp',hue='card',data=credit_card)

#getting the dummy vraibles
x=pd.get_dummies(credit_card['card'],drop_first=True)
y=pd.get_dummies(credit_card['owner'],drop_first=True)
z=pd.get_dummies(credit_card['selfemp'],drop_first=True)
x.rename(columns={'yes':'card'},inplace=True)
y.rename(columns={'yes':'owner'},inplace=True)
z.rename(columns={'yes':'selfemp'},inplace=True)

new=credit_card.drop(['card','owner','selfemp'],axis=1)
new_log= pd.concat([new,x,y,z],axis=1)

#train and test data
x=new_log.drop('card',axis=1)
y=new_log['card']

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
