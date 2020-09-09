import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing as p
import matplotlib.pyplot as plt
import numpy as np
data1 = pd.read_csv("E://excelr data//assignments//Decision Tree//Fraud_check.csv")

def fn(x):
    if(x<=30000):
        return(0)
    else:
        return(1)

data1['Taxable.Income']=data1['Taxable.Income'].apply(fn)
data1['Undergrad'],_=pd.factorize(data1['Undergrad'])
data1['Marital.Status'],_=pd.factorize(data1['Marital.Status'])
data1['Urban'],_=pd.factorize(data1['Urban'])

x1=data1.drop('Taxable.Income',axis=1)
y1=data1['Taxable.Income']

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.33)

classifier_1=DecisionTreeClassifier(max_depth=3,criterion = 'entropy')
classifier_1.fit(x1_train,y1_train)
prediction_1=classifier_1.predict(x1_test)
confusion_matrix(y1_test,prediction_1)
pp_1=classifier_1.predict(x1_train)

tree.plot_tree(classifier_1)

#train accuracy
np.mean(pp_1==y1_train)

#test accuracy
np.mean(prediction_1==y1_test)
