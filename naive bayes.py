import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB as gnb

data_train=pd.read_csv("E://excelr data//assignments//Naive Bayes//SalaryData_Train.csv")
data_test=pd.read_csv("E://excelr data//assignments//Naive Bayes//SalaryData_Test.csv")

data_train['workclass'],_=pd.factorize(data_train['workclass'])
data_train['education'],_=pd.factorize(data_train['education'])
data_train['maritalstatus'],_=pd.factorize(data_train['maritalstatus'])
data_train['occupation'],_=pd.factorize(data_train['occupation'])
data_train['relationship'],_=pd.factorize(data_train['relationship'])
data_train['race'],_=pd.factorize(data_train['race'])
data_train['sex'],_=pd.factorize(data_train['sex'])
data_train['native'],_=pd.factorize(data_train['native'])
data_train['Salary'],_=pd.factorize(data_train['Salary'])

np.max(data_train.age)
np.min(data_train.age)
np.min(data_train.hoursperweek)
np.max(data_train.hoursperweek)

def age(x):
    if(16<=x<20):
        return(0)
    elif (20<x<=30):
        return(1)
    elif(30<x<=50):
        return(2)
    else:
        return(3)

def hoursperweek(x):
    if(x<50):
        return(0)
    else:
        return(1)
        
def capitalgain(x):
    if(x==0):
        return(0)
    else:
        return(1)

data_train['age'],_=pd.factorize(data_train['age'].apply(age))
data_train['hoursperweek'],_=pd.factorize(data_train['hoursperweek'].apply(hoursperweek))
data_train['capitalgain'],_=pd.factorize(data_train['capitalgain'].apply(capitalgain))
data_train=data_train.drop('educationno',axis=1)  
data_train=data_train.drop('capitalgain',axis=1) 

data_test['workclass'],_=pd.factorize(data_test['workclass'])
data_test['education'],_=pd.factorize(data_test['education'])
data_test['maritalstatus'],_=pd.factorize(data_test['maritalstatus'])
data_test['occupation'],_=pd.factorize(data_test['occupation'])
data_test['relationship'],_=pd.factorize(data_test['relationship'])
data_test['race'],_=pd.factorize(data_test['race'])
data_test['sex'],_=pd.factorize(data_test['sex'])
data_test['native'],_=pd.factorize(data_test['native'])
data_test['Salary'],_=pd.factorize(data_test['Salary'])

np.max(data_test.age)
np.min(data_test.age)
np.min(data_test.hoursperweek)
np.max(data_test.hoursperweek)

def age_test(x):
    if(16<=x<20):
        return(0)
    elif (20<x<=30):
        return(1)
    elif(30<x<=50):
        return(2)
    else:
        return(3)

def hoursperweek_test(x):
    if(x<50):
        return(0)
    else:
        return(1)
        
def capitalgain_test(x):
    if(x==0):
        return(0)
    else:
        return(1)

data_test['age'],_=pd.factorize(data_test['age'].apply(age_test))
data_test['hoursperweek'],_=pd.factorize(data_test['hoursperweek'].apply(hoursperweek_test))
data_test['capitalgain'],_=pd.factorize(data_test['capitalgain'].apply(capitalgain_test))
data_test=data_test.drop('educationno',axis=1) 
data_test=data_test.drop('capitalgain',axis=1)  
    
x_train=data_train.drop("Salary",axis=1)
y_train=data_train["Salary"]

x_test=data_test.drop("Salary",axis=1)
y_test=data_test["Salary"]

model=gnb().fit(x_train,y_train)

prediction=model.predict(x_test)

#accuracy
np.mean(y_test==prediction)
