import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2

salary_data=pd.read_csv("E://excelr data//assignments//simple regression//original dataset//Salary_Data.csv")
salary_data.columns

plt.hist(salary_data.YearsExperience)
plt.boxplot(salary_data.YearsExperience)

plt.hist(salary_data.Salary)
plt.boxplot(salary_data.Salary)

plt.plot(salary_data.YearsExperience,salary_data.Salary,"bo");plt.xlabel("YearsExperience");plt.ylabel("Salary")

np.corrcoef(salary_data.YearsExperience,salary_data.Salary)

import statsmodel.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=salary_data).fit()
model.summary()

pred=model.predict(salary_data.iloc[:,0])

plt2.scatter(salary_data.YearsExperience,salary_data.Salary,color="red");plt2.plot(salary_data.YearsExperience,pred,color="black")

emp_data=pd.read_csv("E://excelr data//assignments//simple regression//original dataset//emp_data.csv")
emp_data.columns
plt.hist(emp_data.Salary_hike)
plt.boxplot(emp_data.Salary_hike)
sum(emp_data["Salary_hike"].isnull())


plt.hist(emp_data.Churn_out_rate)
plt.boxplot(emp_data.Churn_out_rate)
sum(emp_data.Churn_out_rate.isnull())

np.corrcoef(emp_data.Salary_hike,emp_data.Churn_out_rate)

import statsmodels.formula.api as srl
model2=srl.ols("Salary_hike~Churn_out_rate",data=emp_data).fit()
model2.summary()
prediction=model2.predict(emp_data.iloc[:,1])

plt2.scatter(emp_data.Salary_hike,emp_data.Churn_out_rate,color="red");plt2.plot(emp_data.Salary_hike,prediction,color="black")

delivery_time=pd.read_csv("E://excelr data//assignments//simple regression//original dataset//delivery_time.csv")

delivery_time


#histograms
plt.hist(delivery_time["Delivery Time"])
plt.hist(delivery_time["Sorting Time"])

#boxplots
plt.boxplot(delivery_time["Delivery Time"])
plt.boxplot(delivery_time["Sorting Time"])

np.corrcoef(delivery_time["Delivery Time"],delivery_time["Sorting Time"])

new=delivery_time.rename(columns={'Delivery Time':"DeliveryTime"})

new=new.rename(columns={'Sorting Time':"SortingTime"})

import statsmodels.formula.api as smf

model4=smf.ols("DeliveryTime~SortingTime",data=new).fit()

model4.summary()

model5=smf.ols("DeliveryTime~np.log(SortingTime)",data=new).fit()

model5.summary()

model6=smf.ols("np.log(DeliveryTime)~np.log(SortingTime)",data=new).fit()

model6.summary()

model7=smf.ols("np.log(DeliveryTime)~SortingTime",data=new).fit()

model7.summary()

model8=smf.ols("DeliveryTime~np.exp(SortingTime)",data=new).fit()

model8.summary()

model9=smf.ols("np.exp(DeliveryTime)~np.exp(SortingTime)",data=new).fit()

model9.summary()

model10=smf.ols("np.exp(DeliveryTime)~SortingTime",data=new).fit()

model10.summary()

prediction=model10.predict(new.iloc[:,1])

plt2.scatter(new.SortingTime,new.DeliveryTime);plt2.plot(new.SortingTime,prediction)


calory=pd.read_csv("E://excelr data//assignments//simple regression//original dataset//calories_consumed.csv")

calory.shape

cal=calory.rename(columns={'Calories Consumed':"calories"})

cal=cal.rename(columns={'Weight gained (grams)':"Weight"})

plt.hist(cal.calories)
plt.hist(cal.Weight)

plt.boxplot(cal.calories)
plt.boxplot(cal.Weight)


np.corrcoef(cal.calories,cal.Weight)

import statsmodels.formula.api as smf

models=smf.ols("Weight~calories",data=cal).fit()

models.summary()

preds=models.predict(cal.iloc[:,1])

plt2.scatter(cal.calories,cal.Weight);plt2.plot(cal.calories,preds)

