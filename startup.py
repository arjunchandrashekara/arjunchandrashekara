#importing the libraries for multi linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import seaborn as sns

#importing tbe file
startups=pd.read_csv("E://excelr data//assignments//multi linear regression//50_Startups.csv")
startups.shape
startups.corr()
sns.pairplot(startups)

#renaming the dataset
new=startups.rename(columns={'R&D Spend':'RnD'})
new=new.rename(columns={'Marketing Spend':'Marketing'})

#buliding the regression model 
import statsmodels.formula.api as smf
model=smf.ols('Profit~RnD+Marketing+Administration',data=new).fit()
model.summary()
prediction=model.predict(new)
model.params
model2=smf.ols('Profit~RnD',data=new).fit()
model2.summary()
model3=smf.ols('Profit~Marketing',data=new).fit()
model3.summary()
model4=smf.ols('Profit~RnD+Marketing',data=new).fit()
model4.summary()

import statsmodels.api as sm
sm.graphics.influence_plot(model)

st_new=new.drop(new.index[[45,46,48,49,19,6]],axis=0)

model_st=smf.ols('Profit~RnD+Marketing+Administration',data=st_new).fit()
model_st.summary()

prediction=model_st.predict(st_new)

plt2.scatter(st_new.Profit,prediction)

np.corrcoef(st_new.Profit,prediction)

rsq_hp=smf.ols('RnD~Marketing+Administration',data=st_new).fit().rsquared
vif=1/(1-rsq_hp)
vif

rsq_hp=smf.ols('Marketing~RnD+Administration',data=st_new).fit().rsquared
vif=1/(1-rsq_hp)
vif

rsq_hp=smf.ols('Administration~Marketing+RnD',data=st_new).fit().rsquared
vif=1/(1-rsq_hp)
vif

sm.graphics.plot_partregress_grid(model_st)
