import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import seaborn as sn
import statsmodels.formula.api as smf
from sklearn import preprocessing as p

computer_data=pd.read_csv("E://excelr data//assignments//multi linear regression//Computer_Data.csv")
computer_data.columns
computer_data=computer_data.drop(['Unnamed: 0'],axis=1)

def fun(x):
    if (x=='yes'):
        return(1)
    else:
        return(0)

x=computer_data.cd.apply(fun)
y=computer_data.multi.apply(fun)
z=computer_data.premium.apply(fun)

comp=computer_data.drop(['cd','multi','premium'],axis=1)
new=pd.concat([comp,x,y,z],axis=1)

def norm(x):
    return(p.scale(x))

norm_new=new.apply(norm)

corr_table=norm_new.corr()
model=smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=norm_new).fit()
model.summary()
model_1=smf.ols('price~hd',data=norm_new).fit()
model_2=smf.ols('price~ram',data=norm_new).fit()
model_1.summary()
model_2.summary()

rsq_hp_speed = smf.ols('speed~hd+ram+screen+cd+multi+premium+ads+trend',data=norm_new).fit().rsquared  
vif_hp_speed = 1/(1-rsq_hp_speed) 
vif_hp_speed

rsq_hp_hd = smf.ols('hd~speed+ram+screen+cd+multi+premium+ads+trend',data=norm_new).fit().rsquared  
vif_hp_hd = 1/(1-rsq_hp_hd) 
vif_hp_hd

rsq_hp_ram = smf.ols('ram~hd+speed+screen+cd+multi+premium+ads+trend',data=norm_new).fit().rsquared  
vif_hp_ram = 1/(1-rsq_hp_ram) 
vif_hp_ram

rsq_hp_screen = smf.ols('screen~ram+hd+speed+cd+multi+premium+ads+trend',data=norm_new).fit().rsquared  
vif_hp_screen = 1/(1-rsq_hp_screen) 
vif_hp_screen

rsq_hp_cd = smf.ols('cd~screen+ram+hd+speed+multi+premium+ads+trend',data=norm_new).fit().rsquared  
vif_hp_cd = 1/(1-rsq_hp_cd) 
vif_hp_cd

rsq_hp_multi = smf.ols('multi~screen+ram+hd+speed+cd+premium+ads+trend',data=norm_new).fit().rsquared  
vif_hp_multi = 1/(1-rsq_hp_multi) 
vif_hp_multi

rsq_hp_premium = smf.ols('premium~multi+screen+ram+hd+speed+cd+ads+trend',data=norm_new).fit().rsquared  
vif_hp_premium = 1/(1-rsq_hp_premium) 
vif_hp_premium

rsq_hp_ads = smf.ols('ads~premium+multi+screen+ram+hd+speed+cd+trend',data=norm_new).fit().rsquared  
vif_hp_ads = 1/(1-rsq_hp_ads) 
vif_hp_ads

rsq_hp_trend = smf.ols('trend~ads+premium+multi+screen+ram+hd+speed+cd',data=norm_new).fit().rsquared  
vif_hp_trend = 1/(1-rsq_hp_trend) 
vif_hp_trend

one=[["speed",vif_hp_speed],["hd",vif_hp_hd],["ram",vif_hp_ram],["screen",vif_hp_screen],["cd",vif_hp_cd],["multi",vif_hp_multi],["premium",vif_hp_premium],["ads",vif_hp_ads],["trend",vif_hp_trend]]
vif=pd.DataFrame(one,columns=['Name','value'])

model_mod=smf.ols('price~speed+ram+screen+cd+multi+premium+ads+trend',data=norm_new).fit()
model_mod.summary()

