import pandas as pd
iris=pd.read_csv("C://Users//Arjun//Desktop//iris.csv")
iris.columns
iris.shape
iris['setosa'].max()
iris_new = pd.DataFrame(iris[iris['setosa']==iris['setosa'].min()])
iris.describe()
a=iris.iloc[:,2]
iris.set_index('150')

dataset = {
        "day":["1st","2nd","3rd"],"temp":[28,21,28],"type":["windy","rainy","sunny"]}

dc=pd.DataFrame(dataset)
import pandas as pd
hello = pd.read_csv("C://Users//Arjun//Desktop//python data sets//Online Retail_Sample.csv",encoding="ISO-8859-1")
hello.shape


column_to_drop=["150","4"]
rows_to_drop=list(range(0,100))
iris=iris.drop(column_to_drop , axis=1).iloc[list(range(50,100))]
iris_a=iris.iloc[:,0:1].drop(iris.index(rows_to_drop))

from matplotlib import pyplot as plt
plt.plot([8,0,5],[3,4,5])
plt.show()

x=[1,2,3]
y=[2,4,5]

x1=[3,2,1]
y2=[2,4,5]

plt.plot(x,y,'g',label='line one',linewidth=5)
plt.plot(x1,y2,'y',label='line two',linewidth=5)


plt.title("info")
plt.ylabel("Y-axis")
plt.xlabel("X-axis")

plt.show()

plt.bar(x,y,label="nothing much")
plt.show()
