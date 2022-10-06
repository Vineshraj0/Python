import pandas as pd
from sklearn.model_selection import train_test_split
pf=pd.read_csv('user_data.csv')
print(pf)

x=pf[['User ID','Gender','Age','EstimatedSalary']]
y=pf[['Purchased']]

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)


from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)  
