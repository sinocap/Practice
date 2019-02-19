import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn import cross_validation, metrics 
from sklearn.ensemble import RandomForestRegressor  
from sklearn.svm import SVR 
from sklearn.metrics import r2_score

df = pd.read_csv("https://raw.githubusercontent.com/DarrenCook/h2o/bk/datasets/ENB2012_data.csv", sep=',',  header=0)
 
print(df.head())

df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load'] 

print(df.head())
 
train=df.sample(frac=0.7,random_state=123456)
test=df.drop(train.index)
 
test_loads1=test[["heating_load"]]
test_loads2=test[["cooling_load"]]
 
Y1=np.array(train['heating_load'])
Y2=np.array(train['cooling_load'])

############################################################
 
print(stats.spearmanr(df['relative_compactness'],df['cooling_load']).correlation)
print(stats.spearmanr(df['surface_area'],df['cooling_load']).correlation)
print(stats.spearmanr(df['wall_area'],df['cooling_load']).correlation)
print(stats.spearmanr(df['roof_area'],df['cooling_load']).correlation)

print(stats.spearmanr(df['overall_height'],df['cooling_load']).correlation)
print(stats.spearmanr(df['orientation'],df['cooling_load']).correlation)
print(stats.spearmanr(df['glazing_area'],df['cooling_load']).correlation)
print(stats.spearmanr(df['glazing_area_distribution'],df['cooling_load']).correlation)
 
train_all=train[['overall_height','relative_compactness','roof_area','surface_area']]
test_all=test[['overall_height','relative_compactness','roof_area','surface_area']]


X_train,X_test,y_train,y_test=cross_validation.train_test_split(train_all,Y2,test_size=0.3, random_state=123456)
 
models = []
names = []

models.append(('RandomForest', RandomForestRegressor()))
models.append(('Support Vector Machine', SVR()))
 
print("\n======================")
print("Cooling Load")
print("======================")

for name, model in models:
   
    model.fit(X_train, y_train)         
    confidence=model.score(X_test, y_test) 
    prediction=model.predict(test_all) 
    r2_cv=metrics.r2_score(test_loads2,prediction)   
    names.append(name)   
  
    msg = "\n\n=============%s============= \n\n R^2 score : %.3f \n confidence : %.3f" %(name, r2_cv, confidence)
    print(msg)    
    plt.plot(np.linspace(0,229,230),test_loads2,color='red')
    plt.plot(prediction,color='blue',linewidth=1) 
    plt.legend(['Cooling Load','prediction'], loc=2)
    plt.show()
 
      
    
    
