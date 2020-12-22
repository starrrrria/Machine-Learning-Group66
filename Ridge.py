#Author: Shijun Zhang 20329500
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

df=pd.read_csv('qsar_aquatic_toxicity.csv',comment='#',header=None)  # Reads the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = np.array(df.iloc[:, 0:8])
X=scaler.fit(X).transform(X)
Y = np.array(df.iloc[:, 8]).reshape(-1, 1)
Y=scaler.fit(Y).transform(Y)
Ytrain=Y

#Change polynomialfeatures
N_range=[1,2]
from sklearn.model_selection import KFold
SquareErrorMeanArray=[]
SquareErrorVarArray=[]
SquareErrorStdArray=[]
C=5
for n in N_range:
    kf = KFold(n_splits=5)
    SquareErrorArray=[]
    Xtrain = PolynomialFeatures(n).fit_transform(X)
    for train, test in kf.split(Xtrain):
        RidgeModel=Ridge(alpha=1/2/C).fit(Xtrain[train],Ytrain[train])
        RidgeYpredict=RidgeModel.predict(Xtrain[test])
        SquareErrorArray.append(mean_squared_error(Y[test],RidgeYpredict))
    print("Ridge Regression Fold=5, n=%d square error's mean is %f" %(n,np.array(SquareErrorArray).mean()))
    print("Ridge Regression Fold=5, n=%d square error's var is %f" %(n,np.array(SquareErrorArray).var()))
    SquareErrorMeanArray.append(np.array(SquareErrorArray).mean())
    SquareErrorVarArray.append(np.array(SquareErrorArray).var())
    SquareErrorStdArray.append(np.array(SquareErrorArray).std())
fig=plt.figure()
plt.plot(N_range,SquareErrorMeanArray,color='blue')
plt.plot(N_range,SquareErrorVarArray,color='red')
plt.legend(["SquareError's Mean in Diffferent Folds","SquareError's Var in Diffferent Folds"])
plt.title("Ridge Regression Predict SquareError's Mean and Variance(Fold=5)")
plt.xlabel("Number of n")
plt.ylabel("Mean or Variance")
plt.show()
fig=plt.figure()
plt.errorbar(N_range,SquareErrorMeanArray,SquareErrorStdArray)
plt.show()

#Uses an L2 penalty instead of an L1 penalty in the cost function
#Change C
C_range=[0.001,0.001,0.01,0.1,1,1.5,2]
Xtrain = PolynomialFeatures(1).fit_transform(X)
from sklearn.model_selection import KFold
SquareErrorMeanArray=[]
SquareErrorVarArray=[]
SquareErrorStdArray=[]
for C in C_range:
    kf = KFold(n_splits=5)
    SquareErrorArray=[]
    for train, test in kf.split(Xtrain):
        RidgeModel=Ridge(alpha=1/2/C).fit(Xtrain[train],Ytrain[train])
        RidgeYpredict=RidgeModel.predict(Xtrain[test])
        SquareErrorArray.append(mean_squared_error(Y[test],RidgeYpredict))
    print("Ridge Regression Fold=5, C=%d square error's mean is %f" %(C,np.array(SquareErrorArray).mean()))
    print("Ridge Regression Fold=5, C=%d square error's var is %f" %(C,np.array(SquareErrorArray).var()))
    SquareErrorMeanArray.append(np.array(SquareErrorArray).mean())
    SquareErrorVarArray.append(np.array(SquareErrorArray).var())
    SquareErrorStdArray.append(np.array(SquareErrorArray).std())
fig=plt.figure()
plt.plot(C_range,SquareErrorMeanArray,color='blue')
plt.plot(C_range,SquareErrorVarArray,color='red')
plt.legend(["SquareError's Mean in Diffferent Folds","SquareError's Var in Diffferent Folds"])
plt.title("Ridge Regression Predict SquareError's Mean and Variance(Fold=5)")
plt.xlabel("Number of C")
plt.ylabel("Mean or Variance")
plt.show()
fig=plt.figure()
plt.errorbar(C_range,SquareErrorMeanArray,SquareErrorStdArray)
plt.show()

#Compare with baseline model

#Ridge modle choose n=1, C=0.1
RidgeModel=Ridge(alpha=1/2/0.1).fit(X,Ytrain)
RidgeYpred=RidgeModel.predict(X)
#Baseline model choose Y average
AverageY=float(sum(Y)) / len(Y)
baselinepredarr=[]
for i in range(0,len(Y)):
    baselinepredarr.append(AverageY)
print("Ridge model predict error mean:")
print(mean_squared_error(Y,RidgeYpred))
print("Baseline model predict error mean:")
print(mean_squared_error(Y,baselinepredarr))

# R square analysis
print("the R2 square accuracy of the ridge model is:")
print(1-mean_squared_error(Y,RidgeYpred)/mean_squared_error(Y,baselinepredarr))