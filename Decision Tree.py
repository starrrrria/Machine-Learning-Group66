import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report,roc_curve
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

def plot_error(list,mean_error_all,std_error_all,title,poly_list,xName):
    plt.figure(dpi = 500)
    if np.array(mean_error_all).ndim == 2:
        for i in range(len(mean_error_all)):
            plt.errorbar(list, mean_error_all[i], yerr=std_error_all[i],label = 'k number is %.3f'%(poly_list[i]))
    else:
        plt.errorbar(list,mean_error_all,yerr=std_error_all,label = 'k number is %.3f'%(poly_list))
    plt.title('%s model mean square error bar'%(title))
    plt.legend()
    plt.xlabel("%s value "%(xName))
    plt.ylabel("error value")
    # plt.savefig("/Users/buxin/Documents/machine learning/week5/%s model mean square error bar with %s value.png"%(title,xName))
    plt.show()

def cv_score(d):
    clf = DecisionTreeRegressor(max_depth=d)
    clf.fit(X_train, y_train)
    return(clf.score(X_train, y_train), clf.score(X_test, y_test))

def plot_curve(train_sizes, cv_results, xlabel):
    # train_scores_mean = cv_results['mean_train_score']
    # train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(6, 4), dpi=120)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    # plt.fill_between(train_sizes,
    #                  train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std,
    #                  alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    # plt.plot(train_sizes, train_scores_mean, '.--', color="r",
    #          label="Training score")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

#                 rewrite dataset
# reader = csv.reader(open("/Users/buxin/Documents/machine learning/week5/qsar_aquatic_toxicity.csv","rU"),delimiter = ';')
# writer = csv.writer(open("/Users/buxin/Documents/machine learning/week5/qsar_aquatic_toxicity_1.csv","w"),delimiter = ',')
# writer.writerows(reader)
# for rows in reader:
#     for parsed_item in rows:
#         parsed_item = parsed_item.replace(';', ',')

#                read data
df = pd.read_csv('/Users/buxin/Documents/machine learning/week5/qsar_aquatic_toxicity_1.csv',header=None)
print(df.head())
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X3 = df.iloc[:,2]
X4 = df.iloc[:,3]
X5 = df.iloc[:,4]
X6 = df.iloc[:,5]
X7 = df.iloc[:,6]
X8 = df.iloc[:,7]
X = np.column_stack((X1,X2,X3,X4,X5,X6,X7,X8))
std=StandardScaler()
X=std.fit(X).transform(X)
y = np.array(df.iloc[:,8]).reshape(-1,1)
y=std.fit(y).transform(y)

#divide train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train_shape:", X_train.shape, " y_train_shape:", y_train.shape)
print("X_test_shape:", X_test.shape,"  y_test_shape:", y_test.shape)

#decide splits number
k_list = [15,20,25,30,50]
mean_error_1 = []
std_error_1 = []
for k in k_list:
    temp = []
    kf = KFold(n_splits=k)
    model = DecisionTreeRegressor(max_depth=2)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        y_predict = model.predict(X[test])
        mean_se = mean_squared_error(y[test], y_predict)
        temp.append(mean_se)
    mean_error_1.append(np.array(temp).mean())
    std_error_1.append(np.array(temp).std())

plt.errorbar(k_list,mean_error_1,yerr=std_error_1,label = 'depth number is 2')
plt.title('DecisionTree model mean square error bar')
plt.legend()
plt.xlabel("k value ")
plt.ylabel("error value")
# plt.savefig("/Users/buxin/Documents/machine learning/week5/k square error bar.png")
plt.show()

#                             decide depth GridSearchCV
# depths = np.arange(1,10)
# param_grid = {'max_depth':np.arange(1,10)}
#
# clf = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
# clf.fit(X,y)
#
# print("best_parms:{0}\nbest_score:{1}".format(clf.best_params_, clf.best_score_))
# plot_curve(np.arange(1,10), clf.cv_results_, xlabel='gini thresholds')

# for i in range(5):
#     scores = [cv_score(d) for d in depths]
#     tr_scores = [s[0] for s in scores] #train score
#     te_scores = [s[1] for s in scores] #test score
#
#     # find highest
#     tr_best_index = np.argmax(tr_scores)
#     te_best_index = np.argmax(te_scores)
#
#     print("bestdepth:", te_best_index+1, " bestdepth_score:", te_scores[te_best_index], '\n')
#
#     plt.figure(dpi=120)
#     plt.grid()
#     plt.xlabel('max depth of decison tree')
#     plt.ylabel('Scores')
#     plt.plot(depths, te_scores, label='test_scores')
#     plt.plot(depths, tr_scores, label='train_scores')
#     plt.legend()
#     # plt.savefig("/Users/buxin/Documents/machine learning/week5/decide depth.png")
#     plt.show()

#                             decide all parameter
thresholds = np.linspace(0, 0.2, 50)
param_grid = [{'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,30,2)}]
clf = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)
clf.fit(X, y)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
# plot_curve(thresholds, clf.cv_results_, xlabel='gini thresholds')

#decide depth number
Depth_list = [2,4,5,8]
splits = 50
mean_error = []
std_error = []
for depth in Depth_list:
    temp = []
    kf = KFold(n_splits=splits)
    model = DecisionTreeRegressor(max_depth=depth)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        y_predict = model.predict(X[test])
        result = model.score(X[test], y[test])
        score = r2_score(y[test], y_predict)
        # print("When depth number is ",depth,"accuracy is ", result, 'R2 score is ',score)
        mean_se = mean_squared_error(y[test], y_predict)
        temp.append(mean_se)
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

plot_error(Depth_list, mean_error, std_error, 'DecisionTree', splits, 'depth')

#baseline
AverageY=float(sum(y)) / len(y)
baselinepredarr=[]
for i in range(0,len(y)):
    baselinepredarr.append(AverageY)
print("Baseline model predict error mean:")
print(mean_squared_error(y,baselinepredarr))

# final model
model_f = DecisionTreeRegressor(max_depth=5)
model_f.fit(X, y)
y_predict_f = model_f.predict(X).reshape(-1,1)
result_f = model_f.score(X, y)
score_f = r2_score(y, y_predict_f)
mean_se_f = mean_squared_error(y, y_predict_f)
print("\naccuracy is ", result_f, 'R2 score is ',score_f,'MSE is ',mean_se_f)
plt.scatter(y_predict_f, y_predict_f-y, label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, colors='red')
plt.xlim([-10, 50])
# plt.savefig("/Users/buxin/Documents/machine learning/week5/final residuals.png")
plt.show()


