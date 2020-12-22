import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("qsar_aquatic_toxicity_1.csv", comment="#", header=None)
df.columns=['TPSA(Tot)', 'SAacc', 'H-050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C-040', 'quantitative response']

def knn(weight):
    kf = KFold(n_splits=5)
    mean_error=[]
    std_error=[]
    scaler = StandardScaler()
    X = np.array(df.iloc[:, 0:8])
    X=scaler.fit(X).transform(X)
    y = np.array(df.iloc[:, 8]).reshape(-1, 1)
    y=scaler.fit(y).transform(y)

    for i in range(1, 20):
        temp=[] 
        for train, test in kf.split(X):
            model = KNeighborsRegressor(n_neighbors=i, weights=weight)
            model.fit(X =X[train],y=y[train])
            ypred = model.predict(X[test])
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    print('minimum mean square error', min(mean_error))
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.errorbar(list(range(1, 20)),mean_error,yerr=std_error,label='Test data')
    ax.set_xlabel('K', fontsize='x-large'); 
    ax.set_ylabel('Mean square error', fontsize='x-large')
    ax.set_title('KNeighborsRegressor prediction mean square error vs K', fontsize='x-large')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(fontsize='large', loc=1, ncol=1)
    plt.show()
knn('uniform')
knn('distance')
