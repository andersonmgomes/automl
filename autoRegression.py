from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
import numpy as np
from itertools import chain, combinations
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
import pandas as pd

METRICS_R2 = 'R2'
METRICS_MAE = 'MAE'
METRICS_MSE = 'MSE'

class ModelResult:
    def __init__(self, algorithm, X_cols, order, model, metrics_detail) -> None:
        self.algorithm = algorithm
        self.X_cols = X_cols
        self.order = order
        self.model = model
        self.metrics_detail = metrics_detail
    
    def __str__(self):
        return str(self.algorithm) + ' -> ' + str(self.X_cols) + ' -> ' + str(self.metrics_detail)
    
    def __repr__(self):
        return str(self)
    
    def getMetric(self, metric):
        return self.metrics_detail[metric]

    def getR2(self):
        return self.getMetric(METRICS_R2)

    def getMAE(self):
        return self.getMetric(METRICS_MAE)

    def getMSE(self):
        return self.getMetric(METRICS_MSE)

class AutoRegression:
    def __init__(self, ds, y_colname, metric_order=METRICS_R2
                 , algorithms = [linear_model.LinearRegression(), svm.SVR(), tree.DecisionTreeRegressor()]) -> None:
        self.__ds_full = ds
        self.__ds_onlynums = self.__ds_full.select_dtypes(exclude=['object'])
        self.__X_full = self.__ds_onlynums.drop(columns=[y_colname])
        self.__Y_full = self.__ds_onlynums[[y_colname]]
        self.__results = None
        self.metric_order = metric_order
        self.algorithms = algorithms
        
    def getBestModel(self):
        return self.getResults().iloc[0]
    
    def getResults(self, buffer=True):
        if buffer and self.__results is not None:
            return self.__results
        #else to get results
        #[algorithm, x_cols, mae, r2, mse, model]
        self.__results = pd.DataFrame(columns=['algorithm', 'features', METRICS_MAE, METRICS_R2, METRICS_MSE, 'model_instance'])
        
        for algo in self.algorithms:
            for col_tuple in all_subsets(self.__X_full.columns):
                if len(col_tuple) == 0:
                    continue
                col_list = list(col_tuple)
                self.__results.loc[len(self.__results)] = self.__score_dataset(algo, col_list)
        
        self.__results.set_index(['algorithm', 'features'])
        self.__results.sort_values(by=METRICS_R2, ascending=False, inplace=True)
                            
        return self.__results           
        
    def __score_dataset(self, algorithm, x_cols):
        X = self.__ds_onlynums[x_cols]
        y = self.__Y_full
        
        #normalizing the variables
        min_max_scaler = preprocessing.MinMaxScaler()
        X_normal = min_max_scaler.fit_transform(X)
        y_normal = min_max_scaler.fit_transform(y)
        
        X_train, X_valid, y_train, y_valid = train_test_split(X_normal, y_normal, train_size=0.8, test_size=0.2, random_state=0)
        
        model = algorithm

        X_train2 = X_train
        X_valid2 = X_valid
        y_train2 = y_train
        y_valid2 = y_valid
        
        if len(x_cols)==1:
            X_train2 = np.asanyarray(X_train).reshape(-1, 1)
            X_valid2 = np.asanyarray(X_valid).reshape(-1, 1)
            y_train2 = np.asanyarray(y_train).reshape(-1, 1)
            y_valid2 = np.asanyarray(y_valid).reshape(-1, 1)

        model.fit(X_train2, y_train2.ravel())
        preds = model.predict(X_valid2)
        
        mae = mean_absolute_error(y_valid2, preds)
        r2 = r2_score(y_valid2, preds)
        mse = mean_squared_error(y_valid2, preds)
        
        return [str(algorithm), str(x_cols), mae, r2, mse, model]

#util methods
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


from ds_utils import getDSPriceHousing
autoreg = AutoRegression(getDSPriceHousing(), 'Price')
                         #, metric_order=METRICS_R2, algorithms = [svm.SVR()])
model = autoreg.getBestModel()
print(autoreg.getResults().head())


