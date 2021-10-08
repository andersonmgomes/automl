from sklearn.base import ClassifierMixin, RegressorMixin
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

class AutoRegression:
    def __init__(self, ds, y_colname, metric_order=METRICS_R2
                 , algorithms = [linear_model.LinearRegression(), svm.SVR(), tree.DecisionTreeRegressor()]
                 , unique_categoric_limit = 15) -> None:
        self.__unique_categoric_limit = unique_categoric_limit
        self.__ds_full = ds
        self.__ds_onlynums = self.__ds_full.select_dtypes(exclude=['object'])
        self.__X_full = self.__ds_onlynums.drop(columns=[y_colname])
        self.__Y_full = self.__ds_onlynums[[y_colname]]
        self.__results = None
        self.metric_order = metric_order
        self.algorithms = algorithms
        
    def addAlgorithm(self, algo):
        self.algorithms.append(algo)
        self.__results = None #cleaning the previous results
    
    def getBestModel(self):
        if self.getBestResult() is None:
            return None
        #else
        return self.getBestResult().model_instance

    def getBestResult(self):
        if len(self.getResults()) == 0:
            return None
        #else
        return self.getResults().iloc[0]
    
    def getResults(self, buffer=True):
        if buffer and self.__results is not None:
            return self.__results
        #else to get results
        #[algorithm, x_cols, mae, r2, mse, model]
        self.__results = pd.DataFrame(columns=['algorithm', 'features', METRICS_MAE, METRICS_R2, METRICS_MSE, 'model_instance'])

        y_is_cat = self.YisCategorical()
        y_is_num = not y_is_cat
        for algo in self.algorithms:
            for col_tuple in all_subsets(self.__X_full.columns):
                if ((len(col_tuple) == 0) #empty subsets
                    or (y_is_cat and isinstance(algo, RegressorMixin)) #Y is incompatible with algorithm        
                    or (y_is_num and isinstance(algo, ClassifierMixin))#Y is incompatible with algorithm
                    ):
                    continue
                #else: all right
                self.__results.loc[len(self.__results)] = self.__score_dataset(algo, col_tuple)
        
        self.__results.set_index(['algorithm', 'features'])
        self.__results.sort_values(by=METRICS_R2, ascending=False, inplace=True)
                            
        return self.__results           
    
    def YisCategorical(self) -> bool:
        if (isinstance(self.__Y_full.iloc[0,0], float)
            or (len(self.__Y_full.unique()) > self.__unique_categoric_limit)):
            return False
        return True    
    
    def YisContinuous(self) -> bool:
        return not self.YisCategorical()
                   
    def __score_dataset(self, algorithm, x_cols):
        X = self.__ds_onlynums[list(x_cols)]
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
        
        return np.array([str(algorithm).replace('()',''), x_cols, mae, r2, mse, model], dtype=object)

#util methods
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

from ds_utils import getDSPriceHousing, getDSFuelConsumptionCo2
autoreg_house = AutoRegression(getDSPriceHousing(), 'Price')
                        # , metric_order=METRICS_R2, algorithms = [svm.SVC()])
autoreg_house.addAlgorithm(svm.SVC())
print(autoreg_house.getResults().head())
print(autoreg_house.getBestResult())
print(autoreg_house.getBestModel())

#CO2EMISSIONS
'''
autoreg_co2 = AutoRegression(getDSFuelConsumptionCo2(), 'CO2EMISSIONS')
print(autoreg_co2.getResults().head())
print(autoreg_co2.getBestResult())
print(autoreg_co2.getBestModel())
'''
