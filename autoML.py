from numpy.lib.function_base import append
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, utils
import numpy as np
from itertools import chain, combinations
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
import pandas as pd 
import time
from memory_profiler import memory_usage
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

class AutoML:
    def __init__(self, ds_source, y_colname = 'y'
                 , algorithms = [linear_model.LinearRegression(), svm.SVR(), tree.DecisionTreeRegressor()
                                 , neighbors.KNeighborsRegressor(), linear_model.LogisticRegression()
                                 , svm.SVC(probability=True), neighbors.KNeighborsClassifier(), tree.DecisionTreeClassifier()]
                 , unique_categoric_limit = 10 
                 , min_x_y_correlation_rate = 0.001
                 , n_features_threshold = 0.999
                 ) -> None:
        #initializing variables
        self.__results = None
        self.algorithms = algorithms
        self.__unique_categoric_limit = unique_categoric_limit
        self.__metrics_regression_list = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        self.__metrics_classification_list = ['f1', 'accuracy', 'roc_auc']
        #metrics reference: https://scikit-learn.org/stable/modules/model_evaluation.html
        self.__min_x_y_correlation_rate = min_x_y_correlation_rate #TODO: #1 MIN_X_Y_CORRELATION_RATE: define this value dynamically
        self.__n_features_threshold = n_features_threshold #TODO: N_FEATURES_THRESHOLD: define this value dynamically
        self.__RANDOM_STATE = 1102
        
        #NaN values
        ds = ds_source.dropna()
        
        #setting Y
        self.y_colname = y_colname
        self.__y_full = ds[[self.y_colname]]
        self.__y_encoder = None
        self.y = np.asanyarray(self.__y_full).reshape(-1, 1).ravel()
        
        if self.YisCategorical():
            #encoding
            self.__y_encoder = OrdinalEncoder(dtype=np.int)
            self.__y_full = pd.DataFrame(self.__y_encoder.fit_transform(self.__y_full), columns=[self.y_colname])
            if len(self.__y_full[self.y_colname].unique()) > 2: #multclass 
                #adjusting the F1 score and ROC_AUC for multclass target
                for i, m in enumerate(self.__metrics_classification_list):
                    if m == 'f1':
                        self.__metrics_classification_list[i] = 'f1_weighted'
                    elif m == 'roc_auc':
                        self.__metrics_classification_list[i] = 'roc_auc_ovr_weighted'
                
        #setting X
        self.X = ds.drop(self.y_colname, axis=1)
        self.__onehot_encoder = OneHotEncoder(sparse=False, dtype=np.int)

        hot_columns = []
        str_columns = []
        for i, col in enumerate(self.X.columns):
            if self.X.dtypes[i] == object: 
                if len(self.X[col].unique()) <= self.__unique_categoric_limit:
                    hot_columns.append(col)
                else:
                   str_columns.append(col)
        
        if len(str_columns) > 0:
            self.X = self.X.drop(str_columns, axis=1)
            
        if len(hot_columns) > 0:
            self.__onehot_encoder.fit(self.X[hot_columns])
            
            hot_cols_names = []
            
            for i, name in enumerate(self.__onehot_encoder.feature_names_in_):
                for cat in self.__onehot_encoder.categories_[i]:
                    hot_cols_names.append(name + '_' + cat.lower().replace(' ','_'))
                    
            self.X = pd.concat([self.X.drop(hot_columns, axis=1)
                                , pd.DataFrame(self.__onehot_encoder.transform(self.X[hot_columns])
                                            , columns=hot_cols_names)], axis=1)
        
        #splitting dataset
        self.X_train, self.X_valid, self.y_train, self.y_valid = self.__train_test_split()
        #normalizing the variables
        self.scaler = preprocessing.MinMaxScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X.columns) #fit only with X_train
        self.X_valid = pd.DataFrame(self.scaler.transform(self.X_valid), columns=self.X.columns)
        self.y_train = np.asanyarray(self.y_train).reshape(-1, 1).ravel()
        self.y_valid = np.asanyarray(self.y_valid).reshape(-1, 1).ravel()
        
        
    def clearResults(self):
        self.__results = None #cleaning the previous results
        
    def processAllFeatureCombinations(self):
        self.setNFeaturesThreshold(0)
        
    def setNFeaturesThreshold(self, threshold):
        self.__n_features_threshold = threshold
        self.clearResults()

    def setMinXYcorrRate(self, rate):
        self.__min_x_y_correlation_rate = rate
        self.clearResults()
        
    def setAlgorithm(self, algo):
        self.algorithms.clear()
        self.algorithms.append(algo)
        self.clearResults()
    
    def getBestModel(self):
        if self.getBestResult(True) is None:
            return None
        #else
        return self.getBestResult(True).model_instance

    def getBestConfusionMatrix(self):
        getConfusionMatrixHeatMap(self.getBestResult().confusion_matrix
                                  , title=(str(self.getBestResult().algorithm)
                                           + ' (' + str(self.getBestResult().n_features) +' features)'))
                
    def getBestResult(self, resultWithModel=False):
        if len(self.getResults(resultWithModel)) == 0:
            return None
        #else
        return self.getResults(resultWithModel).iloc[0]
    
    def getResults(self, resultWithModel=False, buffer=True):
        if buffer and self.__results is not None:
            if resultWithModel:                   
                return self.__results
            #else
            return self.__results.drop('model_instance', axis=1)
                   
        #else to get results
        #dataframe format: [algorithm, features, n_features, train_time, mem_max, [specific metrics], model_instance]
        columns_list = ['algorithm', 'features', 'n_features', 'train_time', 'mem_max']
        if self.YisCategorical():
            columns_list.extend(self.__metrics_classification_list)
            columns_list.append('confusion_matrix')
        else:
            columns_list.extend(self.__metrics_regression_list)
        columns_list.append('model_instance')
        
        self.__results = pd.DataFrame(columns=columns_list)
        del(columns_list)
        
        y_is_cat = self.YisCategorical()
        y_is_num = not y_is_cat
        
        #features engineering
        df_train_full = pd.concat([pd.DataFrame(self.X_train, columns=self.X.columns)
                                   , pd.DataFrame(self.y_train, columns=[self.y_colname])], axis=1)
 
        features_corr = df_train_full.corr()
        #print(features_corr)
        features_candidates = []
        #testing min correlation rate with Y
        for feat_name,corr_value in features_corr[self.y_colname].items():
            if ((abs(corr_value) > self.__min_x_y_correlation_rate)
                and (feat_name != self.y_colname)):
                features_candidates.append(feat_name)
        
        considered_features = []
        features_corr = df_train_full[features_candidates].corr()
        #print(features_corr)
        #testing redudance between features
        for i in range(0, len(features_candidates)):
            no_redudance = True
            for j in range(i+1, len(features_candidates)):
                if ((abs(features_corr.iloc[i][j]) > (1-self.__min_x_y_correlation_rate))):
                    no_redudance = False
                    break
            if no_redudance:
                considered_features.append(features_candidates[i])
            
        subsets = all_subsets(considered_features)
        del(features_corr)
        del(features_candidates)
        
        for algo in self.algorithms:
            if  ((y_is_cat and isinstance(algo, RegressorMixin)) #Y is incompatible with algorithm        
                 or (y_is_num and isinstance(algo, ClassifierMixin))#Y is incompatible with algorithm
            ):
                continue
            #else: all right
            algo_id = str(algo).replace('()','')
            #print('*** Testing algo ' + algo_id + '...')
            for col_tuple in subsets:
                if ((len(col_tuple) == 0)#empty subsets
                    or ((len(col_tuple)/len(considered_features)) < self.__n_features_threshold)
                    ): 
                    continue
                #else: all right
                #print('cols:' + str(col_tuple) + '...')
                t0 = time.perf_counter()
                mem_max, score_result = memory_usage(proc=(self.__score_dataset, (algo, col_tuple)), max_usage=True
                                                     , retval=True, include_children=True)
                self.__results.loc[len(self.__results)] = np.concatenate((np.array([algo_id, col_tuple
                                                                                    , int(len(col_tuple))
                                                                                    , (time.perf_counter() - t0)
                                                                                    , mem_max], dtype=object)
                                                                        , score_result))
        
        sortby = self.__metrics_regression_list[0] #considering the first element the most important
        if y_is_cat:
            sortby = self.__metrics_classification_list[0] #considering the first element the most important
            
        self.__results.sort_values(by=sortby, ascending=False, inplace=True)
        self.__results.reset_index(inplace=True, drop=True)        
        
        if resultWithModel:                   
            return self.__results
        #else
        return self.__results.drop('model_instance', axis=1)
    
    def YisCategorical(self) -> bool:
        y_type = type(self.__y_full.iloc[0,0])
        
        if (y_type == np.bool_
            or y_type == np.str_):
            return True
        #else
        if ((y_type == np.float_)
            or (len(self.__y_full[self.y_colname].unique()) > self.__unique_categoric_limit)):
            return False
        #else
        return True    
    
    def YisContinuous(self) -> bool:
        return not self.YisCategorical()
                   
    def __train_test_split(self):
        y = self.__y_full

        stratify=None
        if self.YisCategorical():
            stratify = y
            
        return train_test_split(self.X, y, train_size=0.8, test_size=0.2, random_state=self.__RANDOM_STATE, stratify=stratify)
        
    def __score_dataset(self, model, x_cols):
        
        X_train2 = self.X_train[list(x_cols)]
        X_valid2 = self.X_valid[list(x_cols)]
        
        if len(x_cols)==1:
            X_train2 = np.asanyarray(X_train2).reshape(-1, 1)
            X_valid2 = np.asanyarray(X_valid2).reshape(-1, 1)

        scoring_list = self.__metrics_regression_list
        if self.YisCategorical():
            scoring_list = self.__metrics_classification_list
        
        metrics_value_list = []
        
        for scor in scoring_list:
            metrics_value_list.append(np.mean(cross_val_score(model, self.X[list(x_cols)], self.y, cv=5, scoring=scor)))
        
        result_list = metrics_value_list

        model.fit(X_train2, self.y_train)
        if self.YisCategorical():
            #confusion matrix
            result_list.append(confusion_matrix(self.y_valid, model.predict(X_valid2)))

        #model
        result_list.append(model)
        
        return np.array(result_list, dtype=object)

#utilitary methods
def all_subsets(ss):
    return list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))

from cf_matrix import make_confusion_matrix

def getConfusionMatrixHeatMap(cf_matrix, title='CF Matrix'):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['Zero', 'One']
    return make_confusion_matrix(cf_matrix, group_names=group_names, categories=categories, cmap='Blues', title=title);    

