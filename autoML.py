from re import escape
from numpy.core.numeric import Infinity, NaN
import pandas as pd 
#import modin.pandas as pd #https://modin.readthedocs.io/
from numpy.lib.function_base import append
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, utils
import numpy as np
from scipy.special import comb
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
import time
from memory_profiler import memory_usage
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import ray
import scipy.stats as sta
from joblib import Parallel, delayed
import warnings
import math
from bitarray import bitarray
from bitarray import util as bautil
from multiprocessing import Pool
from deap import algorithms, base, creator, tools
import random
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from scoop import futures
from datetime import datetime
import math
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

def features_corr_level_Y(i, X, y, threshold):
    #features engineering
    #testing correlation between X and Y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = sta.pearsonr(X, y)[0]
    if ( (corr != corr) #NaN value for correlation because constant feature
        or (abs(corr) < threshold)
        ):
        return None#x[i] below the threshold
    #else: feature ok with Y
    return i

def features_corr_level_X(i, X_0, X_i, threshold):
    #features engineering
    #testing correlation between X_0 and X_i
    for i in range(0, X_i.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = sta.pearsonr(X_0, X_i.iloc[:,i])[0]
        if ( (corr != corr) #NaN value for correlation because constant feature
            or (abs(corr) > threshold)
            ):
            return None#x[i] above the threshold
    #else: feature ok, no redundance
    return i

def __score_dataset(model, x_cols, X_train, X_valid, y_train, y_valid, X, y
                    , YisCategorical, __metrics_regression_list, __metrics_classification_list):
    
    X_train2 = X_train[list(x_cols)]
    X_valid2 = X_valid[list(x_cols)]
    
    if len(x_cols)==1:
        X_train2 = np.asanyarray(X_train2).reshape(-1, 1)
        X_valid2 = np.asanyarray(X_valid2).reshape(-1, 1)

    scoring_list = __metrics_regression_list
    if YisCategorical:
        scoring_list = __metrics_classification_list
    
    metrics_value_list = []
    
    for scor in scoring_list:
        metrics_value_list.append(np.mean(cross_val_score(model, X_valid2, y_valid, cv=5, scoring=scor)))
    
    result_list = metrics_value_list

    model.fit(X_train2, y_train)
    if YisCategorical:
        #confusion matrix
        result_list.append(confusion_matrix(y_valid, model.predict(X_valid2)))

    #model
    result_list.append(model)
    
    log_msg = '   *Model trained: ' + str(scoring_list[0]) 
    log_msg += ' = {:.5f}'.format(metrics_value_list[0]) 
    log_msg += ' | ' + str(len(x_cols)) + ' features' 
    log_msg += ' | ' + str(model)[:str(model).find('(')] 
    log_msg += ' | ' + str(x_cols)
            
    print(log_msg[:150].replace('\n',''))#show only the 150 first caracteres
    
    return np.array(result_list, dtype=object)

def evaluation(individual, n_bits_algos, selected_algos
               , X_bitmap, X_train, X_valid, y_train, y_valid, X, y, __results, main_metric
               , YisCategorical, __metrics_regression_list, __metrics_classification_list):
    def float2bigint(float_value):
        if math.isnan(float_value):
            float_value = -1
        return [int(float_value*100000)]
    
    #print(individual)
    
    algo = individual[-n_bits_algos:]
    algo = bautil.ba2int(bitarray(algo)) % len(selected_algos)
    
    algo = selected_algos[algo]
    
    col_tuple = individual[:len(X_bitmap)-n_bits_algos]
    col_tuple = tuple([X_train.columns[i] for i, c in enumerate(col_tuple) if c == 1])
    
    if len(col_tuple)==0:
        return float2bigint(-1)
    
    def is_ensemble(a):
        return isinstance(a, VotingClassifier) or isinstance(a, StackingClassifier)
    
    if is_ensemble(algo):
        #getting the top 3 best results group by algorithm
        best_estimators = []
        __results.sort_values(by=main_metric, ascending=False, inplace=True)
        for row in __results.iterrows():
            if len(best_estimators)==3:
                break
            candidate_algo = row[1]['algorithm']
            if ((candidate_algo not in best_estimators)
                and (not is_ensemble(candidate_algo))):
                best_estimators.append(candidate_algo)
        algo.estimators = list(zip([str(x) for x in best_estimators],best_estimators))

    #seeking for some previous result
    previous_result = __results[(__results['algorithm'] == algo) & (__results['features'].apply(str)==str(col_tuple))]
    if previous_result.shape[0]>0:
        return float2bigint(previous_result[main_metric])
    #else 
                
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mem_max, score_result = memory_usage(proc=(__score_dataset, (algo, col_tuple, X_train, X_valid
                                                                    , y_train, y_valid, X, y, YisCategorical
                                                                    , __metrics_regression_list, __metrics_classification_list))
                                            , max_usage=True
                                                , retval=True, include_children=True)
    __results.loc[len(__results)] = np.concatenate((np.array([algo, col_tuple
                                                                        , int(len(col_tuple))
                                                                        , (time.perf_counter() - t0)
                                                                        , mem_max], dtype=object)
                                                            , score_result))
    return float2bigint(score_result[0])

def gen_first_people(n_features, n_algos, n_bits_algos):
    first_people = []
    X_bitmap = bautil.int2ba(1, n_features)
    X_bitmap.setall(1)
    for i in range(n_algos):
        c_bitmap = []
        c_bitmap.extend(list(X_bitmap))
        c_bitmap.extend(list(bautil.int2ba(i, n_bits_algos)))
        first_people.append(c_bitmap)
    return first_people

def ga_toolbox(n_cols, n_bits_algos, selected_algos
               , X_bitmap, X_train, X_valid, y_train, y_valid, X, y, __results, main_metric
               , YisCategorical, __metrics_regression_list, __metrics_classification_list, pool=None):
    #genetics algorithm: creating types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    #multprocessing
    toolbox = base.Toolbox()
    if not(pool is None):
        toolbox.register("map", pool.map)     

    #genetics algorithm: initialization
    def initPopulation(pcls, ind_init):
        return pcls(ind_init(c) for c in gen_first_people(X_train.shape[1], len(selected_algos), n_bits_algos))
    toolbox.register("population", initPopulation, list, creator.Individual)
    
    #genetics algorithm: operators
    toolbox.register("evaluate", evaluation, n_bits_algos=n_bits_algos, selected_algos=selected_algos
            , X_bitmap=X_bitmap, X_train=X_train, X_valid=X_valid, y_train=y_train
            , y_valid=y_valid, X=X, y=y, __results=__results, main_metric=main_metric
            , YisCategorical=YisCategorical, __metrics_regression_list=__metrics_regression_list
            , __metrics_classification_list=__metrics_classification_list)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

class AutoML:
    ALGORITHMS = [
        #classifiers
        #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
        KNeighborsClassifier(),
        SVC(probability=True),
        GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_jobs=-1),
        MLPClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        MultinomialNB(), 
        GradientBoostingClassifier(),
        HistGradientBoostingClassifier(),
        VotingClassifier(estimators=[], n_jobs=-1),
        StackingClassifier(estimators=[], n_jobs=-1),
        #regressors        
        XGBRegressor(),
        XGBRFRegressor(),
        linear_model.LinearRegression(),
        svm.SVR(),
        tree.DecisionTreeRegressor(),
        neighbors.KNeighborsRegressor(),
        linear_model.LogisticRegression(),
        GradientBoostingRegressor(),    
    ]    
    
    def __init__(self, ds_source, y_colname = 'y'
                 , algorithms = ALGORITHMS
                 , unique_categoric_limit = 10 
                 , min_x_y_correlation_rate = 0.01
                 , n_features_threshold = 1
                 , pool = None
                 , ds_name = None
                 , ngen = 10) -> None:
        #ray.init(ignore_reinit_error=True)
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
        self.ds_name = ds_name
        self.ngen = ngen
        
        print('Original dataset dimensions:', ds_source.shape)
        #NaN values
        ds = ds_source.dropna()
        print('Dataset dimensions after drop NaN values:', ds.shape)
        
        #shuffle data to minimize bias tendency
        ds = ds.sample(frac=1)

        #setting Y
        self.y_colname = y_colname
        self.__y_full = ds[[self.y_colname]]
        self.__y_encoder = None
        self.y = np.asanyarray(self.__y_full).reshape(-1, 1).ravel()
        
        if self.YisCategorical():
            print('ML problem type: Classification')
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
        else:
            print('ML problem type: Regression')

        print('   Applied metrics:', self.__metrics_classification_list)
        
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
            print('One hot encoder columns:', hot_columns)
            self.__onehot_encoder.fit(self.X[hot_columns])
            
            hot_cols_names = []
            
            for i, name in enumerate(self.__onehot_encoder.feature_names_in_):
                for cat in self.__onehot_encoder.categories_[i]:
                    hot_cols_names.append(name + '_' + cat.lower().replace(' ','_'))
                    
            self.X = pd.concat([self.X.drop(hot_columns, axis=1)
                                , pd.DataFrame(self.__onehot_encoder.transform(self.X[hot_columns])
                                            , columns=hot_cols_names)], axis=1)
        
        #normalizing the variables
        print('Normalizing the variables...')
        self.scaler = preprocessing.MinMaxScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns) 

        #splitting dataset
        print('Splitting dataset...')
        self.X_train, self.X_valid, self.y_train, self.y_valid = self.__train_test_split()
        print('   X_train dimensions:', self.X_train.shape)
        self.y_train = np.asanyarray(self.y_train).reshape(-1, 1).ravel()
        self.y_valid = np.asanyarray(self.y_valid).reshape(-1, 1).ravel()
        
        #running feature engineering in paralel
        n_cols = self.X_train.shape[1]
        print('Features engineering - Testing correlation with Y...')
        considered_features = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(features_corr_level_Y)
                                  (i
                                   , self.X_train.iloc[:,i]
                                   , self.y_train
                                   , self.__min_x_y_correlation_rate)
                                  for i in range(0, n_cols))
        considered_features = [x for x in considered_features if x is not None]
        self.X_train = self.X_train.iloc[:,considered_features]
        self.X_valid = self.X_valid.iloc[:,considered_features]
        
        def n_features_2str():
            return "{:.2f}".format(100*(1-len(considered_features)/self.X.shape[1])) + "% (" + str(len(considered_features)) + " remained)"
        
        print('   Features engineering - Features reduction after correlation test with Y:'
              , n_features_2str())
        
        print('Features engineering - Testing redudance between features...')    
        
        n_cols = self.X_train.shape[1]
        considered_features = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(features_corr_level_X)
                                  (i
                                   ,self.X_train.iloc[:,i]
                                   , self.X_train.iloc[:,i+1:]
                                   , (1-self.__min_x_y_correlation_rate))
                                  for i in range(0, n_cols-1))

        considered_features = [x for x in considered_features if x is not None]
        self.X_train = self.X_train.iloc[:,considered_features]
        self.X_valid = self.X_valid.iloc[:,considered_features]
        
        print('   Features engineering - Features reduction after redudance test:'
              , n_features_2str())
        
        
    def clearResults(self):
        self.__results = None #cleaning the previous results
        
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
    
    def __train_models(self):
        pass
    
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

        selected_algos = []

        for algo in self.algorithms:
            if  ((y_is_cat and isinstance(algo, RegressorMixin)) #Y is incompatible with algorithm        
                 or (y_is_num and isinstance(algo, ClassifierMixin))#Y is incompatible with algorithm
            ):
                continue
            #else: all right
            selected_algos.append(algo)
        
        print('Selected algorithms:', [str(x)[:str(x).find('(')] for x in selected_algos])
        
        #setup the bitmap to genetic algorithm
        n_bits_algos = len(bautil.int2ba(len(selected_algos)-1))
        n_cols = self.X_train.shape[1] + n_bits_algos
        self.X_bitmap = bitarray(n_cols)
        self.X_bitmap.setall(1)

        #main metric column
        main_metric = self.__metrics_regression_list[0] #considering the first element the most important
        if y_is_cat:
            main_metric = self.__metrics_classification_list[0] #considering the first element the most important

        #calculating the size of population (features x algorithms)
        n_train_sets = 0
        for k in range(1, self.X_train.shape[1] + 1):
            n_train_sets += comb(self.X_train.shape[1] + 1, k, exact=False)
            if math.isinf(n_train_sets):
                break

        print('Nº of training possible combinations:'
              , n_train_sets*len(selected_algos)
              , '(' + str(n_train_sets),'features combinations,'
              , str(len(selected_algos)) +' algorithms)')

        if math.isinf(n_train_sets):
            n_train_sets = self.X_train.shape[1]

        n_train_sets = int(n_train_sets)        
        
        toolbox = ga_toolbox(n_cols, n_bits_algos, selected_algos
               , self.X_bitmap, self.X_train, self.X_valid, self.y_train, self.y_valid, self.X, self.y, self.__results, main_metric
               , self.YisCategorical(), self.__metrics_regression_list, self.__metrics_classification_list)
        #running the GA algorithm
        algorithms.eaSimple(toolbox.population(), toolbox, cxpb=0.8, mutpb=0.3, ngen=self.ngen, verbose=False)
        
        #preparing the results
        self.__results.sort_values(by=main_metric, ascending=False, inplace=True)
        self.__results = self.__results.rename_axis('train_order').reset_index()        

        #saving results in a csv file
        filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not(self.ds_name is None):
            filename += '_' + self.ds_name.upper()
        filename += '.csv'
        self.__results.drop('model_instance', axis=1).to_csv('results/' + filename, index=False)
        
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

#utilitary methods
#def all_subsets(ss, min_n = 1):
#    return list(chain(*map(lambda x: combinations(ss, x), range(min_n, len(ss)+1))))

from cf_matrix import make_confusion_matrix

def getConfusionMatrixHeatMap(cf_matrix, title='CF Matrix'):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['Zero', 'One']
    return make_confusion_matrix(cf_matrix, group_names=group_names, categories=categories, cmap='Blues', title=title);    

def testAutoMLByCSV(csv_path, y_colname):
    return testAutoML(pd.read_csv(csv_path), y_colname=y_colname)

import ds_utils as util

def testAutoML(ds, y_colname):
    automl = AutoML(ds, y_colname, min_x_y_correlation_rate=0.06)
    #automl.setAlgorithm(svm.SVC())
    if automl.YisCategorical():
        print(automl.getBestResult().confusion_matrix)
    
    df4print = automl.getResults()
    print(df4print.head())
    print(automl.getBestResult())
    if automl.getBestResult() is not None:
        print(automl.getBestResult()['features'])
    del(automl)

if __name__ == '__main__':
    pool = Pool(processes=6)
    automl = AutoML(util.getDSIris(), 'class'
                    , min_x_y_correlation_rate=0.01
                    , pool=pool
                    , ngen=10
                    , ds_name='iris_ENSEMBLE')
    print(automl.getBestResult())
