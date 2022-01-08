import math
import time
from typing import OrderedDict
import warnings
from datetime import datetime
from multiprocessing import Pool
import inspect
import numpy as np
import scipy.stats as sta
from bitarray import bitarray
from bitarray import util as bautil
from deap import algorithms, base, creator, tools
from joblib import Parallel, delayed
from memory_profiler import memory_usage
import pandas # as pd
from tqdm import tqdm
from modin.config import ProgressBar
import modin.pandas as pd #https://modin.readthedocs.io/
import ray
from scipy.special import comb
from sklearn import linear_model, neighbors, preprocessing, svm, tree, utils
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import confusion_matrix, get_scorer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
#from tpot import TPOTClassifier
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor
import os
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

def flushResults(automl_obj):
    #saving results in a csv file
    filename = automl_obj.start_time.strftime("%Y%m%d_%H%M%S")
    if not(automl_obj.ds_name is None):
        filename += '_' + automl_obj.ds_name.upper()
    filename += '.csv'
    filedir = './results'
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    automl_obj.results.to_csv(os.path.join(filedir, filename), index=False)
    
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

def float2bigint(float_value):
    if math.isnan(float_value):
        float_value = -1
    return [int(float_value*100000)]

def is_Voting_or_Stacking(a):
    return ((a == VotingClassifier) or (a == StackingClassifier)
            or isinstance(a, VotingClassifier) or isinstance(a, StackingClassifier))
    
def evaluation(individual, automl_obj):
    #print(individual)
    algo_instance = individual[-automl_obj.n_bits_algos:]
    # in this point the variable algo_instance is a bitarray
    algo_instance = bautil.ba2int(bitarray(algo_instance)) % len(automl_obj.selected_algos)
    # in this point the variable algo_instance is an integer
    algo_instance =  automl_obj.selected_algos[algo_instance]
    # in this point the variable algo_instance is a Class

    if is_Voting_or_Stacking(algo_instance):
        algo_instance = algo_instance(estimators=[])
    else:
        algo_instance = algo_instance()
    # in this point the variable algo_instance is a object with the default parameters
    
    col_tuple = individual[:len(automl_obj.X_bitmap)-automl_obj.n_bits_algos]
    col_tuple = tuple([automl_obj.X_train.columns[i] for i, c in enumerate(col_tuple) if c == 1])
    
    if len(col_tuple)==0:
        return float2bigint(-1)

    #seeking for some previous result
    previous_result = automl_obj.results[(automl_obj.results['algorithm'] == algo_instance.__class__) 
                                        & (automl_obj.results['features'] == col_tuple)]
    if previous_result.shape[0]>0:
        return float2bigint(previous_result.iloc[0][automl_obj.main_metric])
    
    if is_Voting_or_Stacking(algo_instance):
        #getting the top 3 best results group by algorithm
        best_estimators = []
        automl_obj.results.sort_values(by=automl_obj.main_metric, ascending=False, inplace=True)
        for row in automl_obj.results.iterrows():
            if len(best_estimators)==3:
                break
            
            if is_Voting_or_Stacking(row[1]['algorithm']):
                continue
            
            candidate_algo = row[1]['algorithm']()
            candidate_algo.set_params(**row[1]['params'])
            
            if candidate_algo.__class__ not in [x.__class__ for x in best_estimators]:
                best_estimators.append(candidate_algo)
        
        if len(best_estimators)<2:
            return float2bigint(-1)
        #else
        algo_instance.estimators = list(zip(['e'+str(i) for i in range(1,len(best_estimators)+1)],best_estimators))
        
    X_train2 = automl_obj.X_train[list(col_tuple)]._to_pandas()
    X_test2 = automl_obj.X_test[list(col_tuple)]._to_pandas()
    
    if len(col_tuple)==1:
        X_train2 = np.asanyarray(X_train2).reshape(-1, 1)
        X_test2 = np.asanyarray(X_test2).reshape(-1, 1)

    scoring_list = automl_obj.getMetrics()
    
    #tunning parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if automl_obj.grid_search:
            opt = GridSearchCV(estimator=algo_instance
                               , param_grid=automl_obj.algorithms[algo_instance.__class__]
                               , scoring=automl_obj.main_metric
                               , cv=5
                               , verbose=0, n_jobs=-1
                               )
        else:
            opt = BayesSearchCV(estimator=algo_instance
                                , search_spaces=automl_obj.algorithms[algo_instance.__class__]
                                , scoring=automl_obj.main_metric
                                , n_iter=automl_obj.n_inter_bayessearch, cv=5
                                , verbose=0, n_jobs=-1, random_state=automl_obj.RANDOM_STATE
                                )
        opt.fit(X_train2, automl_obj.y_train)

    def fit_score():
        estimator = algo_instance.set_params(**params)
        row = {'algorithm': estimator.__class__
               , 'params': params
               , 'features': col_tuple
               , 'n_features': len(col_tuple)
               }

        t0 = time.perf_counter()
        
        estimator.fit(X_train2, automl_obj.y_train)
        row['train_time'] = time.perf_counter() - t0 #train_time
        
        t0 = time.perf_counter()
        
        for scor_str in scoring_list:
            row[scor_str] = (get_scorer(scor_str)(estimator, X_test2, automl_obj.y_test))
        
        row['predict_time'] = (time.perf_counter() - t0)/len(automl_obj.y_test) #predict_time, considering one sample at a time
        
        if automl_obj.YisCategorical():
            #confusion matrix
            row['confusion_matrix'] = confusion_matrix(automl_obj.y_test, estimator.predict(X_test2), labels=automl_obj.y_classes)
        
        if (is_Voting_or_Stacking(algo_instance)
            and len(algo_instance.estimators)>0):
            #incluing the estimators in the row
            row['params'].update({'estimators': estimator.estimators})
            
        return row
        
    best_score = -1.0
    #dataframe format: ['algorithm', 'params', 'features', 'n_features', 'train_time', 'predict_time', 'mem_max', <metrics>]
    for params in opt.cv_results_['params']: 
        if isinstance(params, OrderedDict):
            #changing the type to dict (when using BayesSearchCV)
            params = dict(params)
            
        #seeking for some previous result
        previous_result = automl_obj.results[(automl_obj.results['algorithm'] == algo_instance.__class__) 
                                             & ((automl_obj.results['params'] == params) | is_Voting_or_Stacking(algo_instance))
                                            & (automl_obj.results['features'] == col_tuple)]
        if previous_result.shape[0]>0:
            continue
        #else 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mem_max, row_result = memory_usage(proc=(fit_score)
                                                , max_usage=True
                                                , retval=True
                                                , include_children=True)
        row_result['mem_max'] = mem_max

        automl_obj.results.loc[len(automl_obj.results)] = row_result

        if row_result[automl_obj.main_metric] > best_score:
            best_score = row_result[automl_obj.main_metric]
            
        log_msg = '   *Model trained: ' + str(scoring_list[0]) 
        log_msg += ' = {:.5f}'.format(row_result[automl_obj.main_metric]) 
        log_msg += ' | ' + str(algo_instance)[:str(algo_instance).find('(')] 
        log_msg += ' | ' + str(len(col_tuple)) + ' features'
        params_str = str(params)
        params_str = params_str.replace("'n_jobs': -1,","").replace("  ", " ").replace("{ ", "{").replace(" }", "}")
        log_msg += ' | ' + params_str

        print(log_msg[:150].replace('\n',''))#show only the 150 first caracteres
 
    flushResults(automl_obj)
    return float2bigint(best_score) #main metric

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

def ga_toolbox(automl_obj):
    #genetics algorithm: creating types
    with warnings.catch_warnings(): #TODO: solve RuntimeWarning: A class named 'FitnessMax' has already been created...
        warnings.simplefilter("ignore")
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    #multiprocessing
    toolbox = base.Toolbox()
    if not(automl_obj.pool is None):
        #toolbox.register("map", pool.map) #TODO: check if it works
        pass

    #genetics algorithm: initialization
    def initPopulation(pcls, ind_init):
        return pcls(ind_init(c) for c in gen_first_people(automl_obj.X_train.shape[1], len(automl_obj.selected_algos), automl_obj.n_bits_algos))
    toolbox.register("population", initPopulation, list, creator.Individual)
    
    #genetics algorithm: operators
    toolbox.register("evaluate", evaluation, automl_obj=automl_obj)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

class AutoML:
    ALGORITHMS = {
        #classifiers
        #https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
        KNeighborsClassifier: 
            {"n_neighbors": [3,5,7,9,11,13,15,17],
             "p": [2, 3],
             "n_jobs": [-1],
             },
        SVC:
            {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
             "gamma": ["auto", "scale"],
             "class_weight": ["balanced", None],
             "probability": [True]},
        GaussianProcessClassifier:{
            "copy_X_train": [False],
            "warm_start": [True, False],
            "n_jobs": [-1],},
        DecisionTreeClassifier:{
            "criterion": ["gini", "entropy"],
            },
        RandomForestClassifier:{
            "n_estimators": [120,300,500,800,1200],
            "max_depth": [None, 5, 8, 15, 20, 25, 30],
            "min_samples_split": [2, 5, 10, 15, 100],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": [None, "sqrt", "log2"],
            "n_jobs": [-1],
            },
        MLPClassifier:{
            "learning_rate": ['constant', 'invscaling', 'adaptive'], 
            'momentum' : [0.1, 0.5, 0.9], 
            },
        AdaBoostClassifier:{
            "algorithm": ["SAMME", "SAMME.R"],
            },
        GaussianNB:{
            "priors": [None],
            },
        QuadraticDiscriminantAnalysis:{
            "priors": [None],
            },
        XGBClassifier:{
            "eta": [0.01, 0.015, 0.025, 0.05, 0.1],
            "gamma": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": [3, 5, 7, 9, 12, 15, 17, 25],
            "min_child_weight": [1, 3, 5, 7],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1],
            "lambda": [0.01,0.1,1],
            "alpha": [0, 0.1, 0.5, 1],
            "use_label_encoder": [False],
            "eval_metric": ['mlogloss'],
            },
        MultinomialNB:{
            "fit_prior": [True, False],
            }, 
        GradientBoostingClassifier:{
            "loss": ["deviance"],
            },
        HistGradientBoostingClassifier:{
            "warm_start": [True, False],
            },
        #TPOTClassifier(verbosity=0, n_jobs=-1):{},
        linear_model.LinearRegression:{
            "fit_intercept": [True, False],
            "n_jobs": [-1],
            },
        linear_model.LogisticRegression:{
            "C": [0.001, 0.01, 0.1, 1, 10,
                  100, 1000],
            "n_jobs": [-1],
            },
        VotingClassifier:{
            "voting": ["soft"],
            "n_jobs": [-1],
            },
        StackingClassifier:{
            "stack_method": ["auto"],
            "n_jobs": [-1],
            },
        #regressors        
        XGBRegressor:{},
        XGBRFRegressor:{},
        svm.SVR:{},
        tree.DecisionTreeRegressor:{},
        neighbors.KNeighborsRegressor:{},
        GradientBoostingRegressor:{},    
    }    
    
    def __init__(self, ds_source, y_colname = 'y'
                 , algorithms = ALGORITHMS
                 , unique_categoric_limit = 10 
                 , min_x_y_correlation_rate = 0.01
                 , pool = None
                 , ds_name = None
                 , ngen = 10
                 , metrics = None
                 , features_engineering = True
                 , grid_search = False
                 , n_inter_bayessearch = 30
                 , ds_source_header=None
                 , ds_source_header_names=None
                 ) -> None:
        self.start_time = datetime.now()
        ProgressBar.enable()
        ray.init(ignore_reinit_error=True)
        #initializing variables
        self.results = None
        self.algorithms = algorithms
        self.__unique_categoric_limit = unique_categoric_limit
        self.metrics_regression_list = metrics
        self.metrics_classification_list = metrics
        if metrics is None:
            self.metrics_regression_list = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            self.metrics_classification_list = ['f1', 'accuracy', 'roc_auc']
        #metrics reference: https://scikit-learn.org/stable/modules/model_evaluation.html
        self.__min_x_y_correlation_rate = min_x_y_correlation_rate #TODO: #1 MIN_X_Y_CORRELATION_RATE: define this value dynamically
        self.RANDOM_STATE = 1102
        self.ds_name = ds_name
        self.ngen = ngen
        self.pool = pool
        self.grid_search = grid_search
        self.n_inter_bayessearch = n_inter_bayessearch
        
        if type(ds_source) == str and ds_source.endswith('.csv'):
            ds_source = pd.read_csv(ds_source, header=ds_source_header, names=ds_source_header_names)
        
        print('Original dataset dimensions:', ds_source.shape)
        #NaN values
        ds = ds_source.dropna()
        print('Dataset dimensions after drop NaN values:', ds.shape)
        
        #shuffle data to minimize bias tendency
        ds = ds.sample(frac=1)

        #setting Y
        self.y_colname = y_colname
        self.y_full = ds[[self.y_colname]]
        self.__y_encoder = None
        self.y = np.asanyarray(self.y_full).reshape(-1, 1).ravel()
        self.y_is_binary = False
        self.y_classes = None
        
        if self.YisCategorical():
            print('ML problem type: Classification')
            #encoding
            self.__y_encoder = OrdinalEncoder(dtype=int)
            self.y_full = pd.DataFrame(self.__y_encoder.fit_transform(self.y_full), columns=[self.y_colname])

            self.y_classes = np.sort(self.y_full[self.y_colname].unique())
            self.y_is_binary = len(self.y_classes) == 2

            if not self.y_is_binary: #multiclass 
                #adjusting the metrics for multiclass target
                for i, m in enumerate(self.metrics_classification_list):
                    if m == 'f1':
                        self.metrics_classification_list[i] = 'f1_weighted'
                    elif m == 'roc_auc':
                        self.metrics_classification_list[i] = 'roc_auc_ovr_weighted'
                    elif m == 'accuracy':
                        self.metrics_classification_list[i] = 'balanced_accuracy'
                    elif m == 'recall':
                        self.metrics_classification_list[i] = 'recall_weighted'
                    elif m == 'precision':
                        self.metrics_classification_list[i] = 'precision_weighted'

        else:
            print('ML problem type: Regression')

        print('   Applied metrics:', self.metrics_classification_list)
        
        #setting X
        self.X = ds.drop(self.y_colname, axis=1)
        self.__onehot_encoder = OneHotEncoder(sparse=False, dtype=int)

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
        self.X_train, self.X_test, self.y_train, self.y_test = self.__train_test_split()
        print('   X_train dimensions:', self.X_train.shape)
        self.y_train = np.asanyarray(self.y_train).reshape(-1, 1).ravel()
        self.y_test = np.asanyarray(self.y_test).reshape(-1, 1).ravel()
        
        #running feature engineering in parallel
        if features_engineering:
            n_cols = self.X_train.shape[1]
            print('Features engineering - Testing correlation with Y...')
            considered_features = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(features_corr_level_Y)
                                    (i
                                    , self.X_train.iloc[:,i]._to_pandas()
                                    , self.y_train
                                    , self.__min_x_y_correlation_rate)
                                    for i in range(0, n_cols))
            considered_features = [x for x in considered_features if x is not None]
            self.X_train = self.X_train.iloc[:,considered_features]
            self.X_test = self.X_test.iloc[:,considered_features]
            
            def n_features_2str():
                return "{:.2f}".format(100*(1-len(considered_features)/self.X.shape[1])) + "% (" + str(len(considered_features)) + " remained)"
            
            print('   Features engineering - Features reduction after correlation test with Y:'
                , n_features_2str())
            
            print('Features engineering - Testing redudance between features...')    
            
            n_cols = self.X_train.shape[1]
            considered_features = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(features_corr_level_X)
                                    (i
                                    , self.X_train.iloc[:,i]._to_pandas()
                                    , self.X_train.iloc[:,i+1:]._to_pandas()
                                    , (1-self.__min_x_y_correlation_rate))
                                    for i in range(0, n_cols-1))

            considered_features = [x for x in considered_features if x is not None]
            self.X_train = self.X_train.iloc[:,considered_features]
            self.X_test = self.X_test.iloc[:,considered_features]
            
            print('   Features engineering - Features reduction after redudance test:'
                , n_features_2str())

    def clearResults(self):
        self.results = None #cleaning the previous results
        
    def getBestModel(self):
        if self.getBestResult() is None:
            return None
        #else
        return self.getBestResult().estimator

    def getBestConfusionMatrix(self):
        return self.getConfusionMatrix(0)
    
    def getConfusionMatrix(self, result_index):
        if self.YisContinuous():
            return None
        #else: classification problem
        result = self.__fit().iloc[result_index]
        title = self.__class2str(result.algorithm)
        title += ' (' + str(result.n_features) +' features)'
        title += '\n' + str(result.params)
        title = title.replace("'n_jobs': -1,","").replace("  ", " ").replace("{ ", "{").replace(" }", "}")
        
        categories = self.y_classes#['Zero', 'One']
        if self.__y_encoder is not None:
            categories = self.__y_encoder.categories_[0]
            
        group_names = [] #['True Neg','False Pos','False Neg','True Pos']
        for c in categories:
            group_names.append('True_' + str(c)) 
            group_names.append('False_' + str(c))
            
        custom_metrics = dict(result.loc[self.getMetrics()])
        
        return make_confusion_matrix(result.confusion_matrix
                                     , group_names=group_names
                                     , categories=categories
                                     , cmap='Blues'
                                     , title=title
                                     , custom_metrics=custom_metrics);    
                
    def getBestResult(self):
        if len(self.__fit()) == 0:
            return None
        #else
        return self.__fit().iloc[0]
    
    def getResults(self, buffer=True):
        results_df = self.__fit(buffer).drop(['confusion_matrix', 'n_features'], axis=1)
        results_df['algorithm'] = results_df['algorithm'].apply(lambda x: self.__class2str(x))
        results_df['features'] = results_df['features'].apply(lambda x: str(len(x)) + ': ' + str(x).replace('(','').replace(')',''))
        return results_df

    def __class2str(self, cls):
        cls_str = str(cls)
        return cls_str[cls_str.rfind('.')+1:cls_str.rfind("'")] 
            
    def __fit(self, buffer=True):
        if buffer and self.results is not None:
            return self.results
                   
        #else to get results
        t0 = time.perf_counter()

        #dataframe format: ['algorithm', 'params', 'features', 'n_features', 'train_time', 'predict_time', 'mem_max', <metrics>]
        columns_list = ['algorithm', 'params', 'features', 'n_features', 'train_time', 'predict_time', 'mem_max']
        
        columns_list.extend(self.getMetrics())
        if self.YisCategorical():
            columns_list.append('confusion_matrix')
        
        self.results = pandas.DataFrame(columns=columns_list)
        del(columns_list)
        
        y_is_cat = self.YisCategorical()
        y_is_num = not y_is_cat

        self.selected_algos = []
        
        def is_in_class_tree(cls1, cls2):
            #vide https://docs.python.org/3/library/inspect.html
            return cls1 in inspect.getmro(cls2)
        
        for algo in self.algorithms.keys():
            if  ((y_is_cat and is_in_class_tree(RegressorMixin, algo)) #Y is incompatible with algorithm        
                 or (y_is_num and is_in_class_tree(ClassifierMixin, algo))#Y is incompatible with algorithm
            ):
                continue
            #else: all right
            self.selected_algos.append(algo)
        
        print('Selected algorithms:', [self.__class2str(x) for x in self.selected_algos])
        
        #setup the bitmap to genetic algorithm
        self.n_bits_algos = len(bautil.int2ba(len(self.selected_algos)-1))#TODO: analyze de number of bits to use
        self.n_cols = self.X_train.shape[1] + self.n_bits_algos
        self.X_bitmap = bitarray(self.n_cols)
        self.X_bitmap.setall(1)

        #main metric column
        self.main_metric = self.metrics_regression_list[0] #considering the first element the most important
        if y_is_cat:
            self.main_metric = self.metrics_classification_list[0] #considering the first element the most important

        #calculating the size of population (features x algorithms)
        n_train_sets = 0
        for k in range(1, self.X_train.shape[1] + 1):
            n_train_sets += comb(self.X_train.shape[1] + 1, k, exact=False)
            if math.isinf(n_train_sets):
                break

        print('Nº of training possible basic combinations:'
              , n_train_sets*len(self.selected_algos)
              , '(' + str(n_train_sets),'features combinations,'
              , str(len(self.selected_algos)) +' algorithms)')

        if math.isinf(n_train_sets):
            n_train_sets = self.X_train.shape[1]

        n_train_sets = int(n_train_sets)        
        
        toolbox = ga_toolbox(self)
        #running the GA algorithm
        algorithms.eaSimple(toolbox.population(), toolbox, cxpb=0.8, mutpb=0.3, ngen=self.ngen, verbose=False)
        #free GA memory
        del(toolbox)
        
        #preparing the results
        self.results.sort_values(by=[self.main_metric, 'predict_time'], ascending=[False,True], inplace=True)
        self.results = self.results.rename_axis('train_order').reset_index()        

        print('Fit Time (GA):', int(time.perf_counter() - t0), 's')
        return self.results
    
    def getMetrics(self):
        if self.YisCategorical():
            return self.metrics_classification_list
        #else:
        return self.metrics_regression_list
    
    def Ytype(self):
        return type(self.y_full.iloc[0,0])
    
    def YisCategorical(self) -> bool:
        y_type = type(self.y_full.iloc[0,0])
        
        if (y_type == np.bool_
            or y_type == np.str_):
            return True
        #else
        if ((y_type == np.float_)
            or (len(self.y_full[self.y_colname].unique()) > self.__unique_categoric_limit)):
            return False
        #else
        return True    
    
    def YisContinuous(self) -> bool:
        return not self.YisCategorical()
                   
    def __train_test_split(self):
        y = self.y_full

        stratify=None
        if self.YisCategorical():
            stratify = y
            
        return train_test_split(self.X, y, train_size=0.8, test_size=0.2, random_state=self.RANDOM_STATE, stratify=stratify)

#utilitary methods

from cf_matrix import make_confusion_matrix

def testAutoMLByCSV(csv_path, y_colname):
    return testAutoML(pd.read_csv(csv_path), y_colname=y_colname)

import ds_utils as util


def testAutoML(ds, y_colname):
    automl = AutoML(ds, y_colname, min_x_y_correlation_rate=0.06)
    #automl.setAlgorithm(svm.SVC())
    if automl.YisCategorical():
        print(automl.getBestResult().confusion_matrix)
    
    df4print = automl.__fit()
    print(df4print.head())
    print(automl.getBestResult())
    if automl.getBestResult() is not None:
        print(automl.getBestResult()['features'])
    del(automl)

if __name__ == '__main__':
    #pool = Pool(processes=10)
    automl = AutoML('datasets/iris.csv', 'class'
                    , ds_source_header_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    #automl = AutoML(util.getDSWine_RED(), 'quality'
                    , min_x_y_correlation_rate=0.01
                    #, pool=pool
                    , ngen=1
                    , ds_name='iris'
                    #, algorithms={KNeighborsClassifier: 
                    #    {"n_neighbors": [3,5,7]
                    #     , "p": [2, 3]
                    #     , "n_jobs": [-1]}
                    #    , XGBRFRegressor:{}}
                    , features_engineering=True
                    , n_inter_bayessearch=3)
    print(automl.getResults())
    print(automl.getBestResult())
    print(automl.getBestConfusionMatrix())
    print('finish')
