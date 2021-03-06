import ds_utils as util
from autoML import AutoML
from sklearn import svm
from sklearn.datasets import load_boston
import pandas as pd
#import modin.pandas as pd #https://modin.readthedocs.io/
from multiprocessing import Pool
    
def testAutoMLByCSV(csv_path, y_colname, ds_name):
    return testAutoML(pd.read_csv(csv_path), y_colname=y_colname, ds_name=ds_name)

def testAutoML(ds, y_colname, ds_name):
    pool = Pool(processes=6)
    automl = AutoML(ds, y_colname, min_x_y_correlation_rate=0.06, pool=pool, ds_name=ds_name)
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
    testAutoML(util.getDSPriceHousing_ClassProb().drop('Address', axis=1), 'high_price', 'house_class')
    #ray.init(ignore_reinit_error=True)
    #testAutoMLByCSV('datasets/sentimentos.csv', y_colname='classe')
    #df = pd.read_csv('datasets/titanic_original.csv', na_values='?')
    #print(df)
    #testAutoML(df, y_colname='survived')
    #testAutoMLByCSV('datasets/viaturas4Model.csv', 'y')
    #testAutoML(util.getDSFuelConsumptionCo2(), 'CO2EMISSIONS')
    '''
    df = pd.read_csv('datasets/titanic.csv')
    testAutoML(df, 'survived')   

    testAutoML(util.getDSPriceHousing_ClassProb().drop('Address', axis=1), 'high_price')
    testAutoML(util.getDSFuelConsumptionCo2(), 'CO2EMISSIONS')
    testAutoML(util.getDSPriceHousing().drop('Address', axis=1), 'Price')
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['price'] = boston.target
    testAutoML(df, 'price')
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None)
    df.columns = ['buying','maint', 'doors', 'persons', 'lug_boot', 'safety', 'car']
    testAutoML(df, 'car')
    '''