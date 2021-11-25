import ds_utils as util
from autoML import AutoML
from sklearn import svm
from sklearn.datasets import load_boston
import pandas as pd

def testAutoML(ds, y_colname):
    automl = AutoML(ds, y_colname)
    automl.setNFeaturesThreshold(0.999)
    automl.setMinXYcorrRate(0.001)
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
    df = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl', na_values='?')
    testAutoML(df, 'survived')   
    '''
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