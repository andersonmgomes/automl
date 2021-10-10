import ds_utils as util
from autoML import AutoML
from sklearn import svm

def testAutoML(ds, y_colname):
    automl = AutoML(ds, y_colname)
    automl.setNFeaturesThreshold(0.8)
    automl.setMinXYcorrRate(0.25)
    automl.setAlgorithm(svm.SVC())
    if automl.YisCategorical():
        print(automl.getBestResult().confusion_matrix)
    return
    df4print = automl.getResults()
    print(df4print.head())
    print(automl.getBestResult())
    print(automl.getBestResult()['features'])
    del(automl)

if __name__ == '__main__':
    testAutoML(util.getDSPriceHousing_ClassProb(), 'high_price')
    testAutoML(util.getDSFuelConsumptionCo2(), 'CO2EMISSIONS')
    testAutoML(util.getDSPriceHousing(), 'Price')
