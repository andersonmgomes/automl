import pandas as pd
import seaborn as sns
import numpy as np

DS_PATH = 'datasets/'

#load dataset
def getDSFuelConsumptionCo2():
    return pd.read_csv(DS_PATH + 'FuelConsumptionCo2.csv')

def getDSPriceHousing():
    return pd.read_csv(DS_PATH + 'USA-priceHousing.csv')

#classification problem
def getDSPriceHousing_ClassProb():
    ds_house_classprob = getDSPriceHousing()
    price_75 = ds_house_classprob.Price.describe()['75%']
    ds_house_classprob['high_price'] = ds_house_classprob['Price']>price_75
    ds_house_classprob.drop('Price', axis=1, inplace=True)
    return ds_house_classprob

def getDSWine_RED():
    return pd.read_csv(DS_PATH + 'winequality-red.csv', sep=';')

def getCorrHeatMap(ds, annot=True):
    return sns.heatmap(ds.corr(), cmap='coolwarm', fmt='.2f', linewidths=0.1,vmax=1.0, square=True
                       , linecolor='white', robust=True, annot=annot);
    
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def getConfusionMatrixHeatMap(y_true_label, y_predict_label):
    cf_matrix = confusion_matrix(y_true_label, y_predict_label)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    return sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues');    



