import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler


def data_loader(data):
    y = data['bank_account']
    X = data.drop(columns = ['bank_account', 'uniquesId'])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, 
                                                                    random_state=21)
    return X_train, X_test, y_train, y_test
class DataFrameEncoder(TransformerMixin):

    def __init__(self):
        """Encode the data.

        Columns of data type object are appended in the list. After 
        appending Each Column of type object are taken dummies and 
        successively removed and two Dataframes are concated again.

        """
    def fit(self, X, y=None):
        self.object_col = []
        for col in X.columns:
            if(X[col].dtype == np.dtype('O')):
                self.object_col.append(col)
        return self

    def transform(self, X, y=None):
        dummy_df = pd.get_dummies(X[self.object_col],drop_first=True)
        X = X.drop(X[self.object_col],axis=1)
        X = pd.concat([dummy_df,X],axis=1)
        return X


scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
