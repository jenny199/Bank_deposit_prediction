from data import DataFrameEncoder
from data import MinMaxScaler
from models import TrainModels
from data import data_loader
from models import calculate_metrics
import pandas as pd

#loading our csv folder as a pandas
data = pd.read_csv('bank-additional-full.csv')
#splitting the data into test and train set
X_train, X_test, y_train, y_test = data_loader(data)

#creating an object for our class in models.py
train = TrainModels()

#training and predicting for logistic regression
log_reg, ypreds = train.train_logistic(X_train, X_test, y_train, y_test)

#training and predicting for random forest regressor
clf, ypreds2 = train.random_forest(X_train, X_test, y_train, y_test)

#training and predicting using xgboost classifier
model, y_pred = train.xgb_classifier(X_train, X_test, y_train, y_test)

#training and predicting using multilayer percepetron
clf_mlp, ypreds3 = train.train_MLP(X_train, X_test, y_train, y_test)

#calculating the metrics
#calculating logistic legression perfomance
calculate_metrics(y_test, ypreds)

#calculating random forest regressor perfomance
calculate_metrics(y_test, ypreds2)

#calculating XGBoost perfomance
calculate_metrics(y_test, y_pred)

#calculating MLP perfomance
calculate_metrics(y_test, ypreds3)

