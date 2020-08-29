import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
class TrainModels():
  '''
      each function in this class; 
      Receive data to train(features and target) as parameters
      creates a model object and fit the model with the data
      returns a model fitted
  '''

  def __init(self):
      
      return self
  #a function to train a logistic regression
  def train_logistic(self, X_train, X_test, y_train, y_test):

      log_reg = LogisticRegression()
      log_reg.fit(X_train, y_train)
      ypreds = log_reg.predict(X_test)
      return log_reg, ypreds

  def random_forest(self, X_train,X_test, y_train, y_test):
      clf = RandomForestClassifier(n_estimators=100)
      #Train the model using the training sets y_pred=clf.predict(X_test)
      clf.fit(X_train,y_train)
      ypreds2 = clf.predict(X_test)
      return clf, ypreds2
  
  def train_MLP(self, X_train, X_test, y_train, y_test):
      clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                          hidden_layer_sizes=(15,), random_state=1)
      clf_mlp.fit(X_train, y_train)
      ypreds3 = clf_mlp.predict(X_test)
      return clf_mlp, ypreds3
  
  def xgb_classifier(self, X_train, X_test, y_train, y_test):
      #Create XGB Classifier
      model = XGBClassifier()
      #Train the model using the training sets
      model.fit(X_train,y_train)
      y_pred =model.predict(X_test)
      return model,y_pred

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))