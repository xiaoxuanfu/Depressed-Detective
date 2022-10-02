import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
sb.set() 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from math import sqrt



# XGBoostClassifier
def xgbreg(input_file, x_train, x_test, y_train, y_test):
    #df = pd.read_csv(input_file)
    #y = df['label']
    #x = df[['happy','angry','disgust','sad','fear','neutral','surprise']]
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)
    xreg = xgb.XGBClassifier(use_label_encoder=False, eval_metric = 'logloss')
    xreg.fit(x_train, y_train)
    y_train_pred_x = xreg.predict(x_train)
    y_test_pred_x = xreg.predict(x_test)
    return(y_train_pred_x, y_test_pred_x,xreg)
    
# Logistic Regression
def logreg(input_file, x_train, x_test, y_train, y_test,):
    #df = pd.read_csv(input_file)
    #y = df['label']
    #x = df[['happy','angry','disgust','sad','fear','neutral','surprise']]
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)
    logreg = LogisticRegression(random_state=32)
    logreg.fit(x_train, y_train)
    y_train_pred_log = logreg.predict(x_train)
    y_test_pred_log = logreg.predict(x_test)
    return(y_train_pred_log, y_test_pred_log,logreg)

# Random Forest
def forreg(input_file, x_train, x_test, y_train, y_test):
    #df = pd.read_csv(input_file)
    #y = df['label']
    #x = df[['happy','angry','disgust','sad','fear','neutral','surprise']]
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)
    base = RandomForestClassifier(n_estimators=100)
    base.fit(x_train,y_train)
    y_train_pred_for = base.predict(x_train)
    y_test_pred_for = base.predict(x_test)
    return(y_train_pred_for, y_test_pred_for,base)

# Keras
def kerreg(input_file, x_train, x_test, y_train, y_test):
    #df = pd.read_csv(input_file)
    #y = df['label']
    #x = df[['happy','angry','disgust','sad','fear','neutral','surprise']]
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)
    model = Sequential()
    model.add(Dense(12, input_dim=7, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=150, batch_size=10)
    y_train_pred_ker = model.predict(x_train)
    y_test_pred_ker = model.predict(x_test)
    y_train_pred_ker_correct = [0 if y_train_pred_ker[i]<0.5 else 1 for i in range(len(y_train_pred_ker))]
    y_test_pred_ker_correct = [0 if y_test_pred_ker[i]<0.5 else 1 for i in range(len(y_test_pred_ker))]
    return(y_train_pred_ker_correct, y_test_pred_ker_correct,model)

    
def mse(predicted,actual):
    size = actual.size
    mse = ((predicted-actual)**2).sum()/size
    return mse

def r2(predicted,actual):
    size = actual.size
    mse = ((predicted-actual)**2).sum()/size
    var = ((actual-np.mean(actual))**2).sum()/size
    R2 = 1-mse/var
    return R2