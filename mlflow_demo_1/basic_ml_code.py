import warnings
import argparse
import logging
import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

parser=argparse.ArgumentParser()
parser.add_argument("--alpha",type=float,required=False,default=0.5)
parser.add_argument("--l1_ratio",type=float,required=False,default=0.5)
args=parser.parse_args()
def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error
    r2=r2_score(actual,pred)
    return rmse,mae,r2

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    data=pd.read_csv(r"D:\Github\MLflow_rh\mlflow_demo_1\winequality.csv")
    print(data.shape[0])
    print(data.columns)
    # print(data['quality'])
    train,test=train_test_split(data)

    train_x=train.drop(['quality'],axis=1)
    test_x=test.drop(['quality'],axis=1)
    train_y=train[['quality']]
    test_y=test[['quality']]

    alpha=args.alpha
    l1_ratio=args.l1_ratio

    lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    lr.fit(train_x,train_y)
    predicted_qualities=lr.predict(test_x)

    (rmse,mae,r2)=eval_metrics(test_y,predicted_qualities)

    print("Elasticnet model (alpha={:f},l1_ratio={:f}):".format(alpha,l1_ratio))
    print(" RMSE: %s" %rmse)
    print(" MAE: %s" %mae)
    print(" R2: %s" %r2)