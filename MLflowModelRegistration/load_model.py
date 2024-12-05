
import pickle
import mlflow
import warnings
import argparse
import logging
import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

import os
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema,ColSpec

parser=argparse.ArgumentParser()
parser.add_argument("--alpha",type=float,required=False,default=0.5)
parser.add_argument("--l1_ratio",type=float,required=False,default=0.5)
args=parser.parse_args()
def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse,mae,r2




warnings.filterwarnings("ignore")
np.random.seed(40)
data=pd.read_csv(r"D:\Github\MLflow_rh\Data\winequality.csv")
print(data.shape[0])
print(data.columns)
# print(data['quality'])
train,test=train_test_split(data)

train_x=train.drop(['quality'],axis=1)
test_x=test.drop(['quality'],axis=1)
train_y=train[['quality']]
test_y=test[['quality']]

# alpha=args.alpha
# l1_ratio=args.l1_ratio
# exp=mlflow.set_experiment(experiment_name="experiment_6")
# alpha=0.1*i
# l1_ratio=0.1*i
# mlflow.sklearn.autolog(log_input_examples=False,
#                         log_model_signatures=False,log_models=False)
# lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
# lr.fit(train_x,train_y)
# predicted_qualities=lr.predict(test_x)

# (rmse,mae,r2)=eval_metrics(test_y,predicted_qualities)

# print("Elasticnet model (alpha={:f},l1_ratio={:f}):".format(alpha,l1_ratio))
# print(" RMSE: %s" % rmse)
# print(" MAE: %s" % mae)
# print(" R2: %s" % r2)
# params={"alpah":alpha,"l1_ratio":l1_ratio}
# mlflow.log_params(params)
# # mlflow.log_param("alpha",alpha)
# # mlflow.log_param("l1_ratio",l1_ratio)
# mlflow.log_metric("RMSE",rmse)
# mlflow.log_metric("r2",r2)
# mlflow.log_metric("MAE",mae)




#pointing to storage location in meta.yaml
ld=mlflow.sklearn.load_model(model_uri="file:///d:/Github/MLflow_rh/mlruns/480802286042789171/346935a64a964d6a9715f51860021995/artifacts/external_model")

predicted_qualities=ld.predict(test_x)
(rmse,mae,r2)=eval_metrics(test_y,predicted_qualities)


print("new run for load model")
# print("Elasticnet model (alpha={:f},l1_ratio={:f}):".format(alpha,l1_ratio))
print(" RMSE: %s" % rmse)
print(" MAE: %s" % mae)
print(" R2: %s" % r2)