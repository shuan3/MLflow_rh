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

parser=argparse.ArgumentParser()
parser.add_argument("--alpha",type=float,required=False,default=0.5)
parser.add_argument("--l1_ratio",type=float,required=False,default=0.5)
args=parser.parse_args()
def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
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


#mlflow.set_tracking_uri(uri="")
#print(mlflow.get_tracking_uri())

# '''

# start experiment
#mlflow ui
#python -u "d:\Github\MLflow_rh\mlflow_demo_1\basic_ml_code.py" --alpha 0.3 --l1_ratio 0.1
#git rm -r --cached .
# '''
exp=mlflow.set_experiment(experiment_name="experiment_1")
with mlflow.start_run(experiment_id=exp.experiment_id):
    mlflow.autolog(
        log_input_examples=True
    )
    lr=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    lr.fit(train_x,train_y)
    predicted_qualities=lr.predict(test_x)

    (rmse,mae,r2)=eval_metrics(test_y,predicted_qualities)

    print("Elasticnet model (alpha={:f},l1_ratio={:f}):".format(alpha,l1_ratio))
    print(" RMSE: %s" % rmse)
    print(" MAE: %s" % mae)
    print(" R2: %s" % r2)

    
    mlflow.log_param("alpha",alpha)
    mlflow.log_param("l1_ratio",l1_ratio)
    mlflow.log_metric("RMSE",rmse)
    mlflow.log_metric("r2",r2)
    mlflow.log_metric("MAE",mae)
    mlflow.sklearn.log_model(lr,"RayFirstModel")


 


