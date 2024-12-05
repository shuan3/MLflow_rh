
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
import sklearn
import joblib
import cloudpickle
import os

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
exp=mlflow.set_experiment(experiment_name="experiment_7")
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
    sk_model_path="sklearn_model.pkl"
    joblib.dump(lr,sk_model_path)
    
    data_dir="test"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data.to_csv(data_dir+'/data.csv')
    train.to_csv(data_dir+'/train.csv')
    test.to_csv(data_dir+'/test.csv')

    class SklearnWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self,context):
            self.sklearn_model=joblib.load(context.artifacts["sklearn_model"])
        def predict(self,context,model_input):
            return self.sklearn_model.predict(model_input.values)

    artifacts={
        "sklearn_model":sk_model_path,
        "data":data_dir
    }

    conda_env={
        "channels":["defaults"],
        "dependencies":[
            "python={}".format(3.12),
            "pip",
            {"pip":["mlflow=={}".format(mlflow.__version__),
                    "sklearn=={}".format(sklearn.__version__),
                    "cloudpickle=={}".format(cloudpickle.__version__),

            ]},
        ],
        "name":"sklearn_env"
    }

    mlflow.pyfunc.log_model(artifact_path="sklean_mlflow_pyfunc",python_model=SklearnWrapper(),artifacts=artifacts,code_path=[r"D:\Github\MLflow_rh\MLflowModelRegistration\customerized_pythonmodel.py"],conda_env=conda_env)



 


