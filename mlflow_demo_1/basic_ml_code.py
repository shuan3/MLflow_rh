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

# '''

# start experiment
#mlflow ui
#python -u "d:\Github\MLflow_rh\mlflow_demo_1\basic_ml_code.py" --alpha 0.3 --l1_ratio 0.1

# '''
    exp=mlflow.set_experiment(experiment_name="experiment_1")
    with mlflow.start_run(experiment_id=exp.experiment_id):

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








'''
Validate the model before deployment
Run the following code to validate model inference works on the example payload, prior to deploying it to a serving endpoint
from mlflow.models import validate_serving_input

model_uri = 'runs:/035c8f2a33c94e64b63cf57921c14a20/RayFirstModel'

# The logged model does not contain an input_example.
# Manually generate a serving payload to verify your model prior to deployment.
from mlflow.models import convert_input_example_to_serving_input

# Define INPUT_EXAMPLE via assignment with your own input example to the model
# A valid input example is a data instance suitable for pyfunc prediction
serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)
Make Predictions
Predict on a Pandas DataFrame:
import mlflow
logged_model = 'runs:/035c8f2a33c94e64b63cf57921c14a20/RayFirstModel'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))
Predict on a Spark DataFrame:
import mlflow
from pyspark.sql.functions import struct, col
logged_model = 'runs:/035c8f2a33c94e64b63cf57921c14a20/RayFirstModel'

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))
'''