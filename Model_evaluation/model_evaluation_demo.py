import warnings
import argparse
import logging
import cloudpickle
import joblib
import pandas as pd

import numpy as np

import sklearn
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models import make_metric
import matplotlib.pyplot as plt

import os 



from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold


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
    exp=mlflow.set_experiment(experiment_name="experiment_evaluation")
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























        # mlflow.log_param("alpha",alpha)
        # mlflow.log_param("l1_ratio",l1_ratio)
        # mlflow.log_metric("RMSE",rmse)
        # mlflow.log_metric("r2",r2)
        # mlflow.log_metric("MAE",mae)
        # mlflow.sklearn.log_model(lr,"RayFirstModel")

        def squared_diff_plus_one(eval_df,_builtin_metrices):
            return np.sum(np.abs(eval_df["prediction"]-eval_df["target"]+1)**2)
        def sum_on_target_divided_by_two(_eval_df,builtin_metrices):
            return builtin_metrices["sum_on_target"]/2
        

        squared_diff_plus_one_metric=make_metric(eval_fn=squared_diff_plus_one,
                                                 greater_is_better=False,name="squared diff plus one")
        sum_on_target_divided_by_two_metric=make_metric(eval_fn=sum_on_target_divided_by_two,
                                                        greater_is_better=True,
                                                        name="sum on target divided by two")
        
        def prediction_target_scatter(eval_df,_builtin_metrics,artifacts_dir):
            plt.scatter(eval_df["prediction"],eval_df["target"])
            plt.xlabel("Targets")
            plt.ylabel("Predictions")
            plt.title("Targets vs. Predictions")
            plot_path=os.path.join(artifacts_dir,"example_scatter_plot.png")
            plt.savefig(plot_path)
            return {"example_scater_plot_artifact":plot_path}
        #setting base line
        baseline_model=DummyRegressor()
        baseline_model.fit(train_x,train_y)
        baseline_predicted_qualities=baseline_model.predict(test_x)
       
        
        (b1_rmse,b1_mae,b1_r2)=eval_metrics(test_y,baseline_predicted_qualities)
        mlflow.log_metrics({
            "Baseline rmse":b1_rmse,
            "Baseline r2":b1_r2,
            "Baseline mae":b1_mae,
        })
        print(" RMSE: %s" % b1_rmse)
        print(" MAE: %s" % b1_mae)
        print(" R2: %s" % b1_r2)
        baseline_sklearn_model_path="baseline_sklearn_model.pkl"
        joblib.dump(baseline_model,baseline_sklearn_model_path)
        baseline_artifacts={"baseline_sklearn_model":baseline_sklearn_model_path}



        class SklearnWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self,artifacts_name):
                self.artifacts_name=artifacts_name
            def load_context(self,context):
                self.sklearn_model=joblib.load(context.artifacts[self.artifacts_name])
            def predict(self,context,model_input):
                return self.sklearn_model.predict(model_input.values)
        # data_dir="test"
        # artifacts={
        #     "sklearn_model":sk_model_path,
        #     "data":data_dir
        # }

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

        mlflow.pyfunc.log_model(artifact_path="baseline_sklean_mlflow_pyfunc",python_model=SklearnWrapper("baseline_sklearn_model"),artifacts=baseline_artifacts,code_path=[r"D:\Github\MLflow_rh\Model_evaluation\model_evaluation_demo.py"],conda_env=conda_env)

       #define threshold
        thresholds={
            "mean_square_error1":MetricThreshold(
                threshold=0.6,
                min_absolute_change=0.1,
                min_relative_change=0.05,
                greater_is_better=False
            )
        }
        baseline_model_uri=mlflow.get_artifact_uri("baseline_sklean_mlflow_pyfunc")
        #artifacts_uri="file:///d:/Github/MLflow_rh/mlruns/721023801212034524/ba69e959dfa7440d8c715dce2c8e37c4/artifacts"
        artifacts_uri=mlflow.get_artifact_uri("sklean_mlflow_pyfunc")
        #RayFirstModel
        mlflow.evaluate(
            artifacts_uri,
            test,
            targets="quality",
            model_type="regressor",
            evaluators=["default"],
            custom_metrics=[
                sum_on_target_divided_by_two_metric,
                squared_diff_plus_one_metric

            ],
            custom_artifacts=[prediction_target_scatter],
            validation_thresholds=thresholds,
            # baseline_model=baseline_model_uri
            #baseline_model_uri,
        )


        
#pip install shap











