import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client=MlflowClient()
# experiment_id=client.create_experiment(
#     name="Client exp",tags={"version":"v1","priority":"P1"},

# )

# print("experiment id", experiment_id)

#python -u "d:\Github\MLflow_rh\mlflowClient\mlflow_client.py"


experiment_id="a9edbe20f2ca4cb2b42aee6e0f404a75"
# experiment=client.set_experiment_tag(experiment_id,"farmrwotk","sklearn")

experiment=client.get_experiment(experiment_id)

print("name: {}".format(experiment.name))
print("id: {}".format(experiment.experiment_id))



experiments=client.search_experiments(view_type=ViewType.ALL,order_by=["experiment_id ASC"],filter_string="name = Create exp new")
print(experiments)
print(type(experiments))







run=client.create_run(experiment_id="25",tags={
    "version":"v1",
    "priority":"P1"
},
run_name="run the client")


print(run.data.tags)
print(run.info.run_id)
print(run.info.lifecycle_stage)



from mlflow.deployments import get_deploy_client
client=get_deploy_client("sagemaker")
name=""
model_uri=""
config={}
client.create_deployment(name,model_uri,flavor="python_function",congif=config)


#invking endpoint on sage maker
from data import test
import boto3
import json

endpoint_name=""
region=""
sm=boto3.client('sagemaker',region_name=region)
smrt=boto3.client('runtime.sagemaker',region_name=region)
test_data_json=json.dumps({})
prediction=smrt.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=test_data_json,
    ContentType="application/json"
)
prediction=prediction['Body'].read().decode("ascii")
print(prediction)