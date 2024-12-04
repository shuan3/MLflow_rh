Models
Standard format
Central repository
API


Storage format
model, meta of model, model version,hyperparameters
files container image

model signature: specify the input output data types and shapes that the model expects and returns for the model
tensor based example inputs:[{"name":"imgaes"}]
column based

Model signature enforcement (signature enforcement, name-ordering enforcement (input name mathcing), input-type enhancement)


model API
flavor: refers to s[ecific way of serializing and storing a machine learning model. ]





save_model: save model to the local file system while putting metadata on tracking server
sk_model
path
conda_env
code_path
....
https://mlflow.org/docs/latest/models.html
https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model

Log_model: logs a skilearn model as an artifact to a tracking server and make it accessible in MLflow UI or other interfacts.It contains Mlflow.sklearn,mlflow.pytunc
artifact_path
registered_model_name
await_registration_for



load_model: load a sikit learn model from a local file or a run
dst_path
model_uri





Model registration
log_model put optoin registered_model_name

register_model: create a new model version in model registry for the model file specified by model_uri
model_uri  'runs:/{}/model'.format(run_info.run_it)
name
await_registration_for
tags
