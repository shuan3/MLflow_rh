
#storage
backend stora 
DB store (MSSQL) and file store (S3)

artifact sotre



#Netwwork
REST API (Access trakcing server over HTTP)

RBC

Proxt access


Tracking server-> client


command

mlflow server --backend-store-uri sqlite://mlflow.db --default-artifact-uri ./mlflow-artifacts --host 127.0.0.1 --port 5000


scenario 4 
localhost - > remote host - > remote host (postgreSQL) 
|
S3 remote host














Auto logging
mlflow.autolog()
enable auto-logging for each supported library that is installed

Parameters
log_models
log_input_examples
log_model_signatures
log_dataset
disable
exclusuve

mlflow.<lib>.autolog()
use library-specific mlflow auto-logging functions.