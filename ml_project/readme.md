

virtual end
Docker container
docker_env:
  image:XXX
  volumnes[..]
  environment
conda env
System env



export current conda_env

pip 

conda env create allows an option --file for an environment file:

conda env create --name envname --file=environments.yml

pip install conda==4.3.16


mlflow run ...  --build-image


entry points define the difference tasks or operations that can be executed as part of a matchine learning project.
like a .py or .sh file
it includes name,command,parameters, environment

any parameters not declared in the parameters field are treated as strings
pass any additional parameters using the --key value syntax. MLflow will pass them to the entry point command.
each entry poing will have only one command and with multiple parameters.

MLFLOW_EXPERIMENT_NAME
MLFLOW_EXPERIMENT_ID
MLFLOW_TMP_DIR


set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow run --entry-point ElasticNet -P alpha=0.5 -P l1_ratio=0.5 --experiment-name "test"