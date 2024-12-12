conda activate mlflow_demo1

set MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow doctor
mlflow artifacts log-artifacts --local-dir cli_artifacts

mlflow db upgrade sqlite:///mlflow.db

mlflow experiments create --experiment-name cli_experiment

mlflow experiments csv --experiment-id 6 --filename exp_6_file.csv

mlflow runs describe --run-id XXXX




mlflow sagemaker build-and-push-container --container xgb --env-manager conda
