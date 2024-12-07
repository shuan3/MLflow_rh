import mlflow

parameters={
    "alpha":0.3,
    "l1_ratio":0.1
}

experiment_name="test"
entry_point="mlproject"

mlflow.projects.run(
    uri="",
    entry_point=entry_point,
    parameters=parameters,
    experiment_name=experiment_name
)