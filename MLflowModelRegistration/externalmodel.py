
import pickle
import mlflow
import mlflow.sklearn

filename="external_model.pkl"
loaded_model=pickle.load(open(filename,"rb"))

# mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
exp=mlflow.set_experiment(experiment_name="external")
mlflow.start_run()
mlflow.sklearn.log_model(
    loaded_model,'external_model',serialization_format='cloudpickle',
    registered_model_name="registered_external_model"
)


mlflow.end_run()