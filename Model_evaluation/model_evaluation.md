mlflow.evaluate()
API provided by Mlflow to evaluate the performance of mlflow models and saves the evaluation metrics and graph to the trakcing server.
Can generate various model performance plots, such as confusion matrix, precision=recall curve or ROC curve depending on task type.
Can provide model explanations 
logged to mlflow tracking.
supports with python function (pyfunc) flavor

evaluate
model->string
data->numpy array, pandas datafrmae or spark dataframe, mlflow data.dataset.Dataset
model_type->regressor classifier question-answering
targers->list of evaluation labels
dataset_path
feature_names
evaluators-> kist of   mlflow.models.list_evaluators()
evaluator_config->dictionary of configurations to supply (metric_prefix,...)
validation_thresholds
custom_artifacts=[fucntion_name]
baseline_model
env_manager(virtualenv,conda,local)