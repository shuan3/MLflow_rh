Mlflow client
Tracking Server <--> Client

Mlflow client encompasses a collection of tools and libraries taht allow users to interact with MLflow platform programmatically.


Mlflow library is high-level API designed for interaction with MLflow trakcing server.
MLflow client serves as a lower-level API that directly translates to MLflow REST API calls and is primarilly used to interact with the core component of MLflow.


Experiment management
run management and tracking
model versionning and management


functions
create_experiment
set_experiment_tag
get_experiment
get_experiment_by_name
rename_expriement
delete_experiment
restore_experiment
search_experiment
get_metric_history