import xgboost
import shap
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import mlflow
from mlflow.models import MetricThreshold

# load UCI Adult Data Set; segment it into training and test sets
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# construct an evaluation dataset from the test set
eval_data = X_test
eval_data["label"] = y_test

# Log and evaluate the candidate model
candidate_model = xgboost.XGBClassifier().fit(X_train, y_train)

with mlflow.start_run(run_name="candidate") as run:
    candidate_model_uri = mlflow.sklearn.log_model(
        candidate_model, "candidate_model", 
        #signature=infer_signature(test_x,predicted_qualities)
    ).model_uri

    candidate_result = mlflow.evaluate(
        candidate_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
    )

# Log and evaluate the baseline model
baseline_model = DummyClassifier(strategy="uniform").fit(X_train, y_train)

with mlflow.start_run(run_name="baseline") as run:
    baseline_model_uri = mlflow.sklearn.log_model(
        baseline_model, "baseline_model",
         # signature=signature
    ).model_uri

    baseline_result = mlflow.evaluate(
        baseline_model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
    )

# Define criteria for model to be validated against
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.8,  # accuracy should be >=0.8
        min_absolute_change=0.05,  # accuracy should be at least 0.05 greater than baseline model accuracy
        min_relative_change=0.05,  # accuracy should be at least 5 percent greater than baseline model accuracy
        greater_is_better=True,
    ),
}

# Validate the candidate model agaisnt baseline
mlflow.validate_evaluation_results(
    candidate_result=candidate_result,
    baseline_result=baseline_result,
    validation_thresholds=thresholds,
)
