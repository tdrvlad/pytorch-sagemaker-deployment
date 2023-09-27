import torch
import os
import mlflow
from mlflow.deployments import get_deploy_client
from model.model import SimpleModel
import pandas as pd
import dotenv
dotenv.load_dotenv('.env')


class PyTorchModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.model = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        model_path = context.artifacts
        self.model = SimpleModel.load(model_path)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, data: pd.DataFrame) -> pd.DataFrame:
        input_tensor = torch.tensor(data.values, dtype=torch.float)
        output = self.model(input_tensor)
        return output.detach().numpy()


def log_model_to_mlflow(model_path, mlflow_server_uri, experiment_name='Demo'):
    mlflow.set_tracking_uri(mlflow_server_uri)
    client = mlflow.tracking.MlflowClient()

    existing_experiments_names = [experiment.name for experiment in client.search_experiments()]
    if experiment_name not in existing_experiments_names:
        client.create_experiment(experiment_name)
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    run_id = client.create_run(experiment_id).info.run_id

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("Param", 1)
        mlflow.log_metric("Metric", 2)
        mlflow.set_tag(r"Tag", "Value")
        model_info = mlflow.pyfunc.log_model(
            python_model=PyTorchModel(),
            artifact_path=model_path
        )
        model_uri = model_info.model_uri
    return run_id, model_uri


def deploy_model(model_uri, mlflow_server_uri, instance_type="ml.t2.xlarge", endpoint_name='demo-deploy', role_arn=None, region=None, bucket_name=None, image_uri=None):
    if role_arn is None:
        role_arn = os.getenv("AWS_ROLE_ARN")
    if region is None:
        region = os.getenv("AWS_DEFAULT_REGION")
    if bucket_name is None:
        bucket_name = os.getenv("AWS_BUCKET_NAME")
    if image_uri is None:
        image_uri = os.getenv("AWS_IMAGE_URI")

    config = {
        "execution_role_arn": role_arn,
        "bucket_name": bucket_name,
        "image_url": image_uri,
        "region_name": region,
        "instance_type": instance_type,
        "instance_count": 1,
        "timeout_seconds": 3000,
        "variant_name": "demo"
    }
    mlflow.set_tracking_uri(mlflow_server_uri)
    client = get_deploy_client("sagemaker")
    client.create_deployment(
        name=endpoint_name,
        model_uri=model_uri,
        flavor="python_function",
        config=config,
    )


def test_deploy_simple_model():
    MLFLOW_SERVER_URL = "http://127.0.0.1:5000"
    LOCAL_MODEL_DIR = './artifacts'
    MODEL_FILE = 'model.pth'

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, MODEL_FILE)

    model = SimpleModel()
    model.save(LOCAL_MODEL_PATH)
    run_id, model_uri = log_model_to_mlflow(model_path=LOCAL_MODEL_PATH, mlflow_server_uri=MLFLOW_SERVER_URL)
    deploy_model(model_uri, mlflow_server_uri=MLFLOW_SERVER_URL)


if __name__ == '__main__':
    test_deploy_simple_model()
