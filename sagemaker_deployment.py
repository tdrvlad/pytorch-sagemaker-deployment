import os
import boto3
from sagemaker.pytorch import PyTorchModel
import tarfile
from model.model import SimpleModel, save_model
from model.serve import MODEL_FILE_NAME
import torch

import dotenv
dotenv.load_dotenv('.env')
# Load variables defined in the .env file containing the AWS credentials (used by boto3)


def push_model_to_s3(model_path, bucket_name='trained-models-artifacts'):
    s3 = boto3.client('s3')
    archive_path = compress_model(model_path)
    key = os.path.basename(archive_path)
    s3.upload_file(
        Filename=archive_path,
        Bucket=bucket_name,
        Key=key
    )
    return f"s3://{bucket_name}/{key}"


def deploy_model(model_path, role_arn=None, instance_type='ml.m5.large', endpoint_name='simple-model-endpoint', torch_version='1.13.1', py_version='py39'):
    if role_arn is None:
        role_arn = os.getenv("AWS_ROLE_ARN")

    model_uri = push_model_to_s3(model_path=model_path)

    if torch_version != torch.__version__.split("+")[0]:
        raise ValueError("Torch Version differs.")

    pytorch_model = PyTorchModel(
        model_data=model_uri,
        role=role_arn,
        framework_version=torch_version,
        py_version=py_version,
        entry_point='model/serve.py',
        source_dir='.'
    )

    delete_endpoint_config(endpoint_name)

    pytorch_model.deploy(
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=1
    )


def does_endpoint_config_exist(endpoint_config_name):
    sagemaker_client = boto3.client('sagemaker')
    try:
        sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        return True
    except sagemaker_client.exceptions.ClientError:
        return False


def delete_endpoint_config(endpoint_config_name):
    sagemaker_client = boto3.client('sagemaker')
    if does_endpoint_config_exist(endpoint_config_name):
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)


def compress_model(model_path):

    dir_name = os.path.dirname(model_path)
    file_name = os.path.basename(model_path)
    archive_file_name = f"{file_name}.tar.gz"
    with tarfile.open(os.path.join(dir_name, archive_file_name), 'w:gz') as tar:
        tar.add(model_path, arcname=file_name)

    return os.path.join(dir_name, archive_file_name)


def test_deploy_simple_model_directly():
    LOCAL_MODEL_DIR = './artifacts'

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, MODEL_FILE_NAME)

    model = SimpleModel()
    save_model(model, LOCAL_MODEL_PATH)
    deploy_model(LOCAL_MODEL_PATH)


if __name__ == '__main__':
    test_deploy_simple_model_directly()
