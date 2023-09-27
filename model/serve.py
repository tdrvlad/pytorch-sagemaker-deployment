from model.model import load_model
import torch
import os
import json
# Ensure this file is executable: chmod +x serve.py
MODEL_FILE_NAME = 'model.pth'


def model_fn(model_dir):
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        input_tensor = torch.tensor(data)
        return input_tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_data = input_data.to(device)
        predictions = model(input_data)
    return predictions


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return prediction.cpu().numpy().tolist()
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")