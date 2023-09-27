import torch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, dim=-1)
        return x


def save_model(model: torch.nn.Module, save_path):
    torch.jit.script(model).save(save_path)


def load_model(model_path):
    model = torch.jit.load(model_path)
    return model


# # Alternative load and save
#
# def save_model(model: torch.nn.Module, save_path):
#     torch.save(model.state_dict(), save_path)

# def load_model(model_path):
#     model = SimpleModel()
#     model.load_state_dict(torch.load(model_path))
#     return model


def test_simple_model():
    model = SimpleModel()

    save_model(model, 'test_model.pt')
    model = load_model('test_model.pt')

    input_tensor = torch.rand(10)
    input_batch = input_tensor.unsqueeze(0)
    output_batch = model(input_batch)
    output_tensor = output_batch[0]
    output_array = output_tensor.detach().cpu().numpy()

    print(output_array)


if __name__ == '__main__':
    test_simple_model()

