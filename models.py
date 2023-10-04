import torch.nn as nn
from utils import experiment_not_implemented_message


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dim
        self.num_classes = output_dim
        self.fc = nn.Linear(self.input_dimension, self.num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


def get_model(experiment_name, device):
    """
    create model

    Parameters
    ----------
    experiment_name: str

    device: str
        either cpu or cuda


    Returns
    -------
        model (torch.nn.Module)

    """

    if experiment_name == "mnist":
        pass
    elif experiment_name == "faces":
        model = LinearLayer(input_dim=112*92, output_dim=40, bias=True)
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = model.to(device)

    return model
