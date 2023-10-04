import torch.optim as optim


def get_optimizer(optimizer_name, model, lr):
    """returns torch.optim.Optimizer given an optimizer name, a model and learning rate

    Parameters
    ----------
    optimizer_name: str
        possible are {"adam"}

    model: torch.nn.Module

    lr: float


    Returns
    -------
        * torch.optim.Optimizer
    """

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr
        )
    else:
        raise NotImplementedError(
            f"{optimizer_name} is not a possible optimizer name; possible are: 'sgd', 'adam'"
        )
