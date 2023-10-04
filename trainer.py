import torch
from utils import experiment_not_implemented_message
from models import get_model
from optim import get_optimizer
from metric import accuracy


class Trainer:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model: nn.Module
        the model trained by the learner

    criterion: torch.nn.modules.loss
        loss function used to train the `model`

    metric: fn
        function to compute the metric, should accept as input two vectors and return a scalar

    device : str or torch.device)

    optimizer: torch.optim.Optimizer

    is_ready: bool

    Methods
    -------

    fit_epoch: perform several optimizer steps on all batches drawn from `loader`

    fit_epochs: perform multiple training epochs

    evaluate_loader: evaluate `model` on a loader

    """

    def __init__(
            self,
            model,
            criterion,
            metric,
            device,
            optimizer
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer

        self.is_ready = True

    def fit_epoch(self, loader):
        """
        perform several optimizer steps on all batches drawn from `loader`

        Parameters
        ----------
        loader: torch.utils.data.DataLoader

        Returns
        -------
            None
        """
        self.model.train()

        for x, y in loader:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device).type(torch.long)
            x = x.view(-1, x.shape[2]*x.shape[3])

            self.optimizer.zero_grad()

            outs = self.model(x)

            loss = self.criterion(outs, y)

            loss.backward()

            self.optimizer.step()

    def evaluate_loader(self, loader):
        """
        evaluate learner on loader

        Parameters
        ----------
        loader: torch.utils.data.DataLoader

        Returns
        -------
            float: loss
            float: accuracy

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device).type(torch.long)
                x = x.view(-1, x.shape[2] * x.shape[3])

                outs = self.model(x)

                global_loss += self.criterion(outs, y).item() * y.size(0)
                global_metric += self.metric(outs, y).item() * y.size(0)

                n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples


def get_trainer(experiment_name, device, optimizer_name, lr, seed):
    """
    constructs trainer for an experiment for a given seed

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be used;
        possible are {"mnist"}

    device: str
        used device; possible `cpu` and `cuda`

    optimizer_name: str

    lr: float
        learning rate

    seed: int

    Returns
    -------
        Trainer

    """
    torch.manual_seed(seed)

    if experiment_name == "faces":
        criterion = torch.nn.CrossEntropyLoss(reduction="mean").to(device)
        metric = accuracy
    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    model = \
        get_model(experiment_name=experiment_name, device=device)

    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr=lr,
        )

    return Trainer(
        model=model,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer
    )
