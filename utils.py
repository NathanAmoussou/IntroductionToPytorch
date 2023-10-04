from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def experiment_not_implemented_message(experiment_name):
    error = f"{experiment_name} is not available! " \
            f"Possible are: 'mnist', 'faces."

    return error


def get_loader(experiment_name, client_data_path, batch_size, train):
    """

    Parameters
    ----------
    experiment_name: str

    client_data_path: str

    batch_size: int

    train: bool

    Returns
    -------
        * torch.utils.data.DataLoader

    """

    if experiment_name == "mnist":
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        dataset = MNIST(root=client_data_path, train=train, transform=transform)

    elif experiment_name == "faces":

        transform = ToTensor()

        dataset = FacesDataset(root=client_data_path, train=train, transform=transform)

    else:
        raise NotImplementedError(
            experiment_not_implemented_message(experiment_name=experiment_name)
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

