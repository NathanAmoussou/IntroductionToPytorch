import torch
import time
from args import parse_args
from trainer import get_trainer
from loader import get_loader
from tqdm import tqdm

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    seed = (args.seed if (("seed" in args) and (args.seed >= 0)) else int(time.time()))
    torch.manual_seed(seed)

    print("\n=> Load Data..")
    train_loader = \
        get_loader(experiment_name=args.experiment,
                   batch_size=args.batch_size)

    print("\n=> Build Trainer..")
    trainer = \
        get_trainer(experiment_name=args.experiment,
                    device=args.device,
                    optimizer_name=args.optimizer,
                    lr=args.lr,
                    seed=seed)

    print("\n=> Start training..")
    for t in tqdm(range(args.epochs)):
        trainer.fit_epoch(loader=train_loader)
        loss, metric = trainer.evaluate_loader(loader=train_loader)
        print(f"Epoch: {t}, Loss: {loss:.2f}, Accuracy: {metric*100:.2f}%")

