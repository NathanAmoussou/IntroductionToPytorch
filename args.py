import torch
import argparse
import warnings


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--experiment',
        help='name of the experiment, possible are:'
             '{"faces"}',
        type=str,
        required=True
    )

    parser.add_argument(
        '--epochs',
        help='number of epochs; default is 10',
        type=int,
        default=10
    )

    parser.add_argument(
        '--optimizer',
        help='optimizer; default is adam',
        type=str,
        default="adam"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate; default is 1e-4.',
        default=1e-4
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help='batch size used, default is 64.',
        default=64
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or cuda; default is cpu',
        type=str,
        default="cpu"
    )

    parser.add_argument(
        "--seed",
        help='random seed; if not specified the system clock is used to generate the seed',
        type=int,
        default=argparse.SUPPRESS
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA is not available, device is automatically set to \"CPU\"!", RuntimeWarning)

    return args
