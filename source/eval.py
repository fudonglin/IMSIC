import argparse
import torch

import utils
from source import params


def evaluate(path, dataset):
    model = torch.load(path)
    model.eval()
    test_loader = utils.get_MNIST_test_loader() if dataset == 'mnist' else utils.get_Fashion_test_loader()
    utils.print_report(model, test_loader)


if __name__ == '__main__':
    args = params.opt
    evaluate(args.path, args.dataset)
