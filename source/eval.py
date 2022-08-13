import argparse
import torch

import utils

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--path', type=str, default='./../models/cnn/cnn_epoch_8.pt')
parser.add_argument("-dataset", "--dataset", type=str, default='mnist', choices=['mnist', 'fashion-mnist'],
                    help='the root directory for data.')
args = parser.parse_args()


def evaluate(path, dataset):
    model = torch.load(path)
    model.eval()
    test_loader = utils.get_MNIST_test_loader() if dataset == 'mnist' else utils.get_Fashion_test_loader()
    utils.print_report(model, test_loader)


if __name__ == '__main__':
    evaluate(args.path, args.dataset)