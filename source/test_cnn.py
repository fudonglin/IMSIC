import torch

import utils


def test(path, dataset):
    model = torch.load(path)
    model.eval()
    test_loader = utils.get_MNIST_test_loader() if dataset == 'mnist' else utils.get_Fashion_test_loader()
    utils.print_report(model, test_loader)