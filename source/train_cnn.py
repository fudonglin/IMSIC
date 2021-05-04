import os

import numpy as np
import torch
from torch import nn
import net as n
import params
import utils
from tqdm import tqdm
import test_cnn


def train():
    """ Train CNNs

    Args:
       dataset: name of dataset

    """

    torch.manual_seed(1)
    np.random.seed(1)

    opt = params.opt
    dataset = opt.dataset

    # hyper-parameters
    num_epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.cnn_lr

    # initial data and label
    labels, nums, gene_indexes, gene_nums = utils.initial_parameters(dataset)

    total_images, total_labels = torch.empty(0), torch.empty(0, dtype=int)
    if dataset == 'mnist':
        total_images, total_labels = utils.initial_mnist_imbalanced_data(labels, nums, gene_indexes, gene_nums)
    elif dataset == 'fashion-mnist':
        total_images, total_labels = utils.initial_fashion_imbalanced_data(labels, nums, gene_indexes, gene_nums)

    # CNNs
    model = n.ConvNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train CNNs
    batch_num = np.math.ceil(len(total_labels) / batch_size)
    total_step = batch_num
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):

        for it in tqdm(range(batch_num)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size

            images = total_images[start_idx:end_idx]
            labels = total_labels[start_idx:end_idx]

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            # loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (it + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, it + 1, total_step, loss.item(), (correct / total) * 100))

        if (epoch + 1) % 2 == 0:
            # save model
            cnn_dir = opt.cnn_dir
            os.makedirs(cnn_dir, exist_ok=True)
            path = cnn_dir + '/cnn_epoch_{}.pt'.format(epoch+1)
            torch.save(model, path)

            # print report
            test_cnn.test(path, dataset)



