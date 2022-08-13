import torch
from sklearn.metrics import classification_report
from torch.autograd import Variable
import numpy as np
import discriminator as d
import generator as g
import params
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
import os

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
opt = params.opt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return transforms.Compose(
        [transforms.Resize(params.opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


def init_GAN():
    generator = g.Generator()
    discriminator = d.Discriminator()
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    return generator, discriminator, adversarial_loss, categorical_loss, continuous_loss


def get_MNIST_train_loader():
    os.makedirs(opt.data_dir, exist_ok=True)
    train_set = datasets.MNIST(root=opt.data_dir, download=True, train=True, transform=get_transform())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.opt.batch_size, shuffle=True)
    return train_loader


def get_MNIST_test_loader():
    test_set = datasets.MNIST(opt.data_dir, download=True, train=False, transform=get_transform())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.opt.batch_size, shuffle=True)
    return test_loader


def get_Fashion_test_loader():
    test_set = datasets.FashionMNIST(opt.fashion_data_dir, download=True, train=False, transform=get_transform())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.opt.batch_size, shuffle=True)
    return test_loader


def get_static_gen_input():
    static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
    static_label = to_categorical(
        np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
    )
    static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))
    return static_z, static_label, static_code


def sample_image(generator, n_row, batches_done, folder_im):
    static_z, static_label, static_code = get_static_gen_input()
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, folder_im + "/static/%d.png" % batches_done, nrow=n_row, normalize=True)
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    save_image(sample1.data, folder_im + "/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, folder_im + "/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


def sample_image2(generator, n_row, batches_done, folder_im):
    static_z, static_label, static_code = get_static_gen_input()
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, folder_im + "/static/%d.png" % batches_done, nrow=n_row, normalize=True)
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    for i in range(opt.code_dim):
        l = [zeros] * opt.code_dim
        l[i] = c_varied
        c = Variable(FloatTensor(np.concatenate(tuple(l), -1)))
        sample = generator(static_z, static_label, c)
        name = folder_im + '/varying_c' + str(i + 1) + '/%d.png' % batches_done
        save_image(sample.data, name, nrow=n_row, normalize=True)


def get_structure_loss(loss_function, code_input, pred_code, negative_edges, positive_edges):
    loss = 0
    losses = []
    gains = []
    gains2 = []
    for edge in positive_edges:
        losses.append(loss_function(pred_code[:, edge[1]], code_input[:, edge[0]]))

    for edge in negative_edges:
        gains.append(loss_function(pred_code[:, edge[1]], code_input[:, edge[0]]))

    for i in range(pred_code.size()[1]):
        for j in range(i, pred_code.size()[1]):
            gains2.append(loss_function(pred_code[:, i], code_input[:, j]))

    if sum(gains) + sum(gains2) > sum(losses):
        return loss
    else:
        return sum(losses) - sum(gains) - sum(gains2)


def get_imbalance_dataloader(labels, sizes):
    """ Returns imbalanced train loader """

    dataset_list = []
    for i in range(0, len(labels)):
        label = labels[i]
        size = sizes[i]
        temp_dataset = get_imbalanced_dataset(label, size)
        dataset_list.append(temp_dataset)

    dataset = ConcatDataset(dataset_list)
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)


def get_imbalanced_dataset(label, num):
    """ Returns specific number of label """

    os.makedirs(opt.data_dir, exist_ok=True)
    dataset = datasets.MNIST(opt.data_dir, train=True, download=True, transform=get_transform())
    index = (dataset.targets == label)
    dataset.targets = dataset.targets[index]
    dataset.data = dataset.data[index]

    indices = torch.randperm(len(dataset.targets))[:num]
    dataset.targets = dataset.targets[indices]
    dataset.data = dataset.data[indices]
    return dataset


def initial_mnist_imbalanced_data(labels, nums, gene_indexes, gene_nums):
    """ Returns imbalanced data for CNNs """

    images = torch.empty(0).to(device)
    targets = torch.empty(0, dtype=int).to(device)
    length = len(labels)
    for i in range(length):
        label = labels[i]
        num = nums[i]
        generate_index = gene_indexes[i]
        gene_size = gene_nums[i]

        temp_images, temp_labels = get_specific_label(label, num)
        temp_images, temp_labels = temp_images.to(device), temp_labels.to(device)
        temp_gene_images, temp_gene_labels = generate_sample(label, generate_index, gene_size=gene_size)

        images = torch.cat([images, temp_images, temp_gene_images], dim=0)
        targets = torch.cat([targets, temp_labels, temp_gene_labels], dim=0)

    if cuda:
        images, targets = images.cpu(), targets.cpu()
    x, y = shuffle(images.detach().numpy(), targets.detach().numpy())
    return torch.from_numpy(x), torch.from_numpy(y)


def generate_sample(label, index, gene_size=100):
    """ Generates images and targets via GAN """

    path = opt.gan_dir + '/GAN_MNIST.pt'
    generator, _ = get_trained_generator_discriminator(path)
    if cuda:
        generator.cuda()

    data = torch.empty(0).to(device)
    targets = torch.empty(0, dtype=int).to(device)
    repeat = int(gene_size / 100)
    for i in range(repeat):
        gene_data = generate_img(generator, label, index)
        data = torch.cat([data, gene_data], dim=0)

        targets = torch.cat([targets, torch.from_numpy(np.ones(100, dtype=int) * label).to(device)], dim=0)
    return data, targets


def get_trained_generator_discriminator(path):
    generator = g.Generator()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    discriminator = d.Discriminator()
    discriminator.load_state_dict(checkpoint['discriminator'])
    discriminator.eval()
    return generator, discriminator


def generate_img(generator, label, label_index, n_row=10):
    """ Returns generative result """

    static_z, static_label, static_code = get_specific_label_gen_input(label_index)
    z = Variable(torch.FloatTensor(np.random.normal(0, 0.5, (n_row ** 2, opt.latent_dim))))
    z = z.to(device)
    sample = generator(z, static_label, static_code)

    img_dir = opt.gene_img_dir
    os.makedirs(img_dir, exist_ok=True)
    name = opt.gene_img_dir + '/gene_label_{}_index_{}.png'.format(label, label_index)
    save_image(sample.data, name, nrow=n_row, normalize=True)

    return sample


def get_specific_label_gen_input(label_index):
    """ Returns specific static label """

    static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
    static_label = to_categorical(
        np.ones(100, dtype=int).dot(label_index), num_columns=opt.n_classes
    )
    static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))
    return static_z, static_label, static_code


def get_specific_label(label, size):
    """ Returns specific label with specific number """

    torch.manual_seed(0)
    np.random.seed(0)

    os.makedirs(opt.data_dir, exist_ok=True)
    train_set = datasets.MNIST(root=opt.data_dir, download=True, train=True, transform=transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ))

    idx = (train_set.targets == label)
    train_set.targets = train_set.targets[idx]
    train_set.data = train_set.data[idx]

    # randomly select data
    indices = torch.randperm(len(train_set.targets))[:size]
    train_set.targets = train_set.targets[indices]
    train_set.data = train_set.data[indices]

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set.targets))

    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    return images, labels


def shuffle(data, targets, seed=32):
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(targets)
    return data, targets


def print_report(model, test_loader, report_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """ Print classification report """

    pred_labels = []
    true_labels = []
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        pred_labels = pred_labels + predicted.numpy().tolist()
        true_labels = true_labels + labels.numpy().tolist()

    report = classification_report(true_labels, pred_labels, labels=report_labels)
    print(report)


def initial_parameters(dataset):
    """ Initial parameters
    Args:
        dataset: name of dataset

    Returns:
        labels: classes in dataset
        nums: number of each class
        gene_indexes: relationship between generative label and index
        gene_nums: number of each class that will generate
    """

    if dataset == 'fashion-mnist':
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        nums = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
        gene_indexes = [7, 6, 9, 8, 0, 4, 5, 3, 1, 2]
        gene_nums = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        return labels, nums, gene_indexes, gene_nums
    else:
        # mnist
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        nums = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

        gene_indexes = [6, 1, 4, 9, 0, 8, 5, 2, 7, 3]
        gene_nums = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

        return labels, nums, gene_indexes, gene_nums


def initial_fashion_imbalanced_data(labels, nums, gene_indexes, gene_nums):

    images = torch.empty(0).to(device)
    targets = torch.empty(0, dtype=int).to(device)
    length = len(labels)
    for i in range(length):
        label = labels[i]
        size = nums[i]
        gene_index = gene_indexes[i]
        gene_num = gene_nums[i]

        temp_images, temp_labels = get_fashion_specific_label(label, size)
        temp_gene_images, temp_gene_labels = generate_fashion_sample(label, gene_index, gene_num)

        images = torch.cat([images, temp_images, temp_gene_images], dim=0)
        targets = torch.cat([targets, temp_labels, temp_gene_labels], dim=0)

    if cuda:
        images, targets = images.cpu(), targets.cpu()
    x, y = shuffle(images.detach().numpy(), targets.detach().numpy())
    return torch.from_numpy(x), torch.from_numpy(y)


def get_fashion_specific_label(label, size):
    train_set = datasets.FashionMNIST(root=opt.fashion_data_dir, download=True, train=True, transform=transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ))

    idx = (train_set.targets == label)
    train_set.targets = train_set.targets[idx]
    train_set.data = train_set.data[idx]

    indices = torch.randperm(len(train_set.targets))[:size]
    train_set.targets = train_set.targets[indices]
    train_set.data = train_set.data[indices]

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set.targets))

    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    return images.to(device), labels.to(device)


def generate_fashion_sample(label, index, gene_num=100):
    """
    generate images, labels via GAN
    """

    # set code_dim
    opt.code_dim = 3

    generator, _ = get_trained_generator_discriminator(opt.gan_dir + '/GAN_Fashion_MNIST.pt')
    if cuda:
        generator.cuda()

    data = torch.empty(0).to(device)
    targets = torch.empty(0, dtype=int).to(device)
    repeat = int(gene_num / 100)
    for i in range(repeat):
        gene_data = generate_fashion_img(generator, label, index)
        data = torch.cat([data, gene_data], dim=0)

        targets = torch.cat([targets, torch.from_numpy(np.ones(100, dtype=int) * label).to(device)], dim=0)
    return data, targets


def generate_fashion_img(generator, label, label_index, n_row=10, factor=4.9):
    """ Returns generative result """

    static_z, static_label, static_code = get_specific_label_gen_input(label_index)
    z = Variable(torch.FloatTensor(np.random.normal(0, 0.2, (n_row ** 2, opt.latent_dim))))
    z = z.to(device)

    zeros = np.zeros((n_row ** 2, 1))
    c_varied = torch.randn(100, 1) * factor

    # code
    l = [zeros] * opt.code_dim
    l[2] = c_varied
    code = Variable(torch.FloatTensor(np.concatenate(tuple(l), -1)))
    code = code.to(device)

    sample = generator(z, static_label, code)

    # save image
    img_dir = opt.gene_img_dir
    os.makedirs(img_dir, exist_ok=True)
    name = opt.gene_img_dir + '/gene_fashion_label_{}_index_{}.png'.format(label, label_index)
    save_image(sample.data, name, nrow=n_row, normalize=True)

    return sample
