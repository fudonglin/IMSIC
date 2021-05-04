import os
import numpy as np
import itertools
from torch.autograd import Variable
import torch
import utils
import params


# reproducibility
def experiment():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cuda = utils.cuda
    opt = params.opt

    folder_im = "./images_%f_%f" % (opt.code_dim, params.lambda_con)
    os.makedirs(folder_im)

    # create folders based on code dimension
    os.makedirs(folder_im + "/static/", exist_ok=True)
    for i in range(opt.code_dim):
        name = folder_im + '/varying_c' + str(i + 1) + '/'
        os.makedirs(name, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    generator, discriminator, adversarial_loss, categorical_loss, continuous_loss = utils.init_GAN()

    # completed data
    # dataloader = utils.get_MNIST_train_loader()

    # imbalanced data
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sizes = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    dataloader = utils.get_imbalance_dataloader(labels, sizes)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(FloatTensor))

            # generator loss
            optimizer_G.zero_grad()
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = utils.to_categorical(np.random.randint(0, opt.n_classes, batch_size),
                                               num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))
            gen_imgs = generator(z, label_input, code_input)
            validity, _, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # discriminator loss
            optimizer_D.zero_grad()
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # structure loss
            optimizer_info.zero_grad()
            sampled_labels = np.random.randint(0, opt.n_classes, batch_size)
            gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            label_input = utils.to_categorical(sampled_labels, num_columns=opt.n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))
            gen_imgs = generator(z, label_input, code_input)
            _, pred_label, pred_code = discriminator(gen_imgs)

            negative_edges = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2]]
            positive_edges = [[0, 0], [1, 1], [2, 2], [0, 3], [1, 3], [2, 3], [3, 3]]
            structure_loss = utils.get_structure_loss(continuous_loss, code_input, pred_code, negative_edges,
                                                      positive_edges)
            info_loss = params.lambda_cat * categorical_loss(pred_label, gt_labels) + params.lambda_con * structure_loss
            info_loss.backward()
            optimizer_info.step()

            if i == len(dataloader) - 1:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                      % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item()))
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                utils.sample_image2(generator=generator, n_row=10, batches_done=batches_done, folder_im=folder_im)

        if (epoch + 1) % 100 == 0:
            gan_dir = opt.gan_dir
            os.makedirs(gan_dir, exist_ok=True)
            torch.save({'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'parameters': opt}, gan_dir + '/model_final_{}'.format(epoch + 1))
