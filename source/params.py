import argparse

parser = argparse.ArgumentParser()

# directory
parser.add_argument("--data_dir", type=str, default='./../data/mnist',
                    help="the directory for MNIST dataset")
parser.add_argument("--fashion_data_dir", type=str, default='./../data/fashion',
                    help="the directory for FashionMNIST dataset")
parser.add_argument("--gan_dir", type=str, default='./../models/gan',
                    help="the directory for pre-trained GAN")
parser.add_argument("--cnn_dir", type=str, default='./../models/cnn',
                    help="the directory for CNN")
parser.add_argument("--gene_img_dir", type=str, default='./../gene_img/',
                    help="the directory storing temporal synthesized images")

parser.add_argument("--dataset", type=str, default='mnist',
                    choices=['mnist', 'fashion-mnist'], help="dataset")
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size")

# CNN parameters
parser.add_argument("--cnn_lr", type=float, default=0.0001,
                    help="the learning rate for CNN")
parser.add_argument("--epochs", type=int, default=8,
                    help="the number of epoch for CNN")

# GAN parameters
parser.add_argument("--n_epochs", type=int, default=500,
                    help="the number of epoch for GAN")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="the learning rate for GAN")
parser.add_argument("--b1", type=float, default=0.5,
                    help="b1 for ADAM")
parser.add_argument("--b2", type=float, default=0.999,
                    help="b2 for ADAM")
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=62,
                    help="the dimension for the noisy vector z")
parser.add_argument("--n_classes", type=int, default=10,
                    help="the number of classes(discrete variables)")
parser.add_argument("--code_dim", type=int, default=4,
                    help="the dimension for the latent code c(continual variables)")
parser.add_argument("--img_size", type=int, default=32,
                    help="image size")
parser.add_argument("--channels", type=int, default=1,
                    help="the size of CNN channel")
parser.add_argument("--sample_interval", type=int, default=500,
                    help="the number to generate temporal images when training")


opt = parser.parse_args()

# Loss weights
lambda_cat = 1
lambda_con = 0.15
