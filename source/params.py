import argparse

parser = argparse.ArgumentParser()

# directory
parser.add_argument("--data_dir", type=str, default='./../data/mnist')
parser.add_argument("--fashion_data_dir", type=str, default='./../data/fashion')
parser.add_argument("--gan_dir", type=str, default='./../models/gan')
parser.add_argument("--cnn_dir", type=str, default='./../models/cnn')
parser.add_argument("--gene_img_dir", type=str, default='./../gene_img/')

# CNN parameters
parser.add_argument("--dataset", type=str, default='mnist')
parser.add_argument("--path", type=str, default='./../models/cnn/cnn_mnist.pt')
parser.add_argument("--cnn_lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.set_defaults(evaluate=True)

# GAN parameters
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=62)
parser.add_argument("--code_dim", type=int, default=4)
parser.add_argument("--n_classes", type=int, default=10)
parser.add_argument("--img_size", type=int, default=32)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--sample_interval", type=int, default=800)


opt = parser.parse_args()

# Loss weights
lambda_cat = 1
lambda_con = 0.15
