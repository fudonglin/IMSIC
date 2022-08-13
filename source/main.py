
import train_cnn as cnn
from source import params
import eval


def main():
    args = params.opt

    if args.evaluate:
        eval.evaluate(args.path, args.dataset)
    else:
        cnn.train()


if __name__ == '__main__':
    main()
