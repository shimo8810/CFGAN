import argparse

import chainer

from dataset import get_movie_lens_100k
from network import Generator, Discriminator
from updater import CFGANZRUpdater

def main():
    parser = argparse.ArgumentParser(description='CFGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')

    parser.add_argument('--n_z', '-z', type=int, default=100,
                        help='Number of  random noise z')
    args = parser.parse_args()

    train, test = get_movie_lens_100k('./dataset/u1.base', './dataset/u1.test',
                './dataset/u.info', './dataset/u.user', './dataset/u.item')
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

if __name__ == '__main__':
    main()