import argparse


class Args(object):
    @staticmethod
    def get_num_class(dataset):
        num = {
            'cifar10': 10,
            'cifar100': 100
        }
        return num[dataset]


devices = ['cpu', 'cuda']
datasets = ['cifar10', 'cifar100', 'gtsrb']
models = ['resnet32', 'vgg16_bn', 'convstn']


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--data_dir', default='data')
parser.add_argument('--output_dir', default='output')
parser.add_argument('--device', default='cuda', choices=devices)
parser.add_argument('-b', '--batch_size', type=int, default=256)
parser.add_argument('-m', '--model', type=str, required=True, choices=models)
parser.add_argument('-d', '--dataset', type=str, required=True, choices=datasets)
parser.add_argument('--eval', action='store_true', help='whether to evaluate the trained model only')
parser.add_argument('--genesize', type=int, default=10)
parser.add_argument('--popsize', type=int, default=30)
parser.add_argument('--enable_filters', action='store_true', default=True, help='whether to enable advantaged muators')
parser.add_argument('--mutate_prob', type=float, default=0.6)

