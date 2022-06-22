import os
import torch
import argparse
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default='fashion')
parser.add_argument('-pretrained-model', type=str, default='')
parser.add_argument('-out', type=str, default='results.csv')

parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-minibatch', type=int, default=16)

parser.add_argument('-iters-total', type=int, default=300000)
parser.add_argument('-seeds', type=str, default='1')  # e.g., 1,2,3

parser.add_argument('-iters-per-round', type=int, default=5)
parser.add_argument('-iters-per-eval', type=int, default=250)

parser.add_argument('-similarity', type=float, default=0.05)
parser.add_argument('-total-workers', type=int, default=250)
parser.add_argument('-sampled-workers', type=int, default=10)

parser.add_argument('-gpu', type=int, default=1)  # 1 - use GPU; 0 - do not use GPU
parser.add_argument('-cuda-device', type=int, default=0)

parser.add_argument('-permute', type=int, default=1)

parser.add_argument('-save-checkpoint', type=int, default=1)
parser.add_argument('-iters-checkpoint', type=int, default=150000)

args = parser.parse_args()

print(', '.join(f'{k}={v}' for k, v in vars(args).items()))


use_gpu = bool(args.gpu)
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda:' + str(args.cuda_device)) if use_gpu else torch.device('cpu')

use_permute = bool(args.permute)

save_checkpoint = bool(args.save_checkpoint)
iters_checkpoint = args.iters_checkpoint

if args.data == 'fashion':
    dataset = 'FashionMNIST'
    model_name = 'ModelCNNMnist'
elif args.data == 'cifar' or args.data == 'cifar10':
    dataset = 'CIFAR10'
    model_name = 'ModelCNNCifar10'
else:
    raise Exception('Unknown data name')

max_iter = args.iters_total

simulations_str = args.seeds.split(',')
simulations = [int(i) for i in simulations_str]

dataset_file_path = os.path.join(os.path.dirname(__file__), 'data_files')

mixing_ratio = args.similarity

n_nodes = args.total_workers
n_nodes_in_each_round = args.sampled_workers
step_size = args.lr  # learning rate of clients

batch_size_train = args.minibatch
batch_size_eval = 256

iters_per_round = args.iters_per_round  # number of iterations in local training
min_iters_per_eval = args.iters_per_eval

results_file = args.out
save_model_file = results_file + '.model'
if args.pretrained_model != '':
    load_model_file = args.pretrained_model
else:
    load_model_file = None

if dataset == 'CIFAR10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform_train = None