import argparse
from HGT_imp import run_hgt
from HAN_imp import run_han
from RGCN_imp import run_rgcn


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device', default='0', type=str,
                    help='Set the device (GPU ids)')
parser.add_argument('-da', '--dataset', default='B-dataset', type=str,
                    choices=['B-dataset', 'C-dataset', 'F-dataset'],
                    help='Set the data set for training')
parser.add_argument('-sp', '--saved_path', default='saved', type=str,
                    help='Path to save training results')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed to be used')
parser.add_argument('-mo', '--model', default='RGCN', type=str,
                    choices=['HGT', 'HAN', 'RGCN'],
                    help='Baseline model to be used')
# Training Arguments
parser.add_argument('-pr', '--print_every', default=10, type=int,
                    help='The number of epochs to print a training record')
parser.add_argument('-fo', '--nfold', default=10, type=int,
                    help='The number of k in k-fold cross validation')
parser.add_argument('-ep', '--epoch', default=1000, type=int,
                    help='The number of epochs for model training')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    help='Learning rate to be used in optimizer')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to be used')


args = parser.parse_args()
args.saved_path = '../result/' + args.model + '/' + args.saved_path + '_' + str(args.seed)

if args.model == 'HGT':
    run_hgt(args)
elif args.model == 'HAN':
    run_han(args)
elif args.model == 'RGCN':
    run_rgcn(args)