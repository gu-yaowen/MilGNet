import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# General Arguments
parser.add_argument('-id', '--device_id', default='0', type=str,
                    help='Set the device (GPU ids)')
parser.add_argument('-da', '--dataset', default='B-dataset', type=str,
                    # choices=['B-dataset', 'C-dataset', 'F-dataset', 'REDDA'],
                    help='Set the data set for training')
parser.add_argument('-sp', '--saved_path', default='saved', type=str,
                    help='Path to save training results')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Global random seed to be used')
# Training Arguments
parser.add_argument('-pr', '--print_every', default=10, type=int,
                    help='The number of epochs to print a training record')
parser.add_argument('-fo', '--nfold', default=10, type=int,
                    help='The number of k in k-fold cross validation')
parser.add_argument('-ep', '--epoch', default=500, type=int,
                    help='The number of epochs for model training')
parser.add_argument('-bs', '--batch_size', default=128, type=int,
                    help='The size of a batch to be used')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
                    help='Learning rate to be used in optimizer')
parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                    help='weight decay to be used')
parser.add_argument('-ck', '--check_metric', default='loss', type=str,
                    choices=['loss', 'auc', 'aupr'],
                    help='weight decay to be used')
# Model Argument
parser.add_argument('-k', '--k', default=20, type=int,
                    help='The number of topk similarities to be binarized')
parser.add_argument('-ag', '--aggregate_type', default='mean', type=str,
                    choices=['sum', 'mean', 'Linear', 'BiTrans'],
                    help='The type of aggregator to be used for aggregating meta-path instances')
parser.add_argument('-tk', '--topk', default=3, type=int,
                    help='The topk instance predictions to be chosen')
parser.add_argument('-hf', '--hidden_feats', default=128, type=int,
                    help='The dimension of hidden tensor in the model')
parser.add_argument('-nl', '--num_layer', default=2, type=int,
                    help='The number of graph embedding layers to be used')
parser.add_argument('-dp', '--dropout', default=0., type=float,
                    help='The rate of dropout layer')
parser.add_argument('-bn', '--batch_norm', action='store_true',
                    help='The rate of dropout layer')
parser.add_argument('-sk', '--skip', default=False, type=bool,
                    help='')
parser.add_argument('-mil', '--mil', default=False, type=bool,
                    help='')
parser.add_argument('-ip', '--ins_predict', default=False, type=bool,
                    help='')

args = parser.parse_args()
args.saved_path = 'result/' + args.dataset + '/' + args.saved_path + '_' + str(args.seed)
