# import argparse
# parser = argparse.ArgumentParser(description='Search some files')
#
# parser.add_argument('--test',
#                     dest='test',
#                     action='store',
#                     type=bool,
#                     default=False,
#                     help='test mode or not')
# parser.add_argument('--code',
#                     dest='supermask',
#                     action='store',
#                     type=int,
#                     nargs=8,
#                     help='code of son net')
# parser.add_argument('--model',
#                     dest='model',
#                     action='store',
#                     choices={'fl-agcns','fl-rl','fl-random','fl-darts'},
#                     default='fl-agcns',
#                     help='search model')
# parser.add_argument('--dataset',
#                     dest='dataset',
#                     action='store',
#                     choices={'cora','citeseer','pubmed','corafull','physics'},
#                     default='cora',
#                     help='used dataset')
# parser.add_argument('--round',
#                     dest='round',
#                     action='store',
#                     type=int,
#                     default=250,
#                     help='the number of iterations in the search')
# parser.add_argument('--client',
#                     dest='client',
#                     action='store',
#                     type=int,
#                     default=3,
#                     help='the number of clients in the search')
# parser.add_argument('--lr',
#                     dest='lr',
#                     action='store',
#                     type=float,
#                     default=0.02,
#                     help='learning rate')
# parser.add_argument('--save_dir',
#                     dest='save_dir',
#                     action='store',
#                     default='',
#                     help='the directory to save the best code and best population')
# args = parser.parse_args()
#
# print(args.supermask)
import utils
utils.reset_ipport()
# print(1)
# pass
# print(2)