from federal import *
import argparse
import os
parser = argparse.ArgumentParser(description='server end usage')

parser.add_argument('--mode',
                    dest='mode',
                    action='store',
                    # type=bool,
                    default='train',
                    help='test mode or not')
parser.add_argument('--code',
                    dest='supermask',
                    action='store',
                    type=int,
                    nargs=8,
                    default=None,
                    help='code of son net')
parser.add_argument('--model',
                    dest='model',
                    action='store',
                    choices={'fl-agcns','fl-rl','fl-random','fl-darts'},
                    default='fl-agcns',
                    help='search model')
parser.add_argument('--dataset',
                    dest='dataset',
                    action='store',
                    choices={'cora','citeseer','pubmed','corafull','physics'},
                    default='cora',
                    help='used dataset')
# parser.add_argument('--round',
#                     dest='round',
#                     action='store',
#                     type=int,
#                     default=250,
#                     help='the number of iterations in the search')
parser.add_argument('--client',
                    dest='client',
                    action='store',
                    type=int,
                    default=3,
                    help='the number of clients in the search')
# parser.add_argument('--lr',
#                     dest='lr',
#                     action='store',
#                     type=float,
#                     default=0.02,
#                     help='learning rate')
parser.add_argument('--save_dir',
                    dest='save_dir',
                    action='store',
                    default='',
                    help='the directory to save the best code and best population')
args = parser.parse_args()

nfeat, nclass = 0, 0
if args.dataset == 'cora':nfeat, nclass = 1433, 7
elif args.dataset == 'citeseer':nfeat, nclass = 3703, 6
elif args.dataset == 'pubmed':nfeat, nclass = 500, 3
elif args.dataset == 'corafull':nfeat, nclass = 8710, 70
elif args.dataset == 'physics':nfeat, nclass = 8415, 5

controller = None
if __name__ == '__main__':
    # evaluate a specific code
    if args.mode == 'test':
        if args.supermask == None:
            print('there is not code')
            pass
        with open('tmp.pkl','wb') as f:
            pickle.dump(args.supermask,f)
        controller = ControllerCommonNet(args.client)
        controller.configure('SonNet', args.dataset, nfeat, nclass)
        res = controller.work(epochs=50)
        print('Test on sonnet of {}, get the result as\n{}'.format(args.dataset, res))
        controller.broadcast_with_waiting_res('ending')
        controller.close()
    elif args.model == 'fl-random':
        for i in range(50):
            tmp_supermask = utils.random_supermask()
            with open('tmp.pkl', 'wb') as f:
                pickle.dump(tmp_supermask, f)
            controller = ControllerCommonNet(args.client)
            controller.configure('SonNet', args.dataset, nfeat, nclass)
            res = controller.work(epochs=50)
            print('Random search {}th rounds on {} with code:{}, get the result as\n{}'.
                  format(i,args.dataset,tmp_supermask,res))
            controller.broadcast_with_waiting_res('ending')
            controller.close()
        pass
    else:
        if args.model == 'fl-agcns':
            controller = ControllerSuperNet(args.client)
            controller.configure('SuperNet', args.dataset, nfeat, nclass)
        elif args.model == 'fl-darts':
            controller = ControllerDarts(args.client)
            controller.configure('Darts', args.dataset, nfeat, nclass)
        controller.work()
        controller.broadcast_with_waiting_res('ending')
        controller.close()
        if args.model == 'fl-agcns':print('{} gives "opt supermask:{}" on {}.'.format(args.model, controller.output, args.dataset))
        elif args.model == 'fl-darts':print('{} gives "opt supermask:{}" on {}.'.format(args.model, controller.model.generate_supermask(), args.dataset))
        if not args.save_dir == '':
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            path = '{}/{}_optpop.pkl'.format(args.save_dir,args.dataset)
            with open(path, 'wb') as f:
                pickle.dump(controller.supermasks, f)
            path = '{}/{}_optson.pkl'.format(args.save_dir,args.dataset)
            with open(path, 'wb') as f:
                pickle.dump(controller.output, f)
