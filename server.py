from federal import *
from models import *
import argparse
import os
parser = argparse.ArgumentParser(description='server end usage')

parser.add_argument('--mode',
                    dest='mode',
                    action='store',
                    default=None,
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
                    choices={'fl-agcns', 'fl-random',
                             'fl-darts', "fl-graphnas", "fl-fednas"},
                    default='fl-agcns',
                    help='search model')
parser.add_argument('--gcn_model',
                    dest='gcn_model',
                    default=None,
                    choices={'Gat', 'Sage', 'Gcn', 'Sgc', 'Appnp', 'Agnn', 'Arma', 'Gated', 'Gnn'})
parser.add_argument('--dataset',
                    dest='dataset',
                    action='store',
                    choices={'cora', 'citeseer',
                             'pubmed', 'corafull', 'Physics', 'SBM'},
                    default='cora',
                    help='used dataset')
parser.add_argument('--client',
                    dest='client',
                    action='store',
                    type=int,
                    default=3,
                    help='the number of clients in the search')
parser.add_argument('--save_dir',
                    dest='save_dir',
                    action='store',
                    default='',
                    help='the directory to save the best code and best population')
args = parser.parse_args()

nfeat, nclass = 0, 0
if args.dataset == 'cora':
    nfeat, nclass = 1433, 7
elif args.dataset == 'citeseer':
    nfeat, nclass = 3703, 6
elif args.dataset == 'pubmed':
    nfeat, nclass = 500, 3
elif args.dataset == 'corafull':
    nfeat, nclass = 8710, 70
elif args.dataset == 'Physics':
    nfeat, nclass = 8415, 5
elif args.dataset == 'SBM':
    nfeat, nclass = 6, 6

controller = None
if __name__ == '__main__':
    # evaluate a specific code
    if args.mode == "eval":  # eval gcn models
        model = args.gcn_model
        controller = ControllerCommonNet(args.client)
        controller.configure(model, args.dataset, nfeat, nclass)
        res = controller.work(epochs=50)
        print('evaluate {} on {}, get the result as:\n{}'.format(
            args.gcn_model, args.dataset, res))
        controller.broadcast_with_waiting_res('ending')
        controller.close()
    elif args.mode == 'test':
        if args.supermask == None:
            print('there is not code')
            pass
        with open('tmp.pkl', 'wb') as f:
            pickle.dump(args.supermask, f)
        controller = ControllerCommonNet(args.client)
        controller.configure('SonNet', args.dataset, nfeat, nclass)
        res = controller.work(epochs=50)
        print('Test on sonnet of {}, get the result as\n{}'.format(args.dataset, res))
        controller.broadcast_with_waiting_res('ending')
        controller.close()
    elif args.model == 'fl-random':
        print(args.model)
        begin = time.time()
        for i in range(50):
            tmp_supermask = utils.random_supermask()
            with open('tmp.pkl', 'wb') as f:
                pickle.dump(tmp_supermask, f)
            controller = ControllerCommonNet(args.client)
            controller.configure('SonNet', args.dataset, nfeat, nclass)
            res = controller.work(epochs=50)
            print('Random search {}th rounds on {} with code:{}, get the result as\n{}'.
                  format(i, args.dataset, tmp_supermask, res))
            controller.broadcast_with_waiting_res('ending')
            controller.close()
        print("use time: {}".format(time.time() - begin))
        pass
    elif args.model == "fl-graphnas":
        model = GraphNas(INPUT_SIZE)
        if torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_GRAPHNAS)
        model.train()

        for i in range(EPOCHS):
            begin = time.time()
            dummy_code = model.generate_code()
            # for code in dummy_code:
            #     print(code.data[0])
            supermask = model.parse_code(dummy_code)
            # with open('tmp.pkl', 'wb') as f:
            #     pickle.dump(supermask, f)
            # print('Dataset:{}~Supermask:{}'.format(
            #     args.dataset, supermask))
            # print("generate time long:{}".format(time.time() - begin))
            controller = ControllerCommonNet(args.client)
            controller.configure('SonNet', args.dataset, nfeat, nclass)
            res = controller.work(epochs=50)
            print('Epoch:{}~Dataset:{}~Supermask:{}\nresult as\n{}'.format(i+1,
                                                                           args.dataset, supermask, res))

            controller.broadcast_with_waiting_res('val')
            controller.broadcast('get')
            R = 0
            accus = controller.aggregate()
            for idx in range(controller.num_client):
                R += float(accus[idx])

            loss = model.get_loss(dummy_code, supermask, R)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            controller.broadcast_with_waiting_res('ending')
            controller.close()
            print(f"time long:{time.time() - begin}\n")
            # break
        pass
    elif args.model == "fl-graphnas":

        pass
    else:
        if args.model == 'fl-agcns':
            controller = ControllerSuperNet(args.client)
            controller.configure('SuperNet', args.dataset, nfeat, nclass)
        elif args.model == 'fl-darts':
            controller = ControllerDarts(args.client)
            controller.configure('Darts', args.dataset, nfeat, nclass)
        elif args.model == "fl-fednas":
            controller = ControllerFedNas(args.client)
            controller.configure("FedNas", args.dataset, nfeat, nclass)
        controller.work()
        controller.broadcast_with_waiting_res('ending')
        controller.close()
        if args.model == 'fl-agcns':
            print('{} gives "opt supermask:{}" on {}.'.format(
                args.model, controller.output, args.dataset))
        elif args.model in ['fl-darts', "fl-fednas"]:
            print('{} gives "opt supermask:{}" on {}.'.format(
                args.model, controller.model.generate_supermask(), args.dataset))
        if not args.save_dir == '':
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            path = '{}/{}_optpop.pkl'.format(args.save_dir, args.dataset)
            with open(path, 'wb') as f:
                pickle.dump(controller.supermasks, f)
            path = '{}/{}_optson.pkl'.format(args.save_dir, args.dataset)
            with open(path, 'wb') as f:
                pickle.dump(controller.output, f)
