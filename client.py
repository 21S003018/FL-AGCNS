import argparse
from federal import *
import torch.multiprocessing as mp
from models import EPOCHS


parser = argparse.ArgumentParser(description='client end usage')
parser.add_argument('--mode',
                    dest='mode',
                    action='store',
                    # type=bool,
                    default=None,
                    help='test mode or not')
parser.add_argument('--model',
                    dest='model',
                    action='store',
                    choices={'fl-agcns', 'fl-random',
                             'fl-darts', 'fl-graphnas', "fl-fednas"},
                    default='fl-agcns',
                    help='search model')
parser.add_argument('--client',
                    dest='client',
                    action='store',
                    type=int,
                    default=3,
                    help='the number of clients in the search')
args = parser.parse_args()

# start clients for evaluating a specific code
if __name__ == '__main__':
    mp.set_start_method('spawn')
    if args.mode == 'test' or args.mode == "eval":
        clients = []
        for j in range(args.client):
            clients.append(ClientCommonNet(j))
        processes = []
        for client in clients:
            process = mp.Process(target=client.work)
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    elif args.model == 'fl-random':
        for i in range(50):
            clients = []
            for j in range(args.client):
                clients.append(ClientCommonNet(j))
            processes = []
            for client in clients:
                process = mp.Process(target=client.work)
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
    elif args.model == "fl-graphnas":
        for i in range(EPOCHS):
            clients = []
            for j in range(args.client):
                clients.append(ClientCommonNet(j))
            processes = []
            for client in clients:
                process = mp.Process(target=client.work)
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            # break
    else:
        clients = []
        for i in range(args.client):
            if args.model == 'fl-agcns':
                clients.append(ClientSuperNet(i))
            elif args.model == 'fl-darts':
                clients.append(ClientDarts(i))
            elif args.model == 'fl-fednas':
                clients.append(ClientFedNas(i))
        processes = []
        for client in clients:
            process = mp.Process(target=client.work)
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
