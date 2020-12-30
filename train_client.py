import argparse
from federal import *
import torch.multiprocessing as mp
parser = argparse.ArgumentParser(description='Search some files')
parser.add_argument('--model',
                    dest='model',
                    action='store',
                    choices={'fl-agcns','fl-rl','fl-random','fl-darts'},
                    default='fl-agcns',
                    help='search model')
parser.add_argument('--client',
                    dest='client',
                    action='store',
                    type=int,
                    default=3,
                    help='the number of clients in the search')
args = parser.parse_args()

clients = []
if args.model == 'fl-random':
    for i in range(50):
        for j in range(args.client):
            clients.append(ClientCommonNet(j))
        processes = []
        for client in clients:
            process = mp.Process(target=client.work)
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    pass

for i in range(args.client):
    if args.model == 'fl-agcns':clients.append(ClientSuperNet(i))
    elif args.model == 'fl-darts':clients.append(ClientDarts(i))
processes = []
for client in clients:
    process = mp.Process(target=client.work)
    process.start()
    processes.append(process)
for process in processes:
    process.join()
