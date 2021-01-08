## environment

soft：[torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)

hard：4 GPU

## usage

#### server end

```text
usage: server.py [-h] [--mode MODE]
                 [--code SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK]
                 [--model {fl-rl,fl-darts,fl-agcns,fl-random}]
                 [--dataset {citeseer,physics,pubmed,corafull,cora}]
                 [--client CLIENT] [--save_dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           test mode or not
  --code SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK
                        code of son net
  --model {fl-rl,fl-darts,fl-agcns,fl-random}
                        search model
  --dataset {citeseer,physics,pubmed,corafull,cora}
                        used dataset
  --client CLIENT       the number of clients in the search
  --save_dir SAVE_DIR   the directory to save the best code and best population
```

#### client end

```text
usage: client.py [-h] [--mode MODE]
                 [--model {fl-random,fl-darts,fl-agcns,fl-rl}]
                 [--client CLIENT]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           test mode or not
  --model {fl-random,fl-darts,fl-agcns,fl-rl}
                        search model
  --client CLIENT       the number of clients in the search
```

## example

We prepare cora dataset in this repository and use cora as example.

```python
1.python server.py --mode test --code 5 0 0 0 0 0 0 2 --dataset cora
2.python client.py --mode test
```

Please open two terminals, input the first command line into one terminal and input the second one into another terminal.

## note

1. If `connection refused`，please run test.py to unify the ip_port.
2. run server.py before client.py