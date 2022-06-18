This branch is for `cora` `citeseer` `pubmed` `corafull` `Physics`, if running `SBM` goto [SBM](https://github.com/21S003018/FL-AGCNS/tree/sbm)
## Environment

soft：[torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)

hard：many GPUs

## Run Federated Learning model
Let's take `cora` dataset and our fl model as example.
```bash
nohup python -u server.py --client 3 --model fl-agcns --dataset cora 2>&1 >> log/fl-agcns_sbm_server.log &
nohup python -u client.py --client 3 --model fl-agcns 2>&1 >> log/fl-agcns_sbm_client.log &
```
Please open two terminals, input the first command line into one terminal and input the second one into another terminal.<br>
And the concete commands are as follows

**server end**

```text
usage: server.py [-h] [--mode MODE]
                 [--code SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK SUPERMASK]
                 [--model {fl-darts,fl-agcns,fl-random,fl-graphnas,fl-fednas}]
                 [--dataset {citeseer,Physics,pubmed,corafull,cora}]
                 [--client CLIENT]
```

**client end**

```text
usage: client.py [-h] [--mode MODE]
                 [--model {fl-darts,fl-agcns,fl-random,fl-graphnas,fl-fednas}]
                 [--client CLIENT]
```

## Example
We have searched some good architectures and you could evaluate it using the following command.<br>

```bash
# cora
python server.py --mode test --code ? --dataset cora
python client.py --mode test
# citeseer
python server.py --mode test --code ? --dataset citeseer
python client.py --mode test
# pubmed
python server.py --mode test --client 12 --code ? --dataset pubmed
python client.py --mode test --client 12
# corafull
python server.py --mode test --client 16 --code ? --dataset pubmed
python client.py --mode test --client 16
# Physics
python server.py --mode test --client 20 --code ? --dataset Physics
python client.py --mode test --client 20
```

Please open two terminals, input the first command line into one terminal and input the second one into another terminal.

## Note
1. Maybe you should run `python make_data.py` first to prepare the dataset.
2. If `connection refused`，please run reset.py to unify the ip_port.
3. run server.py before client.py