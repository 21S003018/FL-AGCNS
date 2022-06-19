## Environment

soft: [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)
hard: A6000*4

## Run Federated Learning model
Use the command in `cmd.sh`.<br>
Please open two terminals, input the first command line into one terminal and input the second one into another terminal.

## Evaluate GNN model and generetad GNN model

1. Before run the example, you should prepare data first. In this step, you could refer to [SBM generating method](https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/data/SBMs) and you would get three sets named `SBM_CLUSTER_train.pkl`,`SBM_CLUSTER_val.pkl` and `SBM_CLUSTER_test.pkl`, which should be placed under `data/SBM/`.
2. Run `python make_data.py`, and you could get partitioned datasets.

## Example

We have searched some light GCN model, which are recorded in `sbm.py`, you could set
```python
supermask = [5, 7, 4+12*1, 8+12*2, 2+12*3, 6+12*1, 2+12*2, 5]
train(supermask)
```
Run `python sbm.py` directly.

## Note

Note that SBM could be partitioned withour message losss, therefore we evaluate it locally.
