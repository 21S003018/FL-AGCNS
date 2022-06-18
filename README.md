## Environment

soft：[torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)

## Run Federated Learning model
Use the command in `cmd.sh`.<br>
Please open two terminals, input the first command line into one terminal and input the second one into another terminal.

## Evaluate GNN model and generetad GNN model

1. Before run the example, you should prepare data first. In this step, you could refer to [SBM generating method](https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/data/SBMs) and you would get three sets named `SBM_CLUSTER_train.pkl`,`SBM_CLUSTER_val.pkl` and `SBM_CLUSTER_test.pkl`, which should be placed under `data/SBM/`.
2. Run `python make_data.py`, and you could get partitioned datasets.

Run `python sbm.py` directly. Note that SBM could be partitioned withour message losss, therefore we evaluate it locally.
