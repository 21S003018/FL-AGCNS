gcn模型性能验证:
python server.py --client 8 --gcn_model Gat --dataset Physics
python client.py --client 8

code验证:
python server.py --client 8 --mode test --code 2 11 5 20 7 0 47 2 --dataset Physics
python client.py --client 8
2 9 5 8 0 3 7 5
python server.py --client 3 --mode test --code 1 10 6 5 8 0 12 2 --dataset cora
python client.py --client 3 --mode test
graphnas|cora:1 10 6 5 8 0 12 2

random模型:
python server.py --client 3 --model fl-random --dataset cora
python client.py --client 3 --model fl-random

nohup python -u server.py --client 8 --model fl-random --dataset Physics 2>&1 >> log/fl-random_physics_server.log &
nohup python client.py --client 8 --model fl-random 2>&1 >> log/fl-random_physics_client.log &
