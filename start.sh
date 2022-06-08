gcn模型性能验证:
python server.py --client 8 --mode eval --gcn_model Gat --dataset Physics
python client.py --client 8 --mode eval

code验证:
python server.py --client 8 --mode test --code 2 11 5 20 7 0 47 2 --dataset Physics
python client.py --client 8 --mode test
2 9 5 8 0 3 7 5
agcns:3 5 0 7 0 42 0 4(best code)
rd:2 9 5 0 8 0 0 4
rd:4 5 0 3 0 53 69 1
rd:4 1 21 0 13 26 24 2
agcns:2 9 0 0 0 0 37 4

python server.py --client 3 --mode test --code 1 10 6 5 0 0 12 2 --dataset cora
python client.py --client 3 --mode test
graphnas|cora:5 11 0 1 12 4 9 4
fednas|cora:5 9 19 19 0 38 50
graphnas|citeseer:2 4 9 11 0 8 8 4
fednas|citeseer:5 9 0 0 0 11 11 2
graphnas|pubmed:4 7 0 10 10 7 2 3
fednas|pubmed:5 7 0 0 0 22 10 2
graphnas|corafull:3 7 6 0 5 3 2 2
fednas|corafull:4 7 19 0 19 19 10 2
graphnas|physics:2 5 6 8 2 11 3 5
fednas|physics:2 7 20 0 11 60 11 2




random模型:
python server.py --client 3 --model fl-random --dataset cora
python client.py --client 3 --model fl-random

nohup python -u server.py --client 8 --model fl-random --dataset Physics 2>&1 >> log/fl-random_physics_server.log &
nohup python client.py --client 8 --model fl-random 2>&1 >> log/fl-random_physics_client.log &


history result:
fednas:
python -u server.py --mode test --code 5 9 19 19 11 38 50 1 --dataset cora
python -u server.py --mode test --code 5 10 9 11 21 11 11 2 --dataset citeseer
python -u server.py --mode test --code 3 10 10 10 46 22 10 2 --dataset pubmed
python -u server.py --mode test --code 4 7 19 17 19 19 10 2 --dataset corafull

python -u server.py --client 8 --mode test --code 2 7 20 11 11 60 11 2 --dataset physics
python client.py  --client 8 --mode test

agcns:
python -u server.py --mode test --code 5 5 19 31 8 51 61 5 --dataset cora
python -u server.py --mode test --code 2 7 19 0 32 54 55 1 --dataset citeseer
python -u server.py --mode test --code 5 7 19 31 19 31 0 2 --dataset pubmed
python -u server.py --mode test --code 2 9 19 0 26 0 41 5 --dataset corafull

python -u server.py --client 8 --mode test --code 2 9 5 8 0 3 7 5 --dataset physics
python client.py  --client 8 --mode test

python client.py --mode test
