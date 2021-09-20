nohup python server.py --client 3 --model fl-agcns --dataset cora 2>&1 >> log/server.log &
nohup python client.py --client 3 --model fl-agcns 2>&1 >> log/client.log &