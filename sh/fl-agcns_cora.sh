nohup python -u server.py --client 3 --model fl-agcns --dataset cora 2>&1 >> log/fl-agcns_cora_server.log &
nohup python client.py --client 3 --model fl-agcns 2>&1 >> log/fl-agcns_cora_client.log &