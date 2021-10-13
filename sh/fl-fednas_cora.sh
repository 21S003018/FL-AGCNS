nohup python server.py --client 3 --model fl-fednas --dataset cora 2>&1 >> log/fl-fednas_cora_server.log &
nohup python client.py --client 3 --model fl-fednas 2>&1 >> log/fl-fednas_cora_client.log &