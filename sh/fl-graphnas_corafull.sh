nohup python -u server.py --client 3 --model fl-graphnas --dataset corafull 2>&1 >> log/fl-graphnas_corafull_server.log &
nohup python client.py --client 3 --model fl-graphnas 2>&1 >> log/fl-graphnas_corafull_client.log &