nohup python -u server.py --client 3 --model fl-agcns --dataset SBM 2>&1 >> log/fl-agcns_sbm_server.log &
nohup python -u client.py --client 3 --model fl-agcns 2>&1 >> log/fl-agcns_sbm_client.log &