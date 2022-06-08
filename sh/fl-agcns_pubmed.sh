nohup python server.py --client 3 --model fl-agcns --dataset pubmed 2>&1 >> log/fl-agcns_pubmed_server.log &
nohup python client.py --client 3 --model fl-agcns 2>&1 >> log/fl-agcns_pubmed_client.log &