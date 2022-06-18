# run fl model
fl-agcns:
nohup python -u server.py --client 20 --model fl-agcns --dataset SBM 2>&1 >> log/fl-agcns_sbm_server.log &
nohup python -u client.py --client 20 --model fl-agcns 2>&1 >> log/fl-agcns_sbm_client.log &

fl-random:
nohup python -u server.py --client 20 --model fl-random --dataset SBM 2>&1 >> log/fl-random_sbm_server.log &
nohup python -u client.py --client 20 --model fl-random 2>&1 >> log/fl-random_sbm_client.log &

fl-fednas:
nohup python -u server.py --client 20 --model fl-fednas --dataset SBM 2>&1 >> log/fl-fednas_sbm_server.log &
nohup python -u client.py --client 20 --model fl-fednas 2>&1 >> log/fl-fednas_sbm_client.log &

fl-darts:
nohup python -u server.py --client 20 --model fl-darts --dataset SBM 2>&1 >> log/fl-darts_sbm_server.log &
nohup python -u client.py --client 20 --model fl-darts 2>&1 >> log/fl-darts_sbm_client.log &

fl-graphnas:
nohup python -u server.py --client 20 --model fl-graphnas --dataset SBM 2>&1 >> log/fl-graphnas_sbm_server.log &
nohup python -u client.py --client 20 --model fl-graphnas 2>&1 >> log/fl-graphnas_sbm_client.log &

fl-rl:
nohup python -u rl.py 2>&1 >> log/fl-rl_sbm.log &
