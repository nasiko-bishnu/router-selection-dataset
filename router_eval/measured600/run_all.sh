#!/bin/zsh
# Run every test on the measured 600, one after another, logging each to results/logs/<test>.log.
# The big local models (Laya, d3-flash, CLM's Qwen3-8B encoder) need the Mac GPU, so they run one at a time.
#
#   D3_VENV=<d3 env> CLM_VENV=<clm env> D3_PATH=<d3-flash snapshot> QWEN_PATH=<Qwen3-8B snapshot> \
#   CLM_CKPT=<CLM_v0.1-8B.pt> router_eval/measured600/run_all.sh
#
# d3 env: python 3.12 with transformers==5.17.0, torch, accelerate, safetensors, einops, numpy.
# CLM env: python 3.12 with the CLM package (github.com/Contrastive-LM/CLM) installed, torch, transformers,
#          fastapi, uvicorn, numpy, requests.
set -o pipefail
cd "$(dirname "$0")/../.."
M=router_eval/measured600
L=$M/results/logs
mkdir -p $L
quiet() { grep --line-buffered -vE 'Warning|warn|it/s|falling back' }

uv run -q $M/test_pr296.py 2>&1 | quiet | tee $L/pr296.log
uv run -q $M/test_pr292.py 2>&1 | quiet | tee $L/pr292.log
uv run -q $M/test_nasiko.py 2>&1 | quiet | tee $L/nasiko.log
uv run -q $M/test_jev.py 2>&1 | quiet | tee $L/jev.log
uv run -q $M/test_laya.py 2>&1 | quiet | tee $L/laya.log
HF_HUB_OFFLINE=1 $D3_VENV/bin/python $M/test_d3.py 2>&1 | quiet | tee $L/d3.log

# CLM: start the Qwen3-8B encoder, wait until it answers, test, then stop it
HF_HUB_OFFLINE=1 EMB_BATCH=2 $CLM_VENV/bin/python $M/clm_embed_server.py > $L/clm_embed_server.log 2>&1 &
SERVER=$!
until curl -s -m 2 http://127.0.0.1:8090/v1/models >/dev/null; do sleep 3; kill -0 $SERVER || exit 1; done
HF_HUB_OFFLINE=1 $CLM_VENV/bin/python $M/test_clm.py 2>&1 | quiet | tee $L/clm.log
kill $SERVER

python3 $M/harness.py
