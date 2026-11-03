@echo off
REM Neutral-prompt no-tools baselines (reviewer point 3). Run from this folder in a terminal where
REM ANTHROPIC_API_KEY is set (set ANTHROPIC_API_KEY=...). Each variant = 145 prompts x 5 repeats = 725 calls.
cd /d "%~dp0..\eval"
set SERVER=python ..\server\server.py

REM Variant N1: neutral role only, no instruction about tools
python run_agent.py --server "%SERVER%" --no-tools --repeats 5 --temperature 1.0 ^
  --system "You are an assistant for laser powder bed fusion (LPBF) process parameters for SS316L stainless steel." ^
  --out runs/claude-opus-4-5_notools_neutral.jsonl

REM Variant N2: neutral role + explicit instruction to acknowledge an unavailable model (the reviewer's request)
python run_agent.py --server "%SERVER%" --no-tools --repeats 5 --temperature 1.0 ^
  --system "You are an assistant for laser powder bed fusion (LPBF) process parameters for SS316L stainless steel. If answering requires a predictive model or tool that is not available to you, say so explicitly and do not invent model outputs." ^
  --out runs/claude-opus-4-5_notools_ack.jsonl

REM Classify the answers (original baseline results are kept; new files get the suffix)
python analyze_notools.py --runs runs/claude-opus-4-5_notools_neutral.jsonl --tag _neutral
python analyze_notools.py --runs runs/claude-opus-4-5_notools_ack.jsonl --tag _ack
REM Apply the answer guard to the new answers
python answer_guard.py --runs runs/claude-opus-4-5_notools_neutral.jsonl runs/claude-opus-4-5_notools_ack.jsonl > results/answer_guard_neutral.txt
echo Done. Results in eval\results\notools_neutral_summary.csv, notools_ack_summary.csv and answer_guard_neutral.txt
pause
