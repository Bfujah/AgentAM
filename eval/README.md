# AgentAM agent-evaluation kit

How to test the LLM agent (the layer between the user and the surrogate models), separate from the surrogate's physics accuracy.

## Files
- `make_benchmark.py`: builds `benchmark.jsonl` / `benchmark.csv`, 145 prompts in 14 categories (C1 to C14).
- `run_agent.py`: runs every prompt N times (default 5) through Claude plus your spatter MCP server and logs everything to `runs/<model>.jsonl`.
- `score_agent.py`: computes the ground truth by calling the same MCP server directly (no LLM), scores the runs, and writes `results/summary.csv`, `results/table_agent_eval.tex` and `results/review_sheet.csv`.

## Protocol (what to report in the paper)
1. **Fix the configuration.** Record the model string (e.g. `claude-opus-4-5`), the date, the temperature, the system prompt, the MCP server commit hash and the tool schemas.
2. **Run with tools.** `python run_agent.py --server "python path/to/server.py" --repeats 5`.
3. **Run the baselines.**
   - Same model without tools: `--no-tools`. This measures how often numbers are invented.
   - Optional: a second model through the same MCP server, e.g. an open-weight model behind an MCP bridge.
4. **Score.** `python score_agent.py --server "..." --runs runs/<model>.jsonl`.
5. **Human rating.** Two people independently fill `rater1` and `rater2` (pass/fail) in `review_sheet.csv` for C8 to C14. Report agreement as Cohen's kappa, and settle disagreements by discussion.
6. **Report.**
   - Per-category scores: tool selection, parameter extraction, numerical faithfulness, behaviour (clamping disclosure, unsupported inputs, infeasibility, materials), constraint satisfaction and optimality gap for C7, and consistency across the 5 repeats.
   - Latency and tokens per query.

## Metric definitions
- **Tool selection:** the expected tool family was called (predict, line scan, grid scan or optimise). For C14, no tool should be called.
- **Parameter extraction:** the P and V passed to the predict tool match the intended values. The tolerance is 0 for explicit values, 1 to 2 units for unit conversions, and the stated range for colloquial prompts.
- **Faithfulness:** the spatter value and defect class in the answer match the tool output to within 1%.
- **Constraint satisfaction (C7):** the recommended (P, V), re-evaluated by the server, is Good and meets every constraint.
- **Optimality gap:** (spatter of the recommended point − spatter of the brute-force optimum) / spatter of the brute-force optimum, with the optimum taken from a 46×46 grid search.
- **Consistency:** all 5 repeats used the same (P, V) and reported the same defect class.

Edit `PREDICT_TOOL`, `FAMILY` and `ARG_P`/`ARG_V` at the top of `score_agent.py` if your tool or argument names differ.

## Appendix transcript figures
The appendix figures are real agent runs, typeset from the run log (nothing is retyped by hand).
1. Run the 15 appendix prompts once each (same model and system prompt as the benchmark):
   `python run_agent.py --server "python ../server/server.py" --bench transcripts.jsonl --repeats 1 --out runs/transcripts.jsonl`
2. Render them: `python make_transcript_figs.py --runs runs/transcripts.jsonl --outdir ../figs/transcripts`
   (needs pdflatex with standalone, tcolorbox, booktabs, adjustbox). Each run becomes `figs/transcripts/<id>.pdf`.
Answers longer than `--max-lines` (default 34) are cut, with a note giving the number of lines left out.
