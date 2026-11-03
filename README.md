# AgentAM

Code and data for **"AgentAM: An AI Agent for Spatter- and Defect-Aware Process Optimization in Laser Powder Bed Fusion"**
(O. T. Ajenifujah and A. Barati Farimani, Carnegie Mellon University). Preprint: arXiv link to be added.

AgentAM gives an LLM agent natural-language access to CFD-trained polynomial surrogates of spatter and melt pool
geometry for SS316L laser powder bed fusion through the Model Context Protocol (MCP). The server predicts spatter and
melt pool depth, width and length, assigns a geometry-derived defect regime (keyhole, lack of fusion, balling or good),
scans the power–velocity window and searches for low-spatter parameters, including a margin-aware optimizer.
All surrogate evaluation runs locally; only parameter values and model outputs reach the LLM.

## Repository layout

| Folder | Contents |
|---|---|
| `server/` | MCP server (`server.py`), the six surrogate models (`*_poly_model.pkl`) and their coefficients (`surrogate_coefficients.json`) |
| `data/` | The 279 FLOW-3D single-track results (`Agentic_defect_Dataset.csv`) and the data behind the figures |
| `eval/` | 145-prompt benchmark (`benchmark.jsonl`), margin-aware prompts, agent runner, scoring, answer guard, cross-validation, robustness and simulation-budget studies |
| `eval/runs/` | Full logs of every agent run used in the paper (with tools, without tools, neutral prompts, margin-aware prompts) |
| `eval/results/` | Scores and tables produced by the scripts |
| `eval/rating/` | Rubric, blinded rating packets and the two raters' scores for the behaviour categories |
| `analyses/` | Reviewer analyses: threshold sensitivity, rule order, completeness, robustness variants, neutral-prompt baselines |
| `figures/` | Figure scripts, TikZ sources and the final figure PDFs |

## Install

```bash
python -m pip install -r requirements.txt
```

## Use the MCP server with an MCP client

Add the server to your client's MCP configuration, for example in Claude Desktop:

```json
{
  "mcpServers": {
    "agentam": { "command": "python", "args": ["/path/to/AgentAM/server/server.py"] }
  }
}
```

Then ask, for example: *"Find the lowest-spatter defect-free parameters that stay at least 15 µm above the
lack-of-fusion depth."* The models are valid only for SS316L in the trained range P = 160–430 W and
V = 100–1500 mm/s; inputs outside it are clamped and the clamping is reported.

## Reproduce the evaluation

Run from `eval/`:

```bash
# checks that do not need an LLM
python cv_validation.py --server_py ../server/server.py          # regime agreement and 5-fold CV
python robust_eval.py   --server_py ../server/server.py          # margin-aware vs plain optima
python active_learning.py                                         # simulation-budget study

# score the logged agent runs (no API calls)
python score_agent.py --server "python ../server/server.py" --runs runs/claude-opus-4-5.jsonl
python score_margin.py --server_py ../server/server.py --runs runs/margin.jsonl
python analyze_notools.py --runs runs/claude-opus-4-5_notools.jsonl
python answer_guard.py --runs runs/claude-opus-4-5_notools.jsonl

# re-run the agent (needs an Anthropic API key in ANTHROPIC_API_KEY; costs API credits)
python run_agent.py --server "python ../server/server.py" --repeats 5
python run_agent.py --no-tools --repeats 5
```

See `eval/README.md` for the protocol and metric definitions. Run `--help` on any script for its options.
The paper's runs used `claude-opus-4-5`; LLM outputs vary between runs and model versions, so re-runs will not match
the logged transcripts exactly.

## Citation

```bibtex
@article{ajenifujah2026agentam,
  title  = {AgentAM: An AI Agent for Spatter- and Defect-Aware Process Optimization in Laser Powder Bed Fusion},
  author = {Ajenifujah, Olabode T. and Barati Farimani, Amir},
  year   = {2026},
  note   = {arXiv preprint, identifier to be added}
}
```

## License

MIT (see `LICENSE`).
