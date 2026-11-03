"""Run the benchmark against the agent: Claude + the spatter MCP server.

Every prompt is run N_REPEATS times in a fresh conversation. All tool calls, tool
results, the final answer, latency and token use are logged to runs/<model>.jsonl.

  pip install anthropic mcp
  export ANTHROPIC_API_KEY=...
  python run_agent.py --server "python /path/to/spatter_mcp_server.py" --model claude-opus-4-5 --repeats 5

Use the SAME system prompt and model string as in the paper and report both.
For a baseline without tools, add --no-tools (tests hallucination of numbers).
"""
import argparse, asyncio, json, os, sys, shlex, time
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM = ("You are an LPBF process-parameter assistant for SS316L. Use the provided tools for every "
          "quantitative answer; never estimate model outputs yourself. If an input is outside the trained "
          "range or a parameter is not a model input, say so explicitly.")

async def run(args):
    client = Anthropic()
    cmd = [c.strip('"') for c in shlex.split(args.server, posix=(os.name != "nt"))]  # keep Windows backslashes
    if cmd and cmd[0].lower() in ("python", "python3", "python.exe"): cmd[0] = sys.executable  # same env as this script
    params = StdioServerParameters(command=cmd[0], args=cmd[1:])
    items = [json.loads(l) for l in open(args.bench)]
    if args.only:  # "C1" means C1 only (not C10-C14); comma-separated lists allowed, e.g. "C1,C7"
        want = [w.strip() for w in args.only.split(",")]
        items = [i for i in items if any(i["category"] == w or i["category"].startswith(w + "_") for w in want)]
    os.makedirs("runs", exist_ok=True)
    out = open(args.out or f"runs/{args.model}{'_notools' if args.no_tools else ''}.jsonl", "a")
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as mcp:
            await mcp.initialize()
            tools = [] if args.no_tools else [
                dict(name=t.name, description=t.description or "", input_schema=getattr(t, "input_schema", None) or getattr(t, "inputSchema", None))
                for t in (await mcp.list_tools()).tools]
            for it in items:
                turns = it.get("turns") or [it["prompt"]]
                for rep in range(args.repeats):
                    msgs, log = [], dict(id=it["id"], category=it["category"], repeat=rep, model=args.model,
                                         turns=[], t_start=time.time(), tokens_in=0, tokens_out=0)
                    for user in turns:
                        msgs.append(dict(role="user", content=user))
                        calls, results = [], []
                        for _ in range(args.max_steps):
                            resp = client.messages.create(model=args.model, max_tokens=2000, system=(args.system or SYSTEM) + (" " + args.system_extra if args.system_extra else ""),
                                                          tools=tools, messages=msgs, temperature=args.temperature)
                            log["tokens_in"] += resp.usage.input_tokens; log["tokens_out"] += resp.usage.output_tokens
                            msgs.append(dict(role="assistant", content=resp.content))
                            uses = [b for b in resp.content if b.type == "tool_use"]
                            if not uses: break
                            tr = []
                            for u in uses:
                                res = await mcp.call_tool(u.name, u.input)
                                text = "".join(getattr(c, "text", "") for c in res.content)
                                calls.append(dict(name=u.name, args=u.input)); results.append(text)
                                tr.append(dict(type="tool_result", tool_use_id=u.id, content=text))
                            msgs.append(dict(role="user", content=tr))
                        final = "".join(b.text for b in resp.content if b.type == "text")
                        log["turns"].append(dict(user=user, tool_calls=calls, tool_results=results, final_text=final))
                    log["latency_s"] = time.time() - log.pop("t_start")
                    out.write(json.dumps(log) + "\n"); out.flush()
                    print(it["id"], rep, f"{log['latency_s']:.1f}s", [c["name"] for t in log["turns"] for c in t["tool_calls"]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="command that starts the spatter MCP server (stdio)")
    ap.add_argument("--model", default="claude-opus-4-5")
    ap.add_argument("--bench", default="benchmark.jsonl")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0, help="report this value in the paper")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--only", default="", help="category prefix, e.g. C7")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--out", default="", help="output .jsonl (default runs/<model>.jsonl)")
    ap.add_argument("--system", default="", help="replace the system prompt entirely (e.g. the neutral baseline)")
    ap.add_argument("--system-extra", default="", help="text appended to the system prompt, e.g. a no-fabrication instruction")
    asyncio.run(run(ap.parse_args()))
