"""Typeset logged agent runs as transcript figures for the paper appendix.

Each figure shows, for one logged run (from run_agent.py): the user prompt, every tool call
with its arguments, an excerpt of each raw tool result, and the agent's final answer, copied
verbatim from the log (only Markdown is converted to LaTeX). Nothing is retyped by hand.

  python make_transcript_figs.py --runs runs/transcripts.jsonl --outdir ../figs/transcripts
  python make_transcript_figs.py --runs runs/claude-opus-4-5.jsonl --ids C1_explicit-02 --repeat 0 --outdir test

Needs pdflatex with the standalone, tcolorbox, booktabs and adjustbox packages.
"""
import argparse, json, os, re, shutil, subprocess, tempfile

SYM = {"µ": r"\textmu{}", "μ": r"\textmu{}", "³": r"\textsuperscript{3}", "²": r"\textsuperscript{2}", "°": r"\textdegree{}",
       "≤": r"$\le$", "≥": r"$\ge$", "×": r"$\times$", "→": r"$\rightarrow$", "←": r"$\leftarrow$", "≈": r"$\approx$",
       "±": r"$\pm$", "–": "--", "—": "---", "‘": "`", "’": "'", "“": "``", "”": "''", "…": r"\ldots{}", "•": r"\textbullet{}",
       "⁰": r"\textsuperscript{0}", "¹": r"\textsuperscript{1}", "⁴": r"\textsuperscript{4}", "⁵": r"\textsuperscript{5}",
       "⁶": r"\textsuperscript{6}", "Δ": r"$\Delta$", "∼": r"$\sim$", "~": r"$\sim$", "✓": r"$\checkmark$"}

def esc(t):
    out = []
    for ch in t:
        if ch in "\\": out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}": out.append("\\" + ch)
        elif ch == "^": out.append(r"\textasciicircum{}")
        elif ch in SYM: out.append(SYM[ch])
        elif ord(ch) < 128 or ch in "éèàüöä": out.append(ch)
        # emoji and other symbols are dropped (they carry no content in these answers)
    return "".join(out)

def inline(t):
    """escape, then restore **bold**, `code`, *italic*"""
    parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", t)
    res = []
    for p in parts:
        if p.startswith("**") and p.endswith("**") and len(p) > 4: res.append(r"\textbf{" + esc(p[2:-2]) + "}")
        elif p.startswith("`") and p.endswith("`") and len(p) > 2: res.append(r"\texttt{" + esc(p[1:-1]) + "}")
        else: res.append(esc(p))
    return "".join(res)

def md2tex(md, max_lines):
    lines = md.strip().splitlines(); out = []; i = 0; shown = 0; cut = False
    while i < len(lines):
        if shown >= max_lines: cut = True; break
        l = lines[i].rstrip()
        if l.strip().startswith("|"):                       # table block
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c): rows.append(cells)
                i += 1
            n = max(len(r) for r in rows)
            body = [" & ".join(inline(c) for c in r + [""] * (n - len(r))) + r" \\" for r in rows]
            if len(body) > 1: body.insert(1, r"\midrule")
            out.append(r"\par\smallskip\noindent\adjustbox{max width=\linewidth}{\begin{tabular}{" + "l" * n + "}\\toprule\n" + "\n".join(body) + "\n\\bottomrule\\end{tabular}}\\par\\smallskip")
            shown += len(rows); continue
        if not l.strip() or re.fullmatch(r"-{3,}|\*{3,}", l.strip()): out.append(r"\par\smallskip"); i += 1; continue
        m = re.match(r"#{1,6}\s+(.*)", l)
        if m: out.append(r"\par\smallskip\noindent\textbf{" + inline(m.group(1)) + r"}\par"); i += 1; shown += 1; continue
        m = re.match(r"(\s*)([-*]|\d+\.)\s+(.*)", l)
        if m:
            mark = r"\textbullet" if m.group(2) in "-*" else m.group(2)
            ind = 0.8 + 0.8 * (len(m.group(1)) // 2)
            out.append(r"\par\noindent\hangindent=%.1fem\hangafter=1\hspace*{%.1fem}%s\ %s" % (ind + 0.6, ind - 0.8, mark, inline(m.group(3))))
            i += 1; shown += 1; continue
        out.append(r"\par\noindent " + inline(l)); i += 1; shown += 1
    if cut: out.append(r"\par\smallskip\noindent{\color{gray}\itshape[answer continues; %d further lines in the run log]}" % (len(lines) - i))
    return "\n".join(out)

def result_excerpt(text, n=260):
    t = re.sub(r"\s+", " ", text)
    try:
        d = json.loads(text); d = d.get("data", d)
        if isinstance(d, dict): d = {k: v for k, v in d.items() if k != "meta"}
        t = json.dumps(d, separators=(", ", ": "))
        t = re.sub(r"(\d+\.\d{2})\d+", r"\1", t)            # shorten long floats for display only
    except Exception: pass
    return t if len(t) <= n else t[:n] + " ..."

TEMPLATE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{textcomp}\usepackage{mathptmx}\usepackage{amssymb}
\renewcommand{\ttdefault}{pcr}
\usepackage{xcolor}\usepackage{booktabs}\usepackage{adjustbox}\usepackage[most]{tcolorbox}
\definecolor{ink2}{HTML}{52514E}\definecolor{agent}{HTML}{3D3A6B}\definecolor{agentbg}{HTML}{EEEDF6}
\definecolor{serverbg}{HTML}{EAF3EE}\definecolor{paper}{HTML}{F7F6F3}
\begin{document}
\begin{minipage}{%(width)s}\footnotesize\raggedright
\begin{tcolorbox}[colback=paper,colframe=ink2,boxrule=0.4pt,arc=4pt,left=4pt,right=4pt,top=2pt,bottom=2pt]
\raggedright{\scriptsize\bfseries\color{ink2}User}\par\noindent %(prompt)s
\end{tcolorbox}\vspace{-1mm}
%(tools)s
\begin{tcolorbox}[colback=white,colframe=agent,boxrule=0.5pt,arc=2pt,left=4pt,right=4pt,top=2pt,bottom=3pt]
\raggedright\scriptsize{\bfseries\color{agent}Agent answer}\par %(answer)s
\end{tcolorbox}
{\tiny\color{ink2}%(footer)s}
\end{minipage}
\end{document}
"""
TOOLBOX = r"""\begin{tcolorbox}[colback=serverbg,colframe=ink2,boxrule=0.3pt,arc=2pt,left=4pt,right=4pt,top=1pt,bottom=1pt]
\raggedright\scriptsize\textbf{Tool call %(k)d:} \texttt{%(name)s}\,(%(args)s)\\[1pt]
{\tiny\ttfamily\color{ink2}%(res)s}
\end{tcolorbox}\vspace{-1mm}"""

def render(run, outdir, width, max_lines):
    turns = run["turns"]; prompt = " \\par\\noindent ".join(inline(t["user"]) for t in turns)
    calls = [(c, r) for t in turns for c, r in zip(t["tool_calls"], t["tool_results"])]
    tools = "\n".join(TOOLBOX % dict(k=i + 1, name=esc(c["name"]), args=esc(", ".join(f"{k}={v}" for k, v in c["args"].items())),
                                      res=esc(result_excerpt(r))) for i, (c, r) in enumerate(calls)) or \
            r"{\scriptsize\itshape\color{ink2} no tool called}\par\smallskip"
    footer = esc(f"{run['model']} | run {run['id']} (repeat {run['repeat']}) | {len(calls)} tool call(s), {run['latency_s']:.1f} s")
    tex = TEMPLATE % dict(width=width, prompt=prompt, tools=tools, answer=md2tex(turns[-1]["final_text"], max_lines), footer=footer)
    name = f"{run['id']}" + (f"_r{run['repeat']}" if run["repeat"] else "")
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:          # build outside outdir; only the PDF is copied over
        open(os.path.join(tmp, name + ".tex"), "w", encoding="utf-8").write(tex)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=tmp, capture_output=True, text=True)
        ok = r.returncode == 0 and os.path.exists(os.path.join(tmp, name + ".pdf"))
        if ok: shutil.copyfile(os.path.join(tmp, name + ".pdf"), os.path.join(outdir, name + ".pdf"))
        else:
            shutil.copyfile(os.path.join(tmp, name + ".tex"), os.path.join(outdir, name + ".failed.tex"))
            print("\n".join(l for l in r.stdout.splitlines() if l.startswith("!"))[:400])
    print(("ok  " if ok else "FAIL"), name + ".pdf")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--outdir", default="../figs/transcripts")
    ap.add_argument("--ids", default="", help="comma-separated run ids (default: all)")
    ap.add_argument("--repeat", type=int, default=None, help="only this repeat index")
    ap.add_argument("--width", default="9cm"); ap.add_argument("--max-lines", type=int, default=34)
    a = ap.parse_args(); want = set(filter(None, a.ids.split(",")))
    for run in map(json.loads, open(a.runs, encoding="utf-8")):
        if (not want or run["id"] in want) and (a.repeat is None or run["repeat"] == a.repeat):
            render(run, a.outdir, a.width, a.max_lines)
