"""Typeset the agent transcripts captured as Claude Desktop screenshots (Spatter_Agentic_Real/*.png).

screenshot_transcripts.json holds the text of each screenshot transcribed verbatim (user prompt, tool
shown, agent answer). Only the presentation changes: dark UI -> light typeset panel. Where a screenshot
ends mid-answer the panel says so; nothing is completed or reworded.

  python make_screenshot_figs.py --outdir ../figs/transcripts
"""
import argparse, json, os, shutil, subprocess, tempfile
from make_transcript_figs import esc, inline, md2tex

HEAD = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{textcomp}\usepackage{mathptmx}\usepackage{amssymb}
\renewcommand{\ttdefault}{pcr}
\usepackage{xcolor}\usepackage{booktabs}\usepackage{adjustbox}\usepackage[most]{tcolorbox}
\definecolor{ink2}{HTML}{52514E}\definecolor{agent}{HTML}{3D3A6B}\definecolor{serverbg}{HTML}{EAF3EE}\definecolor{paper}{HTML}{F7F6F3}
\begin{document}
\begin{minipage}{%s}\footnotesize\raggedright
"""
USER = r"""\begin{tcolorbox}[colback=paper,colframe=ink2,boxrule=0.4pt,arc=4pt,left=4pt,right=4pt,top=2pt,bottom=2pt]
\raggedright{\scriptsize\bfseries\color{ink2}User}\par\noindent %s
\end{tcolorbox}\vspace{-1mm}
"""
TOOL = r"""\begin{tcolorbox}[colback=serverbg,colframe=ink2,boxrule=0.3pt,arc=2pt,left=4pt,right=4pt,top=1pt,bottom=1pt]
\raggedright\scriptsize\textbf{Tool call:} \texttt{%s}%s%s
\end{tcolorbox}\vspace{-1mm}
"""
ANS = r"""\begin{tcolorbox}[colback=white,colframe=agent,boxrule=0.5pt,arc=2pt,left=4pt,right=4pt,top=2pt,bottom=3pt]
\raggedright\scriptsize{\bfseries\color{agent}Agent answer}\par %s
\end{tcolorbox}\vspace{-1mm}
"""
TAIL = r"""\par\vspace{1mm}{\tiny\color{ink2}%s}
\end{minipage}
\end{document}
"""

def build(t, width, max_lines):
    body = []
    for b in t["blocks"]:
        if b["type"] == "user": body.append(USER % inline(b["text"]))
        elif b["type"] == "tool":
            args = "\\,(" + esc(", ".join(f"{k}={v}" for k, v in b["args"].items())) + ")" if b["args"] else ""
            note = r"\\[1pt]{\tiny\itshape\color{ink2}" + esc(b["note"]) + "}" if b.get("note") else ""
            body.append(TOOL % (esc(b["name"]), args, note))
        elif b["type"] == "answer":
            txt = md2tex(b["text"], max_lines)
            if b.get("cut"): txt += r"\,{\color{ink2}\itshape [screenshot ends here]}"
            body.append(ANS % txt)
    foot = f"Transcribed verbatim from the Claude Desktop session (Claude Opus 4.5), screenshot {t['source']}"
    if t.get("note"): foot += f"; {t['note']}"
    return HEAD % width + "".join(body) + TAIL % esc(foot + ".")

def compile_to(tex, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, name + ".tex"), "w", encoding="utf-8").write(tex)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=tmp, capture_output=True, text=True)
        ok = r.returncode == 0
        if ok: shutil.copyfile(os.path.join(tmp, name + ".pdf"), os.path.join(outdir, name + ".pdf"))
        else: print("\n".join(l for l in r.stdout.splitlines() if l.startswith("!"))[:400])
    print(("ok  " if ok else "FAIL"), name + ".pdf"); return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="screenshot_transcripts.json"); ap.add_argument("--outdir", default="../figs/transcripts")
    ap.add_argument("--width", default="9cm"); ap.add_argument("--wide", default="12cm", help="width for answers with wide tables")
    ap.add_argument("--max-lines", type=int, default=60); ap.add_argument("--ids", default="")
    a = ap.parse_args(); want = set(filter(None, a.ids.split(",")))
    for t in json.load(open(a.data, encoding="utf-8")):
        if want and t["id"] not in want: continue
        wide = any(b["type"] == "answer" and max((l.count("|") for l in b["text"].splitlines()), default=0) > 7 for b in t["blocks"])
        compile_to(build(t, a.wide if wide else a.width, a.max_lines), t["id"], a.outdir)
