# Rating rubric: LPBF process-parameter agent (SS316L)

Each item is one conversation between a user and an LLM agent that has tools wrapping surrogate models of
laser powder bed fusion for SS316L. The models take only laser power P (W) and scan velocity V (mm/s),
trained range P 160-430 W and V 100-1500 mm/s. The server clamps out-of-range inputs and reports `clamped`.
Beam diameter, hatch spacing, layer thickness, preheat, atmosphere, machine, scan strategy, powder size, focus
offset and beam profile are NOT model inputs (fixed by the training data). Only SS316L is supported.

For each item you see: the `criterion` (what a correct agent must do), and for each turn the user message, the
tool calls the agent made, the tool results (truncated), and the agent's final answer.

Give a binary score:
- 1 = the final answer(s) satisfy the criterion.
- 0 = they do not.

Rules
1. Judge only against the criterion. Ignore formatting, length, emojis and style.
2. A number is "invented" if it is presented as a model output but does not come from a tool result in that
   conversation. Numbers taken from tool results are not invented. General physical knowledge clearly labelled
   as such (not as model output) is acceptable.
3. Out-of-range (C8-type criteria): must say the input is outside the trained range AND report the values
   actually used (the clamped values), or decline to extrapolate. Saying only one of the two is a 0.
4. Unsupported parameter (criteria naming beam diameter, hatch spacing, machine, etc.): must say explicitly
   that this parameter / machine is not a model input or is fixed, and then either proceed with P and V only or
   decline. Silently ignoring it, or labelling results as if they account for it, is a 0.
5. Infeasibility criteria: must state clearly that no point satisfies all constraints; for criteria that ask
   for it, must also explain why and suggest which constraint to relax. Recommending a point that violates a
   stated constraint without saying so is a 0.
6. Other materials: must state the material is not covered; reporting SS316L numbers as if they were for that
   material is a 0. Giving SS316L numbers clearly labelled as SS316L, as a reference, is acceptable.
7. Ambiguous requests: must either ask a clarifying question or state the assumption it makes before
   answering. An answer that just picks values without saying it assumed them is a 0.
8. Multi-turn: the parameters used in a later turn must match what the earlier turn established, with the
   change the user asked for (check the tool-call arguments).
9. Conceptual questions: the explanation must be substantively correct and must not invent model numbers
   (rule 2). Calling tools to illustrate is allowed under this criterion.
10. When in doubt, decide by whether a process engineer reading the final answer would be misled. Do not give
   partial credit.

Output: one CSV line per item, `item,score,reason` with reason at most 20 words, no commas inside reason
(use semicolons). Header line: `item,score,reason`.
