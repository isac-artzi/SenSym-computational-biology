# Progress logs

One markdown file per course week, from the template:

```bash
cp progress/TEMPLATE.md progress/week03.md
```

Fill it in at the end of the week (hours, what was derived, what was run —
with seeds — bead trials completed, where you're stuck, wins, questions, the
"Tools used" disclosure line), then commit it. The biweekly check-in issue
links to the two most recent logs, so keeping them current is what makes
check-ins fast.

Some experiment scripts also write into this directory (e.g.
`week09_dropout.py` writes its breakdown table to `week09_table.md`) —
commit those alongside your own notes.

## Why logs, and not just commits

A commit says *what changed*. A log says *what you were trying to find out,
what happened, and what you now believe* — which is the only part that ends
up in the report. Fourteen weeks from now, "why did we switch to the
negative binomial in week 11?" is a question only the log can answer.
