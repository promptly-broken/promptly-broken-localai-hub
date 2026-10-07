import json
import re
from collections import Counter

d = json.load(open("eval_results/v3_diag.json"))["details"]
cats = {2: Counter(), 3: Counter()}
ex = {}


def norm(s):
    return re.sub(r"\s+", " ", s.replace("'", "").replace("[", "").replace("]", "")).strip()


for r in d:
    n, g, o = r["n"], r["generated"], r["oracle"]
    if g == o:
        c = "exact"
    elif not g.startswith("<scratch>\n"):
        c = "format_drift_at_open"
    elif norm(g) == norm(o):
        c = "format_only_other"
    else:
        g_hit, o_hit = "hit" in g, "hit" in o
        gl, ol = g.split("\n"), o.split("\n")
        if g_hit and not o_hit:
            c = "hallucinated_hit"
        elif o_hit and not g_hit:
            c = "missed_hit"
        elif len(gl) != len(ol):
            c = "wrong_structure_or_length"
        else:
            c = "wrong_values_or_hit_text"
    cats[n][c] += 1
    ex.setdefault((n, c), r)

for n in (2, 3):
    sub = [r for r in d if r["n"] == n]
    print(f"size_{n}: {dict(cats[n])}")
    print(f"   starts '<scratch>\\n': {sum(r['generated'].startswith('<scratch>' + chr(10)) for r in sub)}/{len(sub)}"
          f" | starts \"<scratch>['\": {sum(r['generated'].startswith('<scratch>' + chr(91) + chr(39)) for r in sub)}/{len(sub)}")

for key in [(2, "hallucinated_hit"), (2, "wrong_values_or_hit_text"), (3, "wrong_values_or_hit_text"),
            (3, "hallucinated_hit"), (3, "missed_hit"), (3, "wrong_structure_or_length")]:
    if key in ex:
        r = ex[key]
        print(f"\n=== {key} state={r['state']} label={r['label']} ===\nORACLE:\n{r['oracle']}\nGENERATED:\n{r['generated']}")
