"""Build docs/index.html from docs/template.html and the notebook outputs.

Run from the repository root after the notebook:  python docs/build_page.py
Reads outputs/filter_series.csv and report/results.json, embeds the data the page needs, and writes a standalone
docs/index.html (no external data files; fonts come from Google Fonts with system fallbacks).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def sig(v, digits=6):
    return float(f"{v:.{digits}g}")


def main():
    results = json.loads((ROOT / "report" / "results.json").read_text())
    with open(ROOT / "outputs" / "filter_series.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    est = results["estimates"]
    theta = {m: [est[m][k] for k in ("a", "mu", "phi", "sigma_x", "rho")] for m in ("none", "next", "same")}
    data = {
        "dates": [row["date"] for row in rows],
        "r": [sig(float(row["r"])) for row in rows],
        "rv": [sig(float(row["rv"])) for row in rows],
        "theta": theta,
        "rho_bootstrap": est["same, bootstrap estimate"]["rho"],
        "forecasts": results["forecasts"],
        "comparisons": results["comparisons_vs_same_day"],
        "by_year": {k: results["loglik_diff_by_year"][k] for k in ("same - none", "same - GJR")},
        "ess": results["ess"],
        "one_day_date": results["one_day"]["date"],
    }
    template = (DOCS / "template.html").read_text()
    assert template.count("__DATA__") == 1
    page = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    head, body = page.split("</style>", 1)
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}</style>\n</head>\n<body>\n{body.lstrip()}</body>\n</html>\n")
    (DOCS / "index.html").write_text(html)
    print(f"wrote docs/index.html ({len(html) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
