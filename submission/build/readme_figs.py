"""Copy the README figures out of the executed report. Usage: readme_figs.py [report.ipynb] [out dir]. Run from the repo root."""
import base64, json, os, sys
src = sys.argv[1] if len(sys.argv) > 1 else "submission/Report.ipynb"; out = sys.argv[2] if len(sys.argv) > 2 else "docs/img"
FIGS = {"pipeline": "How a prediction is made", "margin": "Share of hours on against price minus upstream", "models": "With the LP, every model scores",
        "week": "Final blend P(on)", "units": "What learning adds to the LP"}  # name -> text found only in that figure's code cell
cells = json.load(open(src))["cells"]
for name, mark in FIGS.items():
    hit = [c for c in cells if c["cell_type"] == "code" and mark in "".join(c["source"])]
    assert len(hit) == 1, (name, len(hit))
    png = [o["data"]["image/png"] for o in hit[0].get("outputs", []) if "image/png" in o.get("data", {})]
    assert png, f"{name}: no figure output"
    with open(os.path.join(out, name + ".png"), "wb") as f: f.write(base64.b64decode("".join(png[0])))
    print(name, "written")
