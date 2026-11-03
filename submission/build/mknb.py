"""Turn a '# %%' source into a notebook. Usage: mknb.py src out KEY=VALUE..."""
import sys, re, nbformat as nbf
src, out, *kv = sys.argv[1:]
s = open(src).read()
for k in kv:
    a, b = k.split("=", 1); s = s.replace(f"@@{a}@@", b)
assert "@@" not in s, re.findall(r"@@\w+@@", s)
cells = []
for part in re.split(r"^# %%", s, flags=re.M)[1:]:
    head, _, body = part.partition("\n"); body = body.strip("\n")
    if "[markdown]" in head:
        cells.append(nbf.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))))
    else:
        cells.append(nbf.v4.new_code_cell(body))
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.validate(nb); nbf.write(nb, out); print(out, len(cells))
