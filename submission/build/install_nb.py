"""Copy an executed clean-run notebook into submission/ and fill in its reference-run line. Usage: install_nb.py <n> <executed.ipynb> <text>"""
import sys, nbformat as nbf
n, src, text = sys.argv[1], sys.argv[2], sys.argv[3]
nb = nbf.read(src, 4); c = nb.cells[0]; old = "**Reference run:** filled in after the reference run"
assert c.cell_type == "markdown" and c.source.count(old) == 1, "header line not found"
c.source = c.source.replace(old, "**Reference run:** " + text); nbf.validate(nb); nbf.write(nb, f"submission/Short_notebook_{n}.ipynb"); print("installed", n)
