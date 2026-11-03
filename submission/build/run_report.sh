#!/bin/bash
# Build and execute the report. Usage: run_report.sh KEY=VALUE ... with the text placeholders.
cd /cluster/home/martnal/sintef; V=/tmp/sintef_env_martnal/venv/bin/python; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
$V -B submission/build/mknb.py submission/build/report.py logs/report/Report_src.ipynb "$@" && $V -B - <<'P'
import nbformat as nbf, os
from nbclient import NotebookClient
nb = nbf.read("logs/report/Report_src.ipynb", 4)
NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
nbf.write(nb, "logs/report/Report_exec.ipynb"); print("ok", sum(len(c.get("outputs", [])) for c in nb.cells))
P
