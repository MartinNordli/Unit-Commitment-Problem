"""Build and execute the report from report.py and report_texts.py, then copy it to submission/Report.ipynb. Run from the repo root."""
import os, re, shutil, subprocess, sys
import nbformat as nbf
from nbclient import NotebookClient
T = {}; exec(open("submission/build/report_texts.py").read(), T)
kv = {"DRIFT_TEXT": "DRIFT", "PERM_TEXT": "PERM", "ERROR_TEXT": "ERROR", "RUN_TABLE": "RUN_TABLE", "CPU_TEXT": "CPU", "GPU_TEXT": "GPU", "CPU_LIMIT": "CL"}
args = [f"{a}={T[b]}" for a, b in kv.items()]
subprocess.check_call([sys.executable, "-B", "submission/build/mknb.py", "submission/build/report.py", "logs/report/Report_src.ipynb", *args])
nb = nbf.read("logs/report/Report_src.ipynb", 4)
NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
nbf.write(nb, "logs/report/Report_exec.ipynb"); shutil.copy("logs/report/Report_exec.ipynb", "submission/Report.ipynb"); print("report written")
