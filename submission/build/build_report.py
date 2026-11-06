"""Build and execute the report from report.py, then copy it to submission/Report.ipynb. Run from the repo root."""
import os, shutil, subprocess, sys
import nbformat as nbf
from nbclient import NotebookClient
subprocess.check_call([sys.executable, "-B", "submission/build/mknb.py", "submission/build/report.py", "logs/report/Report_src.ipynb"])
nb = nbf.read("logs/report/Report_src.ipynb", 4)
NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
nbf.write(nb, "logs/report/Report_exec.ipynb"); shutil.copy("logs/report/Report_exec.ipynb", "submission/Report.ipynb"); print("report written")
