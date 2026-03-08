"""
Upload insurance-synthetic to Databricks and run pytest via serverless compute.
"""

from __future__ import annotations

import os
import sys
import time
import base64
from pathlib import Path


def load_env(path: str) -> None:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()


load_env(os.path.expanduser("~/.config/burning-cost/databricks.env"))

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc, jobs

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/insurance-synthetic"

SRC_FILES = list((PROJECT_ROOT / "src" / "insurance_synthetic").glob("*.py"))
TEST_FILES = list((PROJECT_ROOT / "tests").glob("*.py"))


def upload_file(local_path: Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ws_svc.ImportFormat.AUTO,
        overwrite=True,
    )
    print(f"  Uploaded: {remote_path}")


print("Uploading files to Databricks workspace...")

for subpath in [
    WORKSPACE_PATH,
    f"{WORKSPACE_PATH}/src",
    f"{WORKSPACE_PATH}/src/insurance_synthetic",
    f"{WORKSPACE_PATH}/tests",
]:
    try:
        w.workspace.mkdirs(subpath)
    except Exception:
        pass

for f in SRC_FILES:
    upload_file(f, f"{WORKSPACE_PATH}/src/insurance_synthetic/{f.name}")

for f in TEST_FILES:
    upload_file(f, f"{WORKSPACE_PATH}/tests/{f.name}")

upload_file(PROJECT_ROOT / "pyproject.toml", f"{WORKSPACE_PATH}/pyproject.toml")

print("\nCreating test notebook...")

# We use subprocess pip with --only-binary=:all: to avoid any source compilation.
# pyvinecopulib has pre-built wheels for manylinux2014 x86_64 (which covers
# Databricks serverless). The %pip magic doesn't support --only-binary,
# so we use subprocess directly.
NOTEBOOK_CONTENT = r"""# Databricks notebook source
# MAGIC %pip install polars>=0.20 scipy>=1.10 numpy>=1.21 pytest

# COMMAND ----------

import sys, os, shutil, subprocess

# Attempt to install pyvinecopulib with pre-built binary only
result = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "pyvinecopulib>=0.6.0",
     "--only-binary=:all:",
     "--quiet"],
    capture_output=True, text=True
)
print("pyvinecopulib install:", "OK" if result.returncode == 0 else "FAILED (will use Gaussian fallback)")
if result.returncode != 0:
    print("pip stderr:", result.stderr[-1000:])
    os.environ["INSURANCE_SYNTHETIC_FORCE_FALLBACK"] = "1"

# COMMAND ----------

# Copy source files to /tmp where __pycache__ is allowed
for src_dir in ["/Workspace/insurance-synthetic/src", "/Workspace/insurance-synthetic/tests"]:
    dst_dir = src_dir.replace("/Workspace/insurance-synthetic", "/tmp/insurance-synthetic")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

shutil.copy("/Workspace/insurance-synthetic/pyproject.toml", "/tmp/insurance-synthetic/pyproject.toml")

# COMMAND ----------

env = os.environ.copy()
env["PYTHONPATH"] = "/tmp/insurance-synthetic/src:/tmp/insurance-synthetic:/tmp/insurance-synthetic/tests"
env["PYTHONDONTWRITEBYTECODE"] = "1"

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/tmp/insurance-synthetic/tests/",
     "-v", "--tb=long", "--no-header", "-p", "no:warnings",
     "--import-mode=importlib"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-synthetic",
    env=env,
)

output = result.stdout + "\nSTDERR:\n" + result.stderr
if len(output) > 20000:
    output = output[:10000] + "\n...[middle truncated]...\n" + output[-10000:]

dbutils.notebook.exit(output)
"""

encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=f"{WORKSPACE_PATH}/run_tests",
    content=encoded_nb,
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    overwrite=True,
)
print(f"  Uploaded: {WORKSPACE_PATH}/run_tests")

print("\nSubmitting test job (serverless)...")

result = w.api_client.do("POST", "/api/2.2/jobs/runs/submit", body={
    "run_name": "insurance-synthetic-tests",
    "tasks": [{
        "task_key": "run_tests",
        "notebook_task": {
            "notebook_path": f"{WORKSPACE_PATH}/run_tests",
        },
    }]
})

run_id = result["run_id"]
host = os.environ["DATABRICKS_HOST"].rstrip("/")
print(f"Job submitted: run_id={run_id}")
print(f"Watch at: {host}#job/runs/{run_id}")

print("\nWaiting for tests...")
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    state = run_state.state
    life_cycle = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    result_state = state.result_state.value if state.result_state else ""
    print(f"  {life_cycle} {result_state}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(30)

print("\n" + "=" * 60)

tasks = run_state.tasks or []
task_run_id = None
for t in sorted(tasks, key=lambda x: x.attempt_number or 0):
    task_run_id = t.run_id

if task_run_id is not None:
    try:
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output and output.notebook_output.result:
            pytest_output = output.notebook_output.result
            print(pytest_output)
        if output.error:
            print("Notebook error:", output.error)
        if output.error_trace:
            print("Error trace:", output.error_trace[-5000:])
    except Exception as e:
        print(f"Could not retrieve output: {e}")

if result_state != "SUCCESS":
    print(f"\nJob result: {result_state}")
    sys.exit(1)
else:
    print("\nTESTS PASSED")
