# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-synthetic test runner
# MAGIC
# MAGIC Installs the package and runs the full pytest suite, including the DP tests.

# COMMAND ----------

# MAGIC %pip install pyvinecopulib scipy numpy polars pytest

# COMMAND ----------

import subprocess
import sys
import os

# Install the library from workspace files
os.makedirs("/tmp/insurance_synthetic_pkg/src/insurance_synthetic", exist_ok=True)
os.makedirs("/tmp/insurance_synthetic_pkg/tests", exist_ok=True)

# Copy source files
import shutil

workspace_src = "/Workspace/Users/pricing.frontier@gmail.com/insurance-synthetic/src/insurance_synthetic"
dest_src = "/tmp/insurance_synthetic_pkg/src/insurance_synthetic"

for fname in [
    "__init__.py", "_marginals.py", "_copula.py", "_synthesiser.py",
    "_fidelity.py", "_schemas.py", "dp.py", "py.typed",
]:
    src_path = f"{workspace_src}/{fname}"
    dst_path = f"{dest_src}/{fname}"
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {fname}")
    else:
        print(f"Skipped {fname} (not found)")

# Write pyproject.toml
with open("/tmp/insurance_synthetic_pkg/pyproject.toml", "w") as f:
    f.write("""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "insurance-synthetic"
version = "0.1.7"
requires-python = ">=3.10"
dependencies = ["numpy>=1.21", "polars>=0.20", "scipy>=1.10", "pyvinecopulib>=0.6"]

[tool.hatch.build.targets.wheel]
packages = ["src/insurance_synthetic"]
""")

# Install in editable mode
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/insurance_synthetic_pkg", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("pip install failed")
print("Package installed successfully")

# COMMAND ----------

# Copy test files
workspace_tests = "/Workspace/Users/pricing.frontier@gmail.com/insurance-synthetic/tests"
dest_tests = "/tmp/insurance_synthetic_pkg/tests"

for fname in [
    "conftest.py", "test_marginals.py", "test_copula.py",
    "test_synthesiser.py", "test_fidelity.py", "test_schemas.py",
    "test_generate_validation.py", "test_regressions_p0.py", "test_dp.py",
]:
    src = f"{workspace_tests}/{fname}"
    if os.path.exists(src):
        shutil.copy(src, f"{dest_tests}/{fname}")
        print(f"Copied {fname}")
    else:
        print(f"Skipped {fname} (not found)")

# COMMAND ----------

# Run full test suite
result = subprocess.run(
    [sys.executable, "-m", "pytest", "/tmp/insurance_synthetic_pkg/tests/", "-v", "--tb=short", "-x"],
    capture_output=True, text=True, cwd="/tmp/insurance_synthetic_pkg"
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-3000:])

if result.returncode != 0:
    raise RuntimeError(f"Tests failed (exit code {result.returncode})")
else:
    print("\nAll tests passed")
