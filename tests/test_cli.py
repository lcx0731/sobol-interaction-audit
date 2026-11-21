import json
import subprocess
import sys


def test_installed_cli_runs_outside_source_and_rejects_bad_budget(tmp_path):
    args = [sys.executable, "-m", "interaction_audit.cli"]
    result = subprocess.run(args + ["--power", "6", "--repetitions", "4"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["reference"]["variance"] == "43/15"
    assert report["result"]["total_evaluations"] == 64*4*6
    bad = subprocess.run(args + ["--power", "0"], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode == 2 and "power" in bad.stderr
