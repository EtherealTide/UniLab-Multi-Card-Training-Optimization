"""Validate published evidence without CUDA, network access or third-party packages."""
import ast
import hashlib
import json
from pathlib import Path
import re
import statistics
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    archive_path = ROOT / "artifacts/sac_20261002_evidence.zip"
    expected_archive = "76d0121e4456d441b658c4f3568c8d2af811eaaf631cee543f5fff1de3e3bdff"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == expected_archive, "Archive checksum mismatch"
    manifest = json.loads((ROOT / "artifacts/evidence_manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive_path) as archive:
        for name, details in manifest.items():
            content = archive.read(name)
            assert len(content) == details["bytes"], name
            assert hashlib.sha256(content).hexdigest() == details["sha256"], name
        assert archive.read("unilab_rl_optimization.patch") == (ROOT / "patches/unilab_rl_optimization.patch").read_bytes()
    for path in (ROOT / "scripts").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    summary = json.loads((ROOT / "results/2026-10-03/summary.json").read_text(encoding="utf-8"))
    patterns = {
        "auto_cpu_single": "final_r*_n1_execution.json",
        "auto_cpu_dual": "final_r*_n2_shm_execution.json",
        "pool64_single": "pool_final_r*_n1_execution.json",
        "pool32_single": "local32_r*_n1_execution.json",
        "pool32_per_rank_dual": "pool_final_r*_n2_shm_execution.json",
    }
    raw = ROOT / "results/2026-10-03/raw"
    for group, pattern in patterns.items():
        executions = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw.glob(pattern))]
        assert len(executions) == 3, group
        assert all(record["exit_code"] == 0 for record in executions), group
        values = [record["actual_tail_fps"] for record in executions]
        assert abs(statistics.mean(values) - summary["groups"][group]["mean"]) < 0.001, group
    ratio = summary["groups"]["pool32_per_rank_dual"]["mean"] / summary["groups"]["pool32_single"]["mean"]
    assert abs(ratio - summary["speedup_against_local32"]) < 1e-12
    for doc in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            assert (doc.parent / target.split("#")[0]).exists(), (doc.name, target)
    print(json.dumps({"archive_items_verified": len(manifest), "successful_runs_reconciled": 15,
        "patch_matches_original": True, "python_syntax": "passed", "local_document_links": "passed",
        "speedup_against_local32": ratio}, indent=2))


if __name__ == "__main__":
    main()
