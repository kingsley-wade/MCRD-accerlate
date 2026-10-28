"""Check approval metadata and completion markers across the project specs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "specs/dna-p53-mcrd-accelerated"


def main() -> int:
    checks = {
        "requirements.md": ("Status: `approved`", "Requirements approved by user: `yes`"),
        "design.md": ("Status: `approved`", "Design approved by user: `yes`"),
        "tasks.md": ("Status: `complete`",),
    }
    failures: list[str] = []
    for name, required in checks.items():
        path = SPEC_DIR / name
        if not path.is_file():
            failures.append(f"missing spec: {path.relative_to(ROOT)}")
            continue
        content = path.read_text()
        for marker in required:
            if marker not in content:
                failures.append(f"{name}: missing {marker}")

    tasks_path = SPEC_DIR / "tasks.md"
    if tasks_path.is_file():
        task_markers = re.findall(r"^- \[([ x~!])\] \d", tasks_path.read_text(), re.MULTILINE)
        if not task_markers:
            failures.append("tasks.md: no numbered checklist items found")
        elif any(marker != "x" for marker in task_markers):
            failures.append("tasks.md: one or more numbered tasks are not complete")

    if failures:
        print("Spec validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Spec validation passed: approved requirements/design and complete task checklist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
