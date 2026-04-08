import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_files(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    photo_dir = Path(plan["photo_dir"]).resolve()
    staging_dir = photo_dir / "_photo_agent_staging"
    destination_root = staging_dir / "reviewed_moves"
    destination_root.mkdir(parents=True, exist_ok=True)

    moved = []
    skipped = []
    errors = []

    for item in plan.get("move_paths", []):
        src = Path(item).resolve()
        try:
            src.relative_to(photo_dir)
            try:
                src.relative_to(staging_dir)
                skipped.append({"source": str(src), "reason": "already_in_staging"})
                continue
            except ValueError:
                pass

            if not src.exists():
                skipped.append({"source": str(src), "reason": "missing"})
                continue

            rel = src.relative_to(photo_dir)
            dst = unique_destination(destination_root / rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append({"source": str(src), "destination": str(dst)})
        except Exception as exc:
            errors.append({"source": str(src), "error": repr(exc)})

    log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "plan": str(plan_path),
        "destination_root": str(destination_root),
        "moved_count": len(moved),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "moved": moved,
        "skipped": skipped,
        "errors": errors,
    }
    log_path = staging_dir / "move_log.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    log = move_files(Path(args.plan))
    print(f"destination_root={log['destination_root']}")
    print(f"moved_count={log['moved_count']}")
    print(f"skipped_count={log['skipped_count']}")
    print(f"error_count={log['error_count']}")
    if log["error_count"]:
        for error in log["errors"][:20]:
            print(f"ERROR {error['source']}: {error['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
