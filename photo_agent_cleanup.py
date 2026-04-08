import argparse
import json
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def staged_image_files(staging_dir: Path) -> list[Path]:
    staging_dir = staging_dir.resolve()
    if not staging_dir.exists():
        return []
    return sorted(
        [path for path in staging_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: str(path).lower(),
    )


def delete_staged_images(staging_dir: Path, confirmation: str) -> dict:
    if confirmation != "yes":
        return {
            "deleted_count": 0,
            "deleted_bytes": 0,
            "aborted": True,
            "errors": [],
            "message": "confirmation_not_yes",
        }

    staging_dir = staging_dir.resolve()
    files = staged_image_files(staging_dir)
    deleted = []
    errors = []
    deleted_bytes = 0

    for path in files:
        try:
            path = path.resolve()
            path.relative_to(staging_dir)
            size = path.stat().st_size
            path.unlink()
            deleted.append(str(path))
            deleted_bytes += size
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})

    for directory in sorted([p for p in staging_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "staging_dir": str(staging_dir),
        "deleted_count": len(deleted),
        "deleted_bytes": deleted_bytes,
        "aborted": False,
        "deleted": deleted,
        "errors": errors,
    }
    (staging_dir / "delete_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()

    staging_dir = Path(args.staging_dir)
    files = staged_image_files(staging_dir)
    list_path = staging_dir / "pending_delete_list.txt"
    staging_dir.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(str(path) for path in files), encoding="utf-8")
    print(f"pending_delete_count={len(files)}")
    print(f"pending_delete_list={list_path}")

    result = delete_staged_images(staging_dir, args.confirm)
    print(f"aborted={result['aborted']}")
    print(f"deleted_count={result['deleted_count']}")
    print(f"deleted_bytes={result['deleted_bytes']}")
    print(f"error_count={len(result.get('errors', []))}")
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
