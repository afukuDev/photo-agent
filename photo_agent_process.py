import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import faiss
import imagehash
import numpy as np
import pillow_heif
import torch
from PIL import Image, ImageOps
from sklearn.cluster import DBSCAN
from transformers import AutoImageProcessor, AutoModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic", ".heif"}
MODEL_NAME = "facebook/dinov2-large"


def iter_images(photo_dir: Path, staging_dir: Path) -> list[Path]:
    files = []
    staging_resolved = staging_dir.resolve()
    for path in photo_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            path.relative_to(staging_resolved)
            continue
        except ValueError:
            files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def phash_to_int(hash_value: imagehash.ImageHash) -> int:
    return int(str(hash_value), 16)


def hamming_int(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_quality(path: Path) -> tuple[float, float]:
    image = np.array(load_image(path))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    clipped = float(hist[:5].sum() + hist[251:].sum())
    mean = float(gray.mean())
    mean_balance = 1.0 - min(abs(mean - 127.5) / 127.5, 1.0)
    exposure_score = max(0.0, min(1.0, 0.7 * (1.0 - clipped) + 0.3 * mean_balance))
    return laplacian_var, exposure_score


def normalize_sharpness(values: list[float]) -> list[float]:
    if not values:
        return []
    logs = np.log1p(np.array(values, dtype=np.float32))
    low, high = np.percentile(logs, [5, 95])
    if math.isclose(float(low), float(high)):
        return [1.0 for _ in values]
    normalized = np.clip((logs - low) / (high - low), 0.0, 1.0)
    return normalized.tolist()


def connected_components(edges: list[tuple[int, int]], count: int) -> list[list[int]]:
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return [members for members in groups.values() if len(members) > 1]


def build_duplicate_groups(metadata: list[dict], threshold: int) -> list[list[int]]:
    edges = []
    hashes = [int(item["phash_int"]) for item in metadata]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming_int(hashes[i], hashes[j]) <= threshold:
                edges.append((i, j))
    return connected_components(edges, len(metadata))


def grouped_items(indices: list[int], metadata: list[dict], scores: list[float]) -> list[dict]:
    items = []
    for idx in indices:
        item = dict(metadata[idx])
        item["combined_score"] = scores[idx]
        items.append(item)
    return sorted(items, key=lambda x: x["combined_score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--phash-threshold", type=int, default=10)
    parser.add_argument("--dbscan-eps", type=float, default=0.15)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    pillow_heif.register_heif_opener()
    photo_dir = Path(args.photo_dir).resolve()
    staging_dir = photo_dir / "_photo_agent_staging"
    staging_dir.mkdir(exist_ok=True)
    (staging_dir / "duplicates").mkdir(exist_ok=True)
    (staging_dir / "similar_groups").mkdir(exist_ok=True)

    metadata_path = staging_dir / "metadata.json"
    embeddings_path = staging_dir / "embeddings.npy"
    faiss_path = staging_dir / "faiss.index"
    report_path = staging_dir / "analysis_report.json"
    ascii_work_dir = Path.cwd() / ".photo_agent_work"
    ascii_work_dir.mkdir(exist_ok=True)

    existing_metadata = []
    existing_embeddings = np.empty((0, 1024), dtype=np.float32)
    if metadata_path.exists() and embeddings_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_embeddings = np.load(embeddings_path)

    existing_by_key = {
        (item["path"], item["size"], item["mtime_ns"]): (i, item)
        for i, item in enumerate(existing_metadata)
    }

    image_paths = iter_images(photo_dir, staging_dir)
    print(f"images_found={len(image_paths)}", flush=True)

    metadata: list[dict] = []
    embeddings: list[np.ndarray] = []
    new_paths: list[Path] = []

    for path in image_paths:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        existing = existing_by_key.get(key)
        if existing:
            idx, item = existing
            metadata.append(item)
            embeddings.append(existing_embeddings[idx])
        else:
            new_paths.append(path)

    print(f"existing_reused={len(metadata)}", flush=True)
    print(f"new_to_process={len(new_paths)}", flush=True)

    if new_paths:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device={device}", flush=True)
        processor = AutoImageProcessor.from_pretrained(args.model_name, local_files_only=True)
        model = AutoModel.from_pretrained(args.model_name, local_files_only=True).to(device)
        model.eval()

        next_progress = 100
        for start in range(0, len(new_paths), args.batch_size):
            batch_paths = new_paths[start : start + args.batch_size]
            images = []
            batch_metadata = []
            for path in batch_paths:
                image = load_image(path)
                stat = path.stat()
                phash_int = phash_to_int(imagehash.phash(image))
                sharpness, exposure = image_quality(path)
                images.append(image)
                batch_metadata.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "suffix": path.suffix.lower(),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                        "phash_int": str(phash_int),
                        "sharpness": sharpness,
                        "exposure": exposure,
                    }
                )

            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
                vectors = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(vectors)

            metadata.extend(batch_metadata)
            embeddings.extend(vectors)

            processed = min(start + len(batch_paths), len(new_paths))
            if processed >= next_progress or processed == len(new_paths):
                print(f"processed_new={processed}/{len(new_paths)} total_indexed={len(metadata)}", flush=True)
                while next_progress <= processed:
                    next_progress += 100

    embedding_matrix = np.vstack(embeddings).astype(np.float32) if embeddings else np.empty((0, 1024), dtype=np.float32)
    if len(embedding_matrix):
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        safe_name = hashlib.sha1(str(photo_dir).encode("utf-8")).hexdigest()[:16] + ".index"
        ascii_faiss_path = ascii_work_dir / safe_name
        # FAISS on Windows can fail on non-ASCII paths, so write to an ASCII path first.
        faiss.write_index(index, str(ascii_faiss_path))
        shutil.copyfile(ascii_faiss_path, faiss_path)

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(embeddings_path, embedding_matrix)

    duplicate_indices = build_duplicate_groups(metadata, args.phash_threshold)
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=2, metric="cosine")
    labels = dbscan.fit_predict(embedding_matrix) if len(embedding_matrix) else np.array([])
    semantic_indices = []
    for label in sorted(set(labels.tolist()) - {-1}):
        members = np.where(labels == label)[0].tolist()
        if len(members) > 1:
            semantic_indices.append(members)

    sharpness_scores = normalize_sharpness([float(item["sharpness"]) for item in metadata])
    combined_scores = [
        0.7 * sharpness_scores[i] + 0.3 * float(metadata[i]["exposure"])
        for i in range(len(metadata))
    ]

    report = {
        "photo_dir": str(photo_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model_name,
        "image_count": len(metadata),
        "new_processed": len(new_paths),
        "faiss_index": str(faiss_path),
        "metadata": str(metadata_path),
        "embeddings": str(embeddings_path),
        "duplicate_group_count": len(duplicate_indices),
        "similar_group_count": len(semantic_indices),
        "duplicate_groups": [
            {"group_id": i + 1, "items": grouped_items(indices, metadata, combined_scores)}
            for i, indices in enumerate(duplicate_indices)
        ],
        "similar_groups": [
            {"group_id": i + 1, "items": grouped_items(indices, metadata, combined_scores)}
            for i, indices in enumerate(semantic_indices)
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"faiss_index={faiss_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"duplicate_group_count={len(duplicate_indices)}", flush=True)
    print(f"similar_group_count={len(semantic_indices)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
