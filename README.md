# Photo Agent

Photo Agent is a desktop tool for organizing large local photo libraries, especially iPhone backups that contain HEIC/HEIF images, duplicate shots, and many near-identical photos. It combines pHash, DINOv2, FAISS, DBSCAN, and a PyQt6 review workflow so users can group photos first, review them safely, and only then move unwanted files into a staging area.

MakerHub article:
[Photo Agent：用 DINOv2、pHash 與 FAISS 整理 iPhone HEIC 相簿](https://makerhub.qianpro.shop/posts/photo-agent-dinov2-phash-faiss-iphone-heic)

## Highlights

- Supports HEIC / HEIF / JPG / PNG and other common local image formats
- Detects near-duplicate photos with pHash
- Finds semantically similar photos with DINOv2 embeddings
- Stores normalized vectors in a FAISS `IndexFlatIP` index for cosine-like similarity search
- Uses DBSCAN to cluster similar images into reviewable groups
- Scores each photo with sharpness and exposure, then recommends the best shot
- Uses a staging-first safety flow instead of deleting original files directly
- Provides both an integrated GUI and a standalone review GUI

## Project Structure

- `photo_agent_app.py`
  Integrated GUI. Lets the user choose a folder, configure indexing settings, start indexing, review groups, save a move plan, move files to staging, and delete staged images with confirmation.
- `photo_agent_process.py`
  CLI indexer. Reads images, computes pHash, sharpness, exposure, and DINOv2 embeddings, writes metadata and reports, builds the FAISS index, and produces grouped analysis results.
- `photo_agent_review.py`
  Standalone review GUI for an existing `analysis_report.json`.
- `photo_agent_move.py`
  Applies `move_plan.json` and moves files into `_photo_agent_staging/reviewed_moves`.
- `photo_agent_cleanup.py`
  Deletes staged image files only after exact `yes` confirmation, then writes `delete_log.json`.
- `photo_agent_architecture.png` / `photo_agent_architecture.svg`
  Architecture diagram assets for documentation or release notes.
- `release/README.md`
  Release cover and screenshot checklist.
- `.codex/skills/photo-agent-maintainer/SKILL.md`
  Repo-local skill for future maintenance and documentation work.

## How It Works

Photo Agent currently has three layers: indexing, review, and post-processing.

### 1. Indexing

The indexer scans the target photo directory and skips `_photo_agent_staging`. For each image it computes:

- pHash
- sharpness score
- exposure score
- DINOv2 embedding

Embeddings are L2-normalized and stored in a FAISS `IndexFlatIP` index. Metadata, embeddings, and generated reports are written into `_photo_agent_staging` so later runs can reuse unchanged files. Semantic grouping is performed with DBSCAN using cosine distance.

Important implementation details:

- The default model is `facebook/dinov2-large`
- Model loading uses `local_files_only=True`
- That means the DINOv2 model must already exist in the local Hugging Face cache
- GPU is optional, but CUDA greatly improves embedding speed

### 2. Review

There are two review paths:

- Integrated flow in `photo_agent_app.py`
- Standalone review flow in `photo_agent_review.py`

Photos inside a group are sorted by:

`combined_score = 0.7 * sharpness_normalized + 0.3 * exposure`

The top-ranked photo becomes the AI recommendation. Users can keep the recommendation, manually select photos to keep, skip a group, or save a move plan for later.

### 3. Post-processing

Photo Agent does not delete original photos directly. Instead it:

1. Writes `move_plan.json`
2. Moves selected files into `_photo_agent_staging/reviewed_moves`
3. Requires another explicit `yes` confirmation before permanently deleting staged image files

This staging-first approach is meant to reduce accidental data loss.

## Environment Requirements

- Windows is the primary tested environment
- Python 3.11 is recommended
- Conda is recommended for environment isolation
- NVIDIA GPU is optional
- If GPU is unavailable, the project still runs on CPU, but more slowly

## Installation

### 1. Create a Conda environment

```powershell
conda create -p .conda\photo-agent python=3.11 -y
```

### 2. Install dependencies

```powershell
.conda\photo-agent\python.exe -m pip install torch torchvision transformers huggingface_hub
.conda\photo-agent\python.exe -m pip install faiss-cpu pillow-heif imagehash scikit-learn opencv-python PyQt6
```

If you want CUDA-enabled PyTorch, install the matching CUDA wheel instead of the CPU-only build.

### 3. Download the DINOv2 model into the local Hugging Face cache

```powershell
.conda\photo-agent\python.exe -c "from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-large'); AutoModel.from_pretrained('facebook/dinov2-large')"
```

Because the project uses `local_files_only=True`, the model must be available locally before indexing begins.

## Usage

### Integrated GUI

```powershell
.conda\photo-agent\python.exe photo_agent_app.py
```

Recommended flow:

1. Choose the photo folder
2. Select the DINOv2 model, batch size, pHash threshold, and DBSCAN `eps`
3. Start indexing
4. Review duplicate and similar groups
5. Save the move plan
6. Decide whether to move files into staging immediately
7. If needed later, delete staged images only after explicit confirmation

### Standalone indexing

```powershell
.conda\photo-agent\python.exe photo_agent_process.py --photo-dir "D:\path\to\photos"
```

Useful options:

```powershell
.conda\photo-agent\python.exe photo_agent_process.py `
  --photo-dir "D:\path\to\photos" `
  --batch-size 16 `
  --phash-threshold 10 `
  --dbscan-eps 0.15 `
  --model-name facebook/dinov2-large
```

### Standalone review

```powershell
.conda\photo-agent\python.exe photo_agent_review.py --report "D:\path\to\photos\_photo_agent_staging\analysis_report.json"
```

## Generated Files

Inside the selected photo folder, Photo Agent creates:

```text
_photo_agent_staging/
  analysis_report.json
  metadata.json
  embeddings.npy
  faiss.index
  move_plan.json
  move_log.json
  delete_log.json
  pending_move_list.txt
  pending_delete_list.txt
  reviewed_moves/
  duplicates/
  similar_groups/
```

Notes:

- `duplicates/` and `similar_groups/` currently exist as placeholder directories
- Group reports are stored in JSON, not as copied images inside those folders

## Safety Rules

- Original files are never deleted directly during review
- Files are moved to staging first
- Permanent deletion requires exact `yes`
- Deletion only targets staged image files, not the original photo directory

## Release Assets

For release cover images, screenshots, and packaging notes, see:

[release/README.md](./release/README.md)

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
