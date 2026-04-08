---
name: photo-agent-maintainer
description: Use when maintaining the Photo Agent repo, updating docs, explaining architecture, preparing releases, or editing flows around indexing, review, staging, and safe deletion.
---

# Photo Agent Maintainer

Use this skill when working inside the Photo Agent repository.

## What to Keep in Mind

- The repo has two GUI entry points:
  - `photo_agent_app.py` for the integrated flow
  - `photo_agent_review.py` for standalone review of an existing report
- Indexing lives in `photo_agent_process.py`
- Moving staged files lives in `photo_agent_move.py`
- Deleting staged images lives in `photo_agent_cleanup.py`
- The project is staging-first by design. Do not change docs or behavior in ways that imply direct deletion of original photos.

## Architecture Notes

- pHash is used for near-duplicate grouping
- DINOv2 embeddings are normalized with `faiss.normalize_L2`
- FAISS uses `IndexFlatIP`, so similarity behaves like cosine similarity after normalization
- Semantic clustering uses DBSCAN with cosine distance
- The AI recommendation is the top item after sorting by:
  - `0.7 * sharpness_normalized + 0.3 * exposure`
- `duplicates/` and `similar_groups/` are currently placeholder directories

## Documentation Rules

- Keep the public project name as `Photo Agent`
- When documenting installation, mention that DINOv2 currently loads with `local_files_only=True`
- When documenting release materials, separate:
  - cover image guidance
  - GUI screenshots
  - architecture diagrams
- Link the MakerHub article when relevant:
  - `https://makerhub.qianpro.shop/posts/photo-agent-dinov2-phash-faiss-iphone-heic`

## Editing Caution

- Some GUI strings may contain encoding issues from previous edits; avoid broad rewrites unless intentionally fixing text encoding
- Prefer small documentation and workflow changes over large refactors
- Preserve the safety requirement that permanent deletion must require exact `yes`
