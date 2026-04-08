# Release Assets Checklist

This folder documents the recommended assets for GitHub releases, MakerHub updates, and social previews.

## Recommended Cover Image

Use a 16:9 cover image that focuses on the real pain point:

- phone photo library feels overcrowded
- repeated photos consume storage
- the cleanup flow feels calmer after grouping and review
- avoid overly futuristic or heavy sci-fi visuals

Recommended overlay text for the cover image:

```text
手機相簿又爆滿了？
用 AI 幫你整理重複照片
```

## Recommended Screenshots

Prepare these screenshots before cutting a release:

1. `gui-home.png`
   Integrated GUI with folder selection, model selector, and indexing controls.
2. `gui-indexing.png`
   Indexing in progress with progress bar, current thumbnail, and log output.
3. `gui-review.png`
   Group review screen with thumbnails, scores, and keep/skip actions.
4. `gui-move-confirm.png`
   Save plan and move-to-staging confirmation dialog.
5. `gui-delete-confirm.png`
   Delete-staged-images confirmation flow that requires exact `yes`.

## Existing Diagram Assets

Current architecture assets already in the repo:

- `photo_agent_architecture.png`
- `photo_agent_architecture.svg`

These are useful for release notes, README sections, or technical blog posts, but they are not a replacement for product screenshots.

## Suggested Release Notes Structure

When creating a GitHub release, a practical order is:

1. One-line summary
2. What the tool does
3. What's new in this release
4. Key screenshots
5. Known limitations
6. Link to MakerHub article

## Suggested First Release Copy

```text
Photo Agent is a desktop photo-sorting tool for large local libraries and iPhone backups. It combines pHash, DINOv2, FAISS, DBSCAN, and a safe staging-first workflow so users can review duplicate and similar photos before moving or deleting anything.
```

MakerHub article:

[Photo Agent：用 DINOv2、pHash 與 FAISS 整理 iPhone HEIC 相簿](https://makerhub.qianpro.shop/posts/photo-agent-dinov2-phash-faiss-iphone-heic)
