# Plan: Image Source Tracking & Per-File Contact Sheet Review

## Problem

When training on multiple video files, the review grid shows a **flat list of all frames** across every source. As the dataset grows (hundreds of video files), this list becomes unmanageably long and there is no way to review or curate frames from each video independently.

## Current State

- `pairs.json` already stores `source_file` on every pair entry
- Frame names encode the video stem: `{videoStem}_{frameStem}`
- `load-dataset` returns `sourceFile` per frame but the UI ignores it
- The review grid renders all frames in a single flat grid with no grouping or filtering

## Proposed Solution: Drill-Down (Master → Detail)

The review view becomes **two levels**:

### Level 1 — Source File List

The new landing page for review. A searchable, sortable list of all source files with thumbnails:

```
┌──────────────────────────────────────────────────────────────┐
│  ← Back        3 sources · 124 frames · 1,240 pairs         │
├──────────────────────────────────────────────────────────────┤
│  🔍 Search sources...           Sort: Name ▾                 │
├──────────────────────────────────────────────────────────────┤
│  ☐ [All Frames]                    124 frames · 1,240 pairs  │
│  ──────────────────────────────────────────────────────────  │
│  ☐ [thumb] 🎬 beach_360.mov        47 frames ·   470 pairs  │
│  ☐ [thumb] 🎬 cliff_walk.mp4       32 frames ·   320 pairs  │
│  ☐ [thumb] 🖼 sunset_pano.jpg       1 frame  ·    10 pairs  │
│  ... (scrollable, hundreds of rows)                          │
└──────────────────────────────────────────────────────────────┘
│  ☐ 2 sources selected · 79 frames · 790 pairs    [Delete]   │  (selection bar)
└──────────────────────────────────────────────────────────────┘
```

Features:
- **Thumbnail preview**: First frame from each source displayed inline
- **Type icon**: Film strip for video, image icon for stills
- **Search field**: Filter source list by filename
- **Sort controls**: By name, frame count, or pair count
- **Multi-select checkboxes**: Select entire sources for bulk delete
- **Source-level bulk delete**: Delete all frames from selected sources without drilling in
- **"All Frames" entry**: Top row drills into the existing flat contact sheet

### Level 2 — Contact Sheet (Per-Source)

Click a source row → existing contact sheet grid, scoped to that single source.

Breadcrumb navigation: `All Sources › beach_360.mov`

Back button returns to Level 1 with updated stats.

---

## Implementation Details

### 1. Backend — `load-dataset` response enrichment (`src/main.js`)

Modify the `load-dataset` IPC handler to return a `sourceFiles` summary:

```js
sourceFiles: [
  {
    name: "beach_360.mov",
    type: "video",
    frameCount: 47,
    pairCount: 470,
    firstFrame: "beach_360_frame_000001",  // for thumbnail
  },
  ...
]
```

### 2. Backend — `generate-source-thumbnails` IPC handler (`src/main.js`)

New handler to generate thumbnails for source file preview (uses the first frame's target image).

### 3. Backend — `delete-sources` IPC handler (`src/main.js`)

New handler that deletes all frames/pairs/metadata for given source files. Reuses the same cleanup logic as `delete-frames` but scoped by `source_file`.

### 4. UI — Source List View (`src/index.html`)

New `renderReviewSources()` function for Level 1:
- State: `reviewSourceFiles`, `reviewActiveSource`, `reviewSourceSelection`, `reviewSourceSearch`, `reviewSourceSort`
- View transitions: `review-sources` (Level 1) ↔ `review` (Level 2)
- Lazy-loaded thumbnails via IntersectionObserver (same pattern as contact sheet)

### 5. UI — Filtered Contact Sheet (`src/index.html`)

Modify `renderReview()` for Level 2:
- Filter `reviewFrames` by `reviewActiveSource` when set
- Breadcrumb showing `All Sources › filename`
- Back button returns to `review-sources` with refreshed stats
- "All Frames" mode: `reviewActiveSource = "__all__"`

### 6. Selection & Delete Behavior

| Context | Selection Scope |
|---------|----------------|
| Source list (Level 1) | Select entire source files via checkboxes |
| Contact sheet (Level 2) | Select individual frames (existing behavior) |
| Source-level delete | Removes all frames/pairs for selected sources |
| Frame-level delete | Removes selected frames (existing behavior) |
| Back navigation | Source list stats refresh to reflect deletions |
| Empty sources | Disappear from list after all frames deleted |

---

## Files Changed

| File | Change |
|------|--------|
| `src/main.js` | Enrich `load-dataset`, add `generate-source-thumbnails`, add `delete-sources` |
| `src/preload.js` | Expose new IPC handlers |
| `src/index.html` | Source list view, state fields, drill-down navigation, sort/search/filter |
| `python/equirect_dataset_generator.py` | No changes needed |

## Implementation Order

1. Backend: Enrich `load-dataset` with `sourceFiles`
2. Backend: Add `generate-source-thumbnails` and `delete-sources` IPC handlers
3. Preload: Expose new handlers
4. UI: Source list view with thumbnails, search, sort
5. UI: Source-level multi-select and bulk delete
6. UI: Drill-down to filtered contact sheet with breadcrumb nav
7. UI: Stats refresh on back-navigation, empty source cleanup
8. UI: "All Frames" entry point
9. Test with multi-source dataset
