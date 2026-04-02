# Plan: Image Source Tracking & Per-File Contact Sheet Review

## Problem

When training on multiple video files, the review grid shows a **flat list of all frames** across every source. As the dataset grows, this list becomes unmanageably long and there is no way to review or curate frames from each video independently.

## Current State

- `pairs.json` already stores `source_file` on every pair entry
- Frame names encode the video stem: `{videoStem}_{frameStem}`
- `load-dataset` returns `sourceFile` per frame but the UI ignores it
- The review grid renders all frames in a single flat grid with no grouping or filtering

## Proposed Solution

Add a **source file navigator** to the review view that groups frames by their origin video/image file and lets you review each source independently.

---

### 1. Backend — `load-dataset` response enrichment (`src/main.js`)

Modify the `load-dataset` IPC handler to also return a `sourceFiles` summary:

```
sourceFiles: [
  {
    name: "beach_360.mov",       // original filename
    type: "video",               // "video" | "image"
    frameCount: 47,              // unique frames from this source
    pairCount: 470,              // total conditioning pairs
  },
  {
    name: "sunset_pano.jpg",
    type: "image",
    frameCount: 1,
    pairCount: 10,
  },
  ...
]
```

The existing `frames` array continues to be returned as-is (each frame already carries `sourceFile`). No breaking changes.

### 2. UI — Source file navigator (`src/index.html`)

Add a **horizontally scrollable strip of source-file tabs** between the toolbar and the grid:

```
┌──────────────────────────────────────────────────────────────┐
│  ← Back     47 frames · 470 pairs     Select All / None     │  toolbar
├──────────────────────────────────────────────────────────────┤
│  [All (124)] [beach_360.mov (47)] [cliff_walk.mp4 (32)] ... │  source tabs
├──────────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │     │ │     │ │     │ │     │ │     │ │     │ │     │  │  grid (filtered)
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  │
│  ...                                                         │
└──────────────────────────────────────────────────────────────┘
```

Each tab shows:
- **Type icon** (film strip for video, image icon for still)
- **Filename** (truncated with ellipsis if long)
- **Frame count** badge

Clicking a tab filters the grid to only show frames from that source. The **"All"** tab restores the flat view.

**Styling:**
- Horizontal scrollable row with subtle left/right fade if overflow
- Active tab: accent-colored border/background matching the existing `.radio-btn.active` style
- Tab type icon colored using existing `--video-color` / `--image-color` tokens

### 3. State changes (`src/index.html`)

Add to the `state` object:

```js
reviewSourceFiles: [],         // Array of { name, type, frameCount, pairCount }
reviewActiveSource: null,      // null = "All", or source filename string
```

### 4. Grid filtering logic

When rendering the review grid:
- If `reviewActiveSource` is `null`, show all frames (current behavior)
- If set to a filename, filter: `state.reviewFrames.filter(f => f.sourceFile === state.reviewActiveSource)`

The toolbar stats update to reflect the **visible** frame/pair count, with a secondary indicator showing totals when filtered.

### 5. Selection behavior

| Action | Scope |
|--------|-------|
| Select All / None / Invert | Applies only to **visible** (filtered) frames |
| Selection state | **Persists** across source tab switches — selecting frames in video A, switching to video B, and selecting more frames there accumulates selections |
| Selection bar count | Shows **total** selected count across all sources |
| Delete | Deletes all selected frames regardless of which source tab is active |

This lets you review source A, select bad frames, switch to source B, select more, then delete everything in one batch.

### 6. Keyboard navigation (optional enhancement)

- **Left/Right arrow keys** when no cell is focused: switch between source tabs
- Existing Shift+click range-select continues to work within the visible grid

---

## Files Changed

| File | Change |
|------|--------|
| `src/main.js` | Enrich `load-dataset` response with `sourceFiles` array |
| `src/index.html` | Add source tab strip, state fields, filtering logic, updated stats |
| `src/preload.js` | No changes needed — existing IPC surface is sufficient |
| `python/equirect_dataset_generator.py` | No changes needed — already writes `source_file` in pairs |

## Risks & Considerations

- **Single-image sources**: When source is a still image (not video), it produces only 1 frame. The tab strip handles this naturally — single-frame tabs are just small.
- **Many source files**: If 20+ files are loaded, the tab strip scrolls horizontally. Could add a collapse/dropdown if this becomes unwieldy, but horizontal scroll is sufficient for the common case.
- **Backward compatibility**: Old datasets already have `source_file` in `pairs.json`, so this works retroactively on existing datasets.
- **Performance**: Filtering is in-memory on the already-loaded frame array. Thumbnail lazy-loading via `IntersectionObserver` continues to work since it triggers on DOM visibility, and filtered-out cells aren't rendered.

## Implementation Order

1. Enrich `load-dataset` backend response with `sourceFiles`
2. Add state fields and source tab strip UI
3. Wire up tab click → filter grid
4. Update toolbar stats to reflect filtered view
5. Scope Select All/None/Invert to visible frames
6. Test with multi-video dataset and single-image sources
