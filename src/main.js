const { app, BrowserWindow, ipcMain, dialog, shell, nativeImage } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");

let mainWindow;

// ─── Window ───

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 820,
    minWidth: 800,
    minHeight: 600,
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 18 },
    backgroundColor: "#0c0e12",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// ─── File / Folder Dialogs ───

const SUPPORTED_EXTENSIONS = [
  // Video
  "mov", "mp4", "mkv", "avi", "mxf", "webm", "m4v", "mpg", "mpeg",
  "ts", "mts", "m2ts", "wmv", "flv", "3gp", "ogv", "r3d", "braw",
  // Image
  "jpg", "jpeg", "png", "tiff", "tif", "exr", "hdr", "bmp", "webp", "dpx",
];

const VIDEO_EXTENSIONS = new Set([
  "mov", "mp4", "mkv", "avi", "mxf", "webm", "m4v", "mpg", "mpeg",
  "ts", "mts", "m2ts", "wmv", "flv", "3gp", "ogv", "r3d", "braw",
]);

function getExt(filePath) {
  return path.extname(filePath).replace(".", "").toLowerCase();
}

function isSupported(filePath) {
  return SUPPORTED_EXTENSIONS.includes(getExt(filePath));
}

/** LoRA v1 datasets use frames/; legacy ControlNet used target/ */
function datasetImageSubdir(datasetDir) {
  return fs.existsSync(path.join(datasetDir, "dataset_manifest.json")) ? "frames" : "target";
}

function writePairsFromManifest(manifest, datasetDir) {
  const pairs = (manifest.samples || []).map((s) => ({
    conditioning: s.depth || s.image,
    target: s.image,
    metadata: s.caption_file,
    source_frame: s.id,
    source_file: s.source_file,
  }));
  fs.writeFileSync(
    path.join(datasetDir, "pairs.json"),
    JSON.stringify(
      {
        generator: "equirect_dataset_generator.py",
        format: "stereo_lora_v1",
        total_pairs: pairs.length,
        total_frames: pairs.length,
        pairs,
      },
      null,
      2
    )
  );
}

function collectFiles(dirPath) {
  const results = [];
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith(".")) continue;
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        results.push(...collectFiles(fullPath));
      } else if (isSupported(fullPath)) {
        const stats = fs.statSync(fullPath);
        results.push({
          name: entry.name,
          path: fullPath,
          size: stats.size,
          type: VIDEO_EXTENSIONS.has(getExt(fullPath)) ? "video" : "image",
        });
      }
    }
  } catch (err) {
    console.error("Error scanning directory:", err);
  }
  return results;
}

ipcMain.handle("select-files", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Media Files",
    properties: ["openFile", "multiSelections"],
    filters: [
      {
        name: "Media Files",
        extensions: SUPPORTED_EXTENSIONS,
      },
    ],
  });

  if (result.canceled) return [];

  return result.filePaths
    .filter(isSupported)
    .map((fp) => {
      const stats = fs.statSync(fp);
      return {
        name: path.basename(fp),
        path: fp,
        size: stats.size,
        type: VIDEO_EXTENSIONS.has(getExt(fp)) ? "video" : "image",
      };
    });
});

ipcMain.handle("select-folder", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Folder",
    properties: ["openDirectory"],
  });

  if (result.canceled) return [];
  return collectFiles(result.filePaths[0]);
});

ipcMain.handle("select-output-dir", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Output Directory",
    properties: ["openDirectory", "createDirectory"],
  });

  if (result.canceled) return null;
  return result.filePaths[0];
});

// ─── Python Process ───

let pythonProcess = null;

function findPython() {
  const candidates = [
    "python3",
    "/usr/local/bin/python3",
    "/opt/homebrew/bin/python3",
    path.join(os.homedir(), "miniconda3", "bin", "python3"),
    path.join(os.homedir(), "anaconda3", "bin", "python3"),
    "python",
  ];

  for (const candidate of candidates) {
    try {
      const result = require("child_process").execSync(
        `${candidate} --version 2>&1`,
        { timeout: 5000 }
      );
      const version = result.toString().trim();
      if (version.includes("Python 3")) {
        return candidate;
      }
    } catch (e) {
      // Try next
    }
  }
  return null;
}

function findTrainingPython() {
  const venvPython = path.join(__dirname, "..", ".venv", "bin", "python");
  if (fs.existsSync(venvPython)) {
    return venvPython;
  }
  return findPython();
}

ipcMain.handle("check-dependencies", async () => {
  const pythonPath = findPython();
  if (!pythonPath) {
    return { ok: false, error: "python3_not_found", pythonPath: null };
  }

  // Check for required packages
  try {
    const result = require("child_process").execSync(
      `${pythonPath} -c "import cv2; import numpy; print('ok')"`,
      { timeout: 10000 }
    );
    if (result.toString().trim() === "ok") {
      // Check ffmpeg
      try {
        require("child_process").execSync("ffmpeg -version", { timeout: 5000 });
        return { ok: true, pythonPath };
      } catch {
        return { ok: false, error: "ffmpeg_not_found", pythonPath };
      }
    }
  } catch (e) {
    return {
      ok: false,
      error: "packages_missing",
      pythonPath,
      details: e.stderr?.toString() || e.message,
    };
  }

  return { ok: false, error: "unknown", pythonPath };
});

ipcMain.handle("start-processing", async (event, { files, config, outputDir }) => {
  const pythonPath = findPython();
  if (!pythonPath) {
    return { ok: false, error: "Python 3 not found" };
  }

  const scriptPath = path.join(__dirname, "..", "python", "equirect_dataset_generator.py");

  // Process files sequentially
  const fileList = files.map((f) => f.path);

  // Write a temp job file so the Python script can read the full file list
  const jobFile = path.join(os.tmpdir(), `controlnet_job_${Date.now()}.json`);
  fs.writeFileSync(
    jobFile,
    JSON.stringify({
      files: fileList,
      config: config,
      outputDir: outputDir,
    })
  );

  const args = [scriptPath, "--job-file", jobFile];

  pythonProcess = spawn(pythonPath, args, {
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  pythonProcess.stdout.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      mainWindow?.webContents.send("process-log", line);
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      mainWindow?.webContents.send("process-log", `[stderr] ${line}`);
    }
  });

  pythonProcess.on("close", (code) => {
    // Read results if available
    const resultsFile = path.join(outputDir, "generation_results.json");
    let results = null;
    try {
      if (fs.existsSync(resultsFile)) {
        results = JSON.parse(fs.readFileSync(resultsFile, "utf-8"));
      }
    } catch (e) {
      console.error("Could not read results:", e);
    }

    mainWindow?.webContents.send("process-complete", {
      code,
      results,
    });

    // Clean up job file
    try { fs.unlinkSync(jobFile); } catch {}
    pythonProcess = null;
  });

  return { ok: true };
});

ipcMain.handle("cancel-processing", async () => {
  if (pythonProcess) {
    pythonProcess.kill("SIGTERM");
    pythonProcess = null;
    return { ok: true };
  }
  return { ok: false };
});

ipcMain.handle("open-folder", async (event, folderPath) => {
  shell.openPath(folderPath);
});

// ─── Dataset Review ───

ipcMain.handle("select-dataset-dir", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Dataset Directory",
    properties: ["openDirectory"],
  });

  if (result.canceled) return null;

  const dir = result.filePaths[0];
  const manifestPath = path.join(dir, "dataset_manifest.json");
  const pairsPath = path.join(dir, "pairs.json");
  if (!fs.existsSync(manifestPath) && !fs.existsSync(pairsPath)) {
    return {
      error: "no_manifest",
      message: "Selected folder needs dataset_manifest.json (or legacy pairs.json).",
    };
  }
  return { ok: true, path: dir };
});

ipcMain.handle("load-dataset", async (event, dir) => {
  try {
    const manifestPath = path.join(dir, "dataset_manifest.json");
    const pairsPath = path.join(dir, "pairs.json");
    const useV1 = fs.existsSync(manifestPath);
    const manifest = JSON.parse(
      fs.readFileSync(useV1 ? manifestPath : pairsPath, "utf-8")
    );

    const frameMap = new Map();
    if (useV1 && Array.isArray(manifest.samples)) {
      for (const s of manifest.samples) {
        const name = s.id;
        if (!frameMap.has(name)) {
          frameMap.set(name, { name, pairCount: 1, sourceFile: s.source_file });
        }
      }
    } else {
      for (const pair of manifest.pairs || []) {
        const name = pair.source_frame;
        if (!frameMap.has(name)) {
          frameMap.set(name, { name, pairCount: 0, sourceFile: pair.source_file });
        }
        frameMap.get(name).pairCount++;
      }
    }

    const frames = [];
    for (const [name, info] of frameMap) {
      const rel = useV1 ? `frames/${name}.png` : `target/${name}.png`;
      const targetPath = path.join(dir, rel);
      if (fs.existsSync(targetPath)) {
        frames.push({
          name,
          targetPath,
          pairCount: info.pairCount,
          sourceFile: info.sourceFile,
        });
      }
    }

    // Build source files summary for the drill-down source list
    const sourceMap = new Map();
    for (const frame of frames) {
      const sf = frame.sourceFile || "__unknown__";
      if (!sourceMap.has(sf)) {
        const ext = path.extname(sf).replace(".", "").toLowerCase();
        const isVideo = VIDEO_EXTENSIONS.has(ext);
        sourceMap.set(sf, {
          name: sf,
          type: isVideo ? "video" : "image",
          frameCount: 0,
          pairCount: 0,
          firstFrame: null,
        });
      }
      const entry = sourceMap.get(sf);
      entry.frameCount++;
      entry.pairCount += frame.pairCount;
      if (!entry.firstFrame) entry.firstFrame = frame.name;
    }
    const sourceFiles = [...sourceMap.values()];

    const totalListed = useV1
      ? (manifest.total_samples ?? manifest.samples?.length ?? 0)
      : (manifest.total_pairs || (manifest.pairs || []).length);

    return {
      ok: true,
      frames,
      sourceFiles,
      totalPairs: totalListed,
      totalFrames: frames.length,
      format: useV1 ? "stereo_lora_v1" : "legacy_pairs",
    };
  } catch (err) {
    return { ok: false, error: err.message };
  }
});

ipcMain.handle("generate-thumbnails", async (event, { datasetDir, frameNames }) => {
  const thumbDir = path.join(datasetDir, ".thumbs");
  if (!fs.existsSync(thumbDir)) {
    fs.mkdirSync(thumbDir, { recursive: true });
  }

  const sub = datasetImageSubdir(datasetDir);
  const thumbnails = {};
  for (const name of frameNames) {
    const cachedPath = path.join(thumbDir, `${name}.jpg`);

    // Check cache first
    if (fs.existsSync(cachedPath)) {
      const buf = fs.readFileSync(cachedPath);
      thumbnails[name] = `data:image/jpeg;base64,${buf.toString("base64")}`;
      continue;
    }

    const targetPath = path.join(datasetDir, sub, `${name}.png`);
    if (!fs.existsSync(targetPath)) continue;

    try {
      const img = nativeImage.createFromPath(targetPath);
      if (img.isEmpty()) continue;
      const thumb = img.resize({ height: 150 });
      const jpegBuf = thumb.toJPEG(70);
      fs.writeFileSync(cachedPath, jpegBuf);
      thumbnails[name] = `data:image/jpeg;base64,${jpegBuf.toString("base64")}`;
    } catch (err) {
      console.error(`Thumbnail error for ${name}:`, err);
    }
  }

  return { thumbnails };
});

ipcMain.handle("generate-source-thumbnails", async (event, { datasetDir, sources }) => {
  const thumbDir = path.join(datasetDir, ".thumbs");
  if (!fs.existsSync(thumbDir)) {
    fs.mkdirSync(thumbDir, { recursive: true });
  }

  const thumbnails = {};
  for (const src of sources) {
    const frameName = src.firstFrame;
    if (!frameName) continue;

    const cacheKey = `_src_${src.name}`;
    const cachedPath = path.join(thumbDir, `${cacheKey}.jpg`);

    if (fs.existsSync(cachedPath)) {
      const buf = fs.readFileSync(cachedPath);
      thumbnails[src.name] = `data:image/jpeg;base64,${buf.toString("base64")}`;
      continue;
    }

    const sub = datasetImageSubdir(datasetDir);
    const targetPath = path.join(datasetDir, sub, `${frameName}.png`);
    if (!fs.existsSync(targetPath)) continue;

    try {
      const img = nativeImage.createFromPath(targetPath);
      if (img.isEmpty()) continue;
      const thumb = img.resize({ height: 80 });
      const jpegBuf = thumb.toJPEG(70);
      fs.writeFileSync(cachedPath, jpegBuf);
      thumbnails[src.name] = `data:image/jpeg;base64,${jpegBuf.toString("base64")}`;
    } catch (err) {
      console.error(`Source thumbnail error for ${src.name}:`, err);
    }
  }

  return { thumbnails };
});

ipcMain.handle("delete-sources", async (event, { datasetDir, sourceNames }) => {
  try {
    const dmPath = path.join(datasetDir, "dataset_manifest.json");
    if (fs.existsSync(dmPath)) {
      const manifest = JSON.parse(fs.readFileSync(dmPath, "utf-8"));
      const del = new Set(sourceNames);
      const samples = manifest.samples || [];
      const toDelete = samples.filter((s) => del.has(s.source_file));
      const toKeep = samples.filter((s) => !del.has(s.source_file));
      const deletedSourceDirs = new Set();
      for (const s of toDelete) {
        try {
          fs.unlinkSync(path.join(datasetDir, s.image));
        } catch {}
        try {
          fs.unlinkSync(path.join(datasetDir, s.caption_file));
        } catch {}
        if (s.depth) {
          try {
            fs.unlinkSync(path.join(datasetDir, s.depth));
          } catch {}
        }
        try {
          fs.unlinkSync(path.join(datasetDir, ".thumbs", `${s.id}.jpg`));
        } catch {}
        const videoStem = path.parse(s.source_file).name;
        const sourceDir = path.join(datasetDir, "source_equirects", videoStem);
        const frameSuffix = s.id.startsWith(videoStem + "_")
          ? s.id.slice(videoStem.length + 1)
          : s.id;
        const sourceFile = path.join(sourceDir, `${frameSuffix}.png`);
        try {
          fs.unlinkSync(sourceFile);
        } catch {}
        deletedSourceDirs.add(sourceDir);
      }
      for (const dir of deletedSourceDirs) {
        try {
          if (fs.readdirSync(dir).length === 0) fs.rmdirSync(dir);
        } catch {}
      }
      for (const srcName of sourceNames) {
        try {
          fs.unlinkSync(path.join(datasetDir, ".thumbs", `_src_${srcName}.jpg`));
        } catch {}
      }
      manifest.samples = toKeep;
      manifest.total_samples = toKeep.length;
      fs.writeFileSync(dmPath, JSON.stringify(manifest, null, 2));
      writePairsFromManifest(manifest, datasetDir);
      const resultsPath = path.join(datasetDir, "generation_results.json");
      try {
        const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
        results.total_samples = toKeep.length;
        results.total_frames = toKeep.length;
        fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
      } catch {}
      return {
        ok: true,
        deletedSources: sourceNames.length,
        deletedFrames: toDelete.length,
        deletedPairs: toDelete.length,
        remainingFrames: toKeep.length,
        remainingPairs: toKeep.length,
      };
    }

    const pairsPath = path.join(datasetDir, "pairs.json");
    const manifest = JSON.parse(fs.readFileSync(pairsPath, "utf-8"));

    const sourcesToDelete = new Set(sourceNames);
    const pairsToDelete = [];
    const pairsToKeep = [];

    for (const pair of manifest.pairs) {
      if (sourcesToDelete.has(pair.source_file)) {
        pairsToDelete.push(pair);
      } else {
        pairsToKeep.push(pair);
      }
    }

    // Collect unique frames being deleted
    const framesToDelete = new Set(pairsToDelete.map(p => p.source_frame));

    // Delete conditioning images and metadata
    for (const pair of pairsToDelete) {
      const condPath = path.join(datasetDir, pair.conditioning);
      const metaPath = path.join(datasetDir, pair.metadata);
      try { fs.unlinkSync(condPath); } catch {}
      try { fs.unlinkSync(metaPath); } catch {}
    }

    // Delete target images, source equirects, and thumbnails
    const deletedSourceDirs = new Set();
    for (const name of framesToDelete) {
      try { fs.unlinkSync(path.join(datasetDir, "target", `${name}.png`)); } catch {}
      try { fs.unlinkSync(path.join(datasetDir, ".thumbs", `${name}.jpg`)); } catch {}

      const samplePair = pairsToDelete.find(p => p.source_frame === name);
      if (samplePair && samplePair.source_file) {
        const videoStem = path.parse(samplePair.source_file).name;
        const sourceDir = path.join(datasetDir, "source_equirects", videoStem);
        const frameSuffix = name.startsWith(videoStem + "_") ? name.slice(videoStem.length + 1) : name;
        const sourceFile = path.join(sourceDir, `${frameSuffix}.png`);
        try { fs.unlinkSync(sourceFile); } catch {}
        deletedSourceDirs.add(sourceDir);
      }
    }

    // Delete source thumbnail cache entries
    for (const srcName of sourceNames) {
      const cacheKey = `_src_${srcName}`;
      try { fs.unlinkSync(path.join(datasetDir, ".thumbs", `${cacheKey}.jpg`)); } catch {}
    }

    // Clean up empty source_equirects subdirectories
    for (const dir of deletedSourceDirs) {
      try {
        const remaining = fs.readdirSync(dir);
        if (remaining.length === 0) fs.rmdirSync(dir);
      } catch {}
    }

    // Update manifest
    manifest.pairs = pairsToKeep;
    manifest.total_pairs = pairsToKeep.length;
    const remainingFrames = new Set(pairsToKeep.map(p => p.source_frame));
    manifest.total_frames = remainingFrames.size;
    fs.writeFileSync(pairsPath, JSON.stringify(manifest, null, 2));

    // Update generation_results.json
    const resultsPath = path.join(datasetDir, "generation_results.json");
    try {
      const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
      results.total_pairs = pairsToKeep.length;
      results.total_frames = remainingFrames.size;
      fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    } catch {}

    return {
      ok: true,
      deletedSources: sourceNames.length,
      deletedFrames: framesToDelete.size,
      deletedPairs: pairsToDelete.length,
      remainingFrames: remainingFrames.size,
      remainingPairs: pairsToKeep.length,
    };
  } catch (err) {
    return { ok: false, error: err.message };
  }
});

ipcMain.handle("delete-frames", async (event, { datasetDir, frameNames }) => {
  try {
    const dmPath = path.join(datasetDir, "dataset_manifest.json");
    if (fs.existsSync(dmPath)) {
      const manifest = JSON.parse(fs.readFileSync(dmPath, "utf-8"));
      const del = new Set(frameNames);
      const samples = manifest.samples || [];
      const toDelete = samples.filter((s) => del.has(s.id));
      const toKeep = samples.filter((s) => !del.has(s.id));
      const deletedSourceDirs = new Set();
      for (const s of toDelete) {
        try {
          fs.unlinkSync(path.join(datasetDir, s.image));
        } catch {}
        try {
          fs.unlinkSync(path.join(datasetDir, s.caption_file));
        } catch {}
        if (s.depth) {
          try {
            fs.unlinkSync(path.join(datasetDir, s.depth));
          } catch {}
        }
        try {
          fs.unlinkSync(path.join(datasetDir, ".thumbs", `${s.id}.jpg`));
        } catch {}
        const videoStem = path.parse(s.source_file).name;
        const sourceDir = path.join(datasetDir, "source_equirects", videoStem);
        const frameSuffix = s.id.startsWith(videoStem + "_")
          ? s.id.slice(videoStem.length + 1)
          : s.id;
        const sourceFile = path.join(sourceDir, `${frameSuffix}.png`);
        try {
          fs.unlinkSync(sourceFile);
        } catch {}
        deletedSourceDirs.add(sourceDir);
      }
      for (const dir of deletedSourceDirs) {
        try {
          if (fs.readdirSync(dir).length === 0) fs.rmdirSync(dir);
        } catch {}
      }
      manifest.samples = toKeep;
      manifest.total_samples = toKeep.length;
      fs.writeFileSync(dmPath, JSON.stringify(manifest, null, 2));
      writePairsFromManifest(manifest, datasetDir);
      const resultsPath = path.join(datasetDir, "generation_results.json");
      try {
        const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
        results.total_samples = toKeep.length;
        results.total_frames = toKeep.length;
        fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
      } catch {}
      return {
        ok: true,
        deletedFrames: frameNames.length,
        deletedPairs: toDelete.length,
        remainingFrames: toKeep.length,
        remainingPairs: toKeep.length,
      };
    }

    const pairsPath = path.join(datasetDir, "pairs.json");
    const manifest = JSON.parse(fs.readFileSync(pairsPath, "utf-8"));

    const framesToDelete = new Set(frameNames);
    const pairsToDelete = [];
    const pairsToKeep = [];

    for (const pair of manifest.pairs) {
      if (framesToDelete.has(pair.source_frame)) {
        pairsToDelete.push(pair);
      } else {
        pairsToKeep.push(pair);
      }
    }

    // Delete conditioning images and metadata
    for (const pair of pairsToDelete) {
      const condPath = path.join(datasetDir, pair.conditioning);
      const metaPath = path.join(datasetDir, pair.metadata);
      try { fs.unlinkSync(condPath); } catch {}
      try { fs.unlinkSync(metaPath); } catch {}
    }

    // Delete target images, source equirects, and thumbnails
    const deletedSourceDirs = new Set();
    for (const name of frameNames) {
      // Target
      try { fs.unlinkSync(path.join(datasetDir, "target", `${name}.png`)); } catch {}

      // Thumbnail cache
      try { fs.unlinkSync(path.join(datasetDir, ".thumbs", `${name}.jpg`)); } catch {}

      // Source equirect — find the matching pair to get source_file info
      const samplePair = pairsToDelete.find(p => p.source_frame === name);
      if (samplePair && samplePair.source_file) {
        const videoStem = path.parse(samplePair.source_file).name;
        const sourceDir = path.join(datasetDir, "source_equirects", videoStem);
        // The frame file in source_equirects is the part after the video stem prefix
        const frameSuffix = name.startsWith(videoStem + "_") ? name.slice(videoStem.length + 1) : name;
        const sourceFile = path.join(sourceDir, `${frameSuffix}.png`);
        try { fs.unlinkSync(sourceFile); } catch {}
        deletedSourceDirs.add(sourceDir);
      }
    }

    // Clean up empty source_equirects subdirectories
    for (const dir of deletedSourceDirs) {
      try {
        const remaining = fs.readdirSync(dir);
        if (remaining.length === 0) fs.rmdirSync(dir);
      } catch {}
    }

    // Update manifest
    manifest.pairs = pairsToKeep;
    manifest.total_pairs = pairsToKeep.length;
    // Count remaining unique frames
    const remainingFrames = new Set(pairsToKeep.map(p => p.source_frame));
    manifest.total_frames = remainingFrames.size;
    fs.writeFileSync(pairsPath, JSON.stringify(manifest, null, 2));

    // Update generation_results.json
    const resultsPath = path.join(datasetDir, "generation_results.json");
    try {
      const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
      results.total_pairs = pairsToKeep.length;
      results.total_frames = remainingFrames.size;
      fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    } catch {}

    return {
      ok: true,
      deletedFrames: frameNames.length,
      deletedPairs: pairsToDelete.length,
      remainingFrames: remainingFrames.size,
      remainingPairs: pairsToKeep.length,
    };
  } catch (err) {
    return { ok: false, error: err.message };
  }
});

// ─── Training Pipeline ───

let trainingProcess = null;

ipcMain.handle("install-training-deps", async () => {
  const pythonPath = findTrainingPython();
  if (!pythonPath) return { ok: false, error: "Python 3 not found" };

  // Determine which requirements file to use
  const isMac = process.platform === "darwin";
  const reqFile = isMac ? "requirements_mac.txt" : "requirements_nvidia.txt";
  const reqPath = path.join(__dirname, "..", "train", reqFile);

  if (!fs.existsSync(reqPath)) {
    return { ok: false, error: `Requirements file not found: ${reqFile}` };
  }

  const proc = spawn(pythonPath, ["-m", "pip", "install", "-r", reqPath], {
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  proc.stdout.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      mainWindow?.webContents.send("training-log", line);
    }
  });
  proc.stderr.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      mainWindow?.webContents.send("training-log", line);
    }
  });

  return new Promise((resolve) => {
    proc.on("close", (code) => {
      resolve({ ok: code === 0, error: code !== 0 ? `pip exited with code ${code}` : null });
    });
  });
});

ipcMain.handle("detect-hardware", async () => {
  const pythonPath = findTrainingPython();
  if (!pythonPath) return { ok: false, error: "python3_not_found" };

  const scriptPath = path.join(__dirname, "..", "train", "train_lora.py");
  try {
    const result = require("child_process").execSync(
      `${pythonPath} ${scriptPath} --detect-hardware`,
      { timeout: 30000 }
    );
    return { ok: true, ...JSON.parse(result.toString().trim()) };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

ipcMain.handle("prepare-training-dataset", async (event, { datasetDir, caption }) => {
  const pythonPath = findTrainingPython();
  if (!pythonPath) return { ok: false, error: "Python 3 not found" };

  const scriptPath = path.join(__dirname, "..", "train", "prepare_dataset.py");
  const args = [scriptPath, datasetDir];
  if (caption) args.push("--caption", caption);

  const proc = spawn(pythonPath, args, {
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  let lastLine = "";
  proc.stdout.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      lastLine = line;
      mainWindow?.webContents.send("training-log", line);
    }
  });
  proc.stderr.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      mainWindow?.webContents.send("training-log", `[stderr] ${line}`);
    }
  });

  return new Promise((resolve) => {
    proc.on("close", (code) => {
      if (code === 0) {
        try {
          resolve({ ok: true, ...JSON.parse(lastLine) });
        } catch {
          resolve({ ok: true });
        }
      } else {
        resolve({ ok: false, error: `Process exited with code ${code}` });
      }
    });
  });
});

ipcMain.handle("start-training", async (event, { datasetDir, preset, overrides, outputDir }) => {
  const pythonPath = findTrainingPython();
  if (!pythonPath) return { ok: false, error: "Python 3 not found" };

  const scriptPath = path.join(__dirname, "..", "train", "train_lora.py");

  // Write temp job file
  const jobFile = path.join(os.tmpdir(), `train_job_${Date.now()}.json`);
  fs.writeFileSync(jobFile, JSON.stringify({
    dataset_dir: datasetDir,
    preset: preset,
    overrides: overrides || {},
    output_dir: outputDir,
  }));

  trainingProcess = spawn(pythonPath, [scriptPath, "--job-file", jobFile], {
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  trainingProcess.stdout.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      mainWindow?.webContents.send("training-log", line);
    }
  });

  trainingProcess.stderr.on("data", (data) => {
    const lines = data.toString().split("\n").filter(Boolean);
    for (const line of lines) {
      mainWindow?.webContents.send("training-log", `[stderr] ${line}`);
    }
  });

  trainingProcess.on("close", (code) => {
    mainWindow?.webContents.send("training-complete", { code });
    try { fs.unlinkSync(jobFile); } catch {}
    trainingProcess = null;
  });

  const progressFile = outputDir
    ? path.join(outputDir, "training_progress.json")
    : null;

  return { ok: true, progressFile };
});

ipcMain.handle("cancel-training", async () => {
  if (trainingProcess) {
    trainingProcess.kill("SIGTERM");
    trainingProcess = null;
    return { ok: true };
  }
  return { ok: false };
});

ipcMain.handle("get-training-progress", async (event, progressFile) => {
  if (!progressFile || !fs.existsSync(progressFile)) {
    return { ok: false, error: "Progress file not found" };
  }
  try {
    const data = JSON.parse(fs.readFileSync(progressFile, "utf-8"));

    // Load validation images as base64 (structured: {step, conditioning, samples})
    if (data.validation_images && data.validation_images.length > 0) {
      const loadImg = (imgPath) => {
        const fullPath = path.isAbsolute(imgPath)
          ? imgPath
          : path.join(path.dirname(progressFile), imgPath);
        if (!fs.existsSync(fullPath)) return null;
        try {
          const img = nativeImage.createFromPath(fullPath);
          if (img.isEmpty()) return null;
          const resized = img.resize({ height: 256 });
          return {
            path: imgPath,
            dataUri: `data:image/jpeg;base64,${resized.toJPEG(80).toString("base64")}`,
          };
        } catch { return null; }
      };

      const loaded = [];
      for (const entry of data.validation_images.slice(-8)) {
        const out = { step: entry.step, conditioning: null, samples: [] };
        if (entry.conditioning) out.conditioning = loadImg(entry.conditioning);
        for (const sp of (entry.samples || [])) {
          const s = loadImg(sp);
          if (s) out.samples.push(s);
        }
        if (out.conditioning || out.samples.length > 0) loaded.push(out);
      }
      data.validation_images_loaded = loaded;
    }

    return { ok: true, ...data };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});
