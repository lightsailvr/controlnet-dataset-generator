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
  // Try common Python 3 paths on macOS
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

ipcMain.handle("check-dependencies", async () => {
  const pythonPath = findPython();
  if (!pythonPath) {
    return { ok: false, error: "python3_not_found", pythonPath: null };
  }

  // Check for required packages
  try {
    const result = require("child_process").execSync(
      `${pythonPath} -c "import py360convert; import cv2; import numpy; print('ok')"`,
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
  const pairsPath = path.join(dir, "pairs.json");
  if (!fs.existsSync(pairsPath)) {
    return { error: "no_pairs_json", message: "Selected folder does not contain a pairs.json manifest." };
  }
  return { ok: true, path: dir };
});

ipcMain.handle("load-dataset", async (event, dir) => {
  try {
    const pairsPath = path.join(dir, "pairs.json");
    const manifest = JSON.parse(fs.readFileSync(pairsPath, "utf-8"));

    // Group pairs by source_frame to get unique frames
    const frameMap = new Map();
    for (const pair of manifest.pairs) {
      const name = pair.source_frame;
      if (!frameMap.has(name)) {
        frameMap.set(name, { name, pairCount: 0, sourceFile: pair.source_file });
      }
      frameMap.get(name).pairCount++;
    }

    const frames = [];
    for (const [name, info] of frameMap) {
      const targetPath = path.join(dir, "target", `${name}.png`);
      if (fs.existsSync(targetPath)) {
        frames.push({
          name,
          targetPath,
          pairCount: info.pairCount,
          sourceFile: info.sourceFile,
        });
      }
    }

    return {
      ok: true,
      frames,
      totalPairs: manifest.total_pairs || manifest.pairs.length,
      totalFrames: frames.length,
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

  const thumbnails = {};
  for (const name of frameNames) {
    const cachedPath = path.join(thumbDir, `${name}.jpg`);

    // Check cache first
    if (fs.existsSync(cachedPath)) {
      const buf = fs.readFileSync(cachedPath);
      thumbnails[name] = `data:image/jpeg;base64,${buf.toString("base64")}`;
      continue;
    }

    // Generate thumbnail from target image
    const targetPath = path.join(datasetDir, "target", `${name}.png`);
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

ipcMain.handle("delete-frames", async (event, { datasetDir, frameNames }) => {
  try {
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
