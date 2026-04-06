const { app, BrowserWindow, ipcMain, dialog, shell, nativeImage } = require("electron");
const path = require("path");
const { spawn, spawnSync } = require("child_process");
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

/** Whether the current dataset job was started via WSL (for logging / future use). */
let pythonProcessViaWSL = false;

// ─── WSL2 bridge (Windows + FoundationStereo in Linux conda env) ───

let _wslAvailableCache = null;
function isWSLAvailable() {
  if (_wslAvailableCache !== null) return _wslAvailableCache;
  if (process.platform !== "win32") {
    _wslAvailableCache = false;
    return false;
  }
  try {
    const r = spawnSync("wsl.exe", ["-l", "-q"], {
      encoding: "utf-8",
      timeout: 8000,
      windowsHide: true,
    });
    _wslAvailableCache = r.status === 0 && Boolean((r.stdout || "").trim());
  } catch {
    _wslAvailableCache = false;
  }
  return _wslAvailableCache;
}

/** Convert Windows path to WSL /mnt/x/... */
function winPathToWSL(winPath) {
  if (!winPath || typeof winPath !== "string") return winPath;
  const resolved = path.win32.resolve(winPath);
  const m = /^([a-zA-Z]):[\\/]?(.*)$/i.exec(resolved);
  if (!m) return winPath;
  const letter = m[1].toLowerCase();
  const rest = (m[2] || "").replace(/\\/g, "/").replace(/^\//, "");
  return rest ? `/mnt/${letter}/${rest}` : `/mnt/${letter}`;
}

let _wslCondaEnvCache = null;
/**
 * Check whether `conda run -n foundation_stereo` works inside default WSL distro.
 * Cached for the session.
 */
/** @type {string[]} Collects debug lines for the current probe cycle. */
let _probeDebug = [];

function wslFoundationStereoCondaOk() {
  if (_wslCondaEnvCache !== null) return _wslCondaEnvCache;
  if (process.platform !== "win32" || !isWSLAvailable()) {
    _wslCondaEnvCache = { ok: false, reason: "wsl_unavailable" };
    _probeDebug.push(`[conda-probe] skip: ${JSON.stringify(_wslCondaEnvCache)}`);
    return _wslCondaEnvCache;
  }
  const condaCmd = `${WSL_CONDA_INIT} conda run --no-capture-output -n foundation_stereo python3 -c "import torch; print(torch.cuda.is_available())"`;
  _probeDebug.push(`[conda-probe] cmd: ${condaCmd}`);
  try {
    const r = spawnSync(
      "wsl.exe",
      ["bash", "-lc", condaCmd],
      { encoding: "utf-8", timeout: 120000, windowsHide: true }
    );
    _probeDebug.push(`[conda-probe] status: ${r.status}`);
    _probeDebug.push(`[conda-probe] stdout: ${(r.stdout || "").slice(0, 800)}`);
    _probeDebug.push(`[conda-probe] stderr: ${(r.stderr || "").slice(0, 800)}`);
    if (r.error) _probeDebug.push(`[conda-probe] error: ${r.error.message}`);
    const out = ((r.stdout || "") + (r.stderr || "")).trim();
    const cudaOk = out.includes("True");
    _wslCondaEnvCache = {
      ok: r.status === 0,
      cuda: cudaOk,
      reason: r.status === 0 ? null : out || `exit ${r.status}`,
    };
  } catch (e) {
    _probeDebug.push(`[conda-probe] exception: ${e.message || String(e)}`);
    _wslCondaEnvCache = { ok: false, reason: e.message || String(e) };
  }
  _probeDebug.push(`[conda-probe] result: ${JSON.stringify(_wslCondaEnvCache)}`);
  return _wslCondaEnvCache;
}

function depthBackendNeedsWSLFoundationStereo(depthBackend) {
  const b = (depthBackend || "auto").toLowerCase();
  return b === "foundation_stereo" || b === "auto";
}

function translateJobForWSL(job) {
  const out = JSON.parse(JSON.stringify(job));
  out.files = (out.files || []).map((f) => winPathToWSL(f));
  out.outputDir = winPathToWSL(out.outputDir);
  const cfg = out.config || {};
  if (cfg.foundationStereoRoot && String(cfg.foundationStereoRoot).trim()) {
    cfg.foundationStereoRoot = winPathToWSL(path.win32.resolve(cfg.foundationStereoRoot.trim()));
  }
  if (cfg.foundationStereoCkpt && String(cfg.foundationStereoCkpt).trim()) {
    cfg.foundationStereoCkpt = winPathToWSL(path.win32.resolve(cfg.foundationStereoCkpt.trim()));
  }
  out.config = cfg;
  return out;
}

/** Bash-safe single-quoted string (paths with apostrophes). */
function bashSingleQuote(s) {
  return "'" + String(s).replace(/'/g, "'\\''") + "'";
}

/**
 * Shell snippet that initializes conda in a non-interactive bash -lc context.
 * ~/.bashrc usually guards behind an interactive check, so `conda` is missing.
 * We source conda.sh directly from common install locations.
 */
const WSL_CONDA_INIT = [
  'for _d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "/opt/conda";',
  "do",
  '  if [ -f "$_d/etc/profile.d/conda.sh" ]; then',
  '    . "$_d/etc/profile.d/conda.sh"; break;',
  "  fi;",
  "done;",
  "unset _d;",
].join(" ");

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

ipcMain.handle("check-depth-backend", async () => {
  _wslCondaEnvCache = null;
  _probeDebug = [];
  _probeDebug.push(`[probe] platform: ${process.platform}`);
  const repoRoot = path.join(__dirname, "..");
  const probePath = path.join(repoRoot, "python", "depth_backend_probe.py");
  _probeDebug.push(`[probe] probePath: ${probePath} exists: ${fs.existsSync(probePath)}`);
  if (!fs.existsSync(probePath)) {
    return { ok: false, error: "depth_backend_probe.py missing", _debug: _probeDebug };
  }

  const pythonPath = findPython();
  _probeDebug.push(`[probe] native pythonPath: ${pythonPath}`);
  let data = { ok: false, error: "Python 3 not found" };
  if (pythonPath) {
    const r = spawnSync(pythonPath, [probePath], {
      encoding: "utf-8",
      timeout: 120000,
      cwd: repoRoot,
    });
    _probeDebug.push(`[probe] native status: ${r.status} error: ${r.error?.message || "none"}`);
    _probeDebug.push(`[probe] native stdout: ${(r.stdout || "").slice(0, 800)}`);
    _probeDebug.push(`[probe] native stderr: ${(r.stderr || "").slice(0, 800)}`);
    if (r.error) {
      data = { ok: false, error: r.error.message };
    } else if (r.status !== 0) {
      data = {
        ok: false,
        error: (r.stderr || r.stdout || `exit ${r.status}`).trim(),
      };
    } else {
      try {
        data = { ok: true, ...JSON.parse((r.stdout || "").trim()) };
      } catch (e) {
        data = { ok: false, error: e.message || "invalid JSON from native probe" };
      }
    }
  }

  const nativeFsReady = data.foundation_stereo_ready === true;
  _probeDebug.push(`[probe] nativeFsReady: ${nativeFsReady}`);

  if (process.platform === "win32") {
    _probeDebug.push("[probe] === entering WSL section ===");
    const conda = wslFoundationStereoCondaOk();
    const thirdParty = path.join(repoRoot, "third_party", "FoundationStereo");
    const thirdPartyExists = fs.existsSync(thirdParty);
    _probeDebug.push(`[probe] thirdParty: ${thirdParty} exists: ${thirdPartyExists}`);
    const envPrefix = thirdPartyExists
      ? `export FOUNDATION_STEREO_ROOT=${bashSingleQuote(winPathToWSL(thirdParty))} && `
      : "";
    const probeWsl = winPathToWSL(probePath);
    const cmd = `${WSL_CONDA_INIT} ${envPrefix}conda run --no-capture-output -n foundation_stereo python3 ${bashSingleQuote(probeWsl)}`;
    _probeDebug.push(`[probe] WSL probe cmd: ${cmd}`);
    const wr = spawnSync("wsl.exe", ["bash", "-lc", cmd], {
      encoding: "utf-8",
      timeout: 120000,
      windowsHide: true,
    });
    _probeDebug.push(`[probe] WSL probe status: ${wr.status}`);
    _probeDebug.push(`[probe] WSL probe stdout: ${(wr.stdout || "").slice(0, 1000)}`);
    _probeDebug.push(`[probe] WSL probe stderr: ${(wr.stderr || "").slice(0, 1000)}`);
    if (wr.error) _probeDebug.push(`[probe] WSL probe spawn error: ${wr.error.message}`);
    let wslProbe = null;
    if (!wr.error && wr.status === 0) {
      try {
        wslProbe = JSON.parse((wr.stdout || "").trim());
      } catch (parseErr) {
        _probeDebug.push(`[probe] WSL JSON parse fail: ${parseErr.message} raw: ${(wr.stdout || "").slice(0, 300)}`);
        wslProbe = null;
      }
    }
    const wslFsReady = Boolean(wslProbe && wslProbe.foundation_stereo_ready === true);
    _probeDebug.push(`[probe] wslProbe: ${JSON.stringify(wslProbe)} wslFsReady: ${wslFsReady}`);
    data = {
      ...data,
      native_foundation_stereo_ready: nativeFsReady,
      native_cuda: data.cuda === true,
      wsl_installed: isWSLAvailable(),
      wsl_conda_ok: conda.ok,
      wsl_conda_cuda: conda.cuda,
    };
    if (wslProbe) {
      data.wsl_cuda = wslProbe.cuda;
      data.wsl_foundation_stereo_ready = wslFsReady;
      data.wsl_foundation_stereo_root = wslProbe.foundation_stereo_root;
    }
    if (!nativeFsReady && wslFsReady) {
      data.foundation_stereo_ready = true;
      data.cuda = wslProbe.cuda;
      data.via_wsl = true;
      data.ok = true;
    }
    if (isWSLAvailable() && !conda.ok) {
      data.wsl_env_hint =
        "WSL2 is installed but the foundation_stereo conda env is not ready. Open Ubuntu (WSL) and run: bash scripts/setup_depth_env.sh";
    }
    if (isWSLAvailable() && conda.ok && !wslFsReady && !nativeFsReady) {
      data.wsl_weights_hint =
        "Conda env OK but weights or cfg.yaml not found. Download 23-51-11 into third_party/FoundationStereo/pretrained_models/ (see README).";
    }
  }

  data._debug = _probeDebug;
  return data;
});

ipcMain.handle("start-processing", async (event, { files, config, outputDir }) => {
  const repoRoot = path.join(__dirname, "..");
  const scriptPath = path.join(repoRoot, "python", "equirect_dataset_generator.py");

  const fileList = files.map((f) => f.path);
  const jobPayload = { files: fileList, config, outputDir };

  const useWsl =
    process.platform === "win32" &&
    isWSLAvailable() &&
    wslFoundationStereoCondaOk().ok &&
    depthBackendNeedsWSLFoundationStereo(config.depthBackend);

  const depthExplicit = (config.depthBackend || "").toLowerCase() === "foundation_stereo";
  if (depthExplicit && !useWsl) {
    const conda = wslFoundationStereoCondaOk();
    if (!isWSLAvailable()) {
      return {
        ok: false,
        error:
          "FoundationStereo on Windows requires WSL2. Install Ubuntu from Microsoft Store, then run scripts/setup_depth_env.sh inside WSL.",
      };
    }
    return {
      ok: false,
      error: conda.reason
        ? `WSL foundation_stereo env not ready: ${conda.reason}`
        : "WSL foundation_stereo conda env or CUDA not available. Run scripts/setup_depth_env.sh inside WSL (see README).",
    };
  }

  let pythonPath = null;
  if (!useWsl) {
    pythonPath = findPython();
    if (!pythonPath) {
      return { ok: false, error: "Python 3 not found" };
    }
  }

  const jobFile = path.join(os.tmpdir(), `controlnet_job_${Date.now()}.json`);
  const jobToWrite = useWsl ? translateJobForWSL(jobPayload) : jobPayload;
  fs.writeFileSync(jobFile, JSON.stringify(jobToWrite));

  pythonProcessViaWSL = Boolean(useWsl);

  if (useWsl) {
    const jobWsl = winPathToWSL(jobFile);
    const scriptWsl = winPathToWSL(scriptPath);
    const thirdParty = path.join(repoRoot, "third_party", "FoundationStereo");
    const cfg = config || {};
    let fsRootWsl = "";
    if (cfg.foundationStereoRoot && String(cfg.foundationStereoRoot).trim()) {
      fsRootWsl = winPathToWSL(path.win32.resolve(String(cfg.foundationStereoRoot).trim()));
    } else if (fs.existsSync(thirdParty)) {
      fsRootWsl = winPathToWSL(thirdParty);
    }
    const envExports = fsRootWsl
      ? `export FOUNDATION_STEREO_ROOT=${bashSingleQuote(fsRootWsl)} && `
      : "";
    const cmd = `${WSL_CONDA_INIT} ${envExports}conda run --no-capture-output -n foundation_stereo python3 ${bashSingleQuote(
      scriptWsl
    )} --job-file ${bashSingleQuote(jobWsl)}`;
    pythonProcess = spawn("wsl.exe", ["bash", "-lc", cmd], {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      windowsHide: true,
    });
  } else {
    const args = [scriptPath, "--job-file", jobFile];
    pythonProcess = spawn(pythonPath, args, {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
  }

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
    pythonProcessViaWSL = false;
  });

  return { ok: true };
});

ipcMain.handle("cancel-processing", async () => {
  if (pythonProcess) {
    const proc = pythonProcess;
    const viaWsl = pythonProcessViaWSL;
    try {
      proc.kill("SIGTERM");
      if (process.platform === "win32" && viaWsl) {
        setTimeout(() => {
          try {
            proc.kill("SIGKILL");
          } catch {
            /* ignore */
          }
        }, 3000);
      }
    } catch {
      /* ignore */
    }
    pythonProcess = null;
    pythonProcessViaWSL = false;
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
  const reqFile = isMac ? "requirements.txt" : "requirements_nvidia.txt";
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
