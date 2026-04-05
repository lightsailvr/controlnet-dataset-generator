const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  selectFiles: () => ipcRenderer.invoke("select-files"),
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectOutputDir: () => ipcRenderer.invoke("select-output-dir"),
  checkDependencies: () => ipcRenderer.invoke("check-dependencies"),
  checkDepthBackend: () => ipcRenderer.invoke("check-depth-backend"),
  startProcessing: (payload) => ipcRenderer.invoke("start-processing", payload),
  cancelProcessing: () => ipcRenderer.invoke("cancel-processing"),
  openFolder: (path) => ipcRenderer.invoke("open-folder", path),
  onProcessLog: (callback) => {
    const handler = (_event, msg) => callback(msg);
    ipcRenderer.on("process-log", handler);
    return () => ipcRenderer.removeListener("process-log", handler);
  },
  onProcessComplete: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on("process-complete", handler);
    return () => ipcRenderer.removeListener("process-complete", handler);
  },
  // Training
  detectHardware: () => ipcRenderer.invoke("detect-hardware"),
  prepareTrainingDataset: (payload) => ipcRenderer.invoke("prepare-training-dataset", payload),
  startTraining: (payload) => ipcRenderer.invoke("start-training", payload),
  cancelTraining: () => ipcRenderer.invoke("cancel-training"),
  getTrainingProgress: (progressFile) => ipcRenderer.invoke("get-training-progress", progressFile),
  installTrainingDeps: () => ipcRenderer.invoke("install-training-deps"),
  onTrainingLog: (callback) => {
    const handler = (_event, msg) => callback(msg);
    ipcRenderer.on("training-log", handler);
    return () => ipcRenderer.removeListener("training-log", handler);
  },
  onTrainingComplete: (callback) => {
    const handler = (_event, data) => callback(data);
    ipcRenderer.on("training-complete", handler);
    return () => ipcRenderer.removeListener("training-complete", handler);
  },
  // Review
  selectDatasetDir: () => ipcRenderer.invoke("select-dataset-dir"),
  loadDataset: (dir) => ipcRenderer.invoke("load-dataset", dir),
  generateThumbnails: (payload) => ipcRenderer.invoke("generate-thumbnails", payload),
  generateSourceThumbnails: (payload) => ipcRenderer.invoke("generate-source-thumbnails", payload),
  deleteFrames: (payload) => ipcRenderer.invoke("delete-frames", payload),
  deleteSources: (payload) => ipcRenderer.invoke("delete-sources", payload),
});
