const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  selectFiles: () => ipcRenderer.invoke("select-files"),
  selectFolder: () => ipcRenderer.invoke("select-folder"),
  selectOutputDir: () => ipcRenderer.invoke("select-output-dir"),
  checkDependencies: () => ipcRenderer.invoke("check-dependencies"),
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
  selectDatasetDir: () => ipcRenderer.invoke("select-dataset-dir"),
  loadDataset: (dir) => ipcRenderer.invoke("load-dataset", dir),
  generateThumbnails: (payload) => ipcRenderer.invoke("generate-thumbnails", payload),
  deleteFrames: (payload) => ipcRenderer.invoke("delete-frames", payload),
});
