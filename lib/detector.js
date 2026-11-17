import fs from "node:fs";
import path from "node:path";
import sharp from "sharp";
import * as ort from "onnxruntime-node";

export const SUPPORTED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

export function parseIni(content) {
  const result = {};
  let section = "default";
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith(";") || line.startsWith("#")) continue;
    if (line.startsWith("[") && line.endsWith("]")) {
      section = line.slice(1, -1).trim();
      if (!result[section]) result[section] = {};
      continue;
    }
    const eq = line.indexOf("=");
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    const val = line.slice(eq + 1).trim();
    if (!result[section]) result[section] = {};
    result[section][key] = val;
  }
  return result;
}

export function loadAppConfig(rootDir) {
  const iniPath = path.join(rootDir, "config.ini");
  const defaults = {
    model: {
      source: "local",
      dir: "model",
      onnx_file: "model.onnx",
      config_file: "config.json",
      hf_repo: "",
      hf_subpath: "",
      hf_revision: "main",
    },
    samples: {
      dir: "samples",
    },
  };
  if (!fs.existsSync(iniPath)) return defaults;
  const parsed = parseIni(fs.readFileSync(iniPath, "utf-8"));
  return {
    model: { ...defaults.model, ...(parsed.model || {}) },
    samples: { ...defaults.samples, ...(parsed.samples || {}) },
  };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

function nmsClassAgnostic(detections, nmsThre) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const cand of sorted) {
    let suppressed = false;
    for (const k of kept) {
      if (iou(cand.box, k.box) > nmsThre) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push(cand);
  }
  return kept;
}

export function listHfOnnxArtifacts(onnxFile) {
  const base = String(onnxFile || "").trim();
  if (!base) return [];

  const names = [base];
  const extraNames = [
    `${base}.data`,
    `${base}.external_data`,
  ];

  for (const extra of extraNames) {
    names.push(extra);
  }

  for (let i = 0; i < 32; i += 1) {
    names.push(`${base}.data_${i}`);
    names.push(`${base}.data_${String(i).padStart(4, "0")}`);
  }

  return [...new Set(names)];
}

async function downloadFromHf(repo, subpath, revision, filename, destPath, { allowMissing = false } = {}) {
  const prefix = subpath ? `${subpath.replace(/^\/+|\/+$/g, "")}/` : "";
  const url = `https://huggingface.co/${repo}/resolve/${revision || "main"}/${prefix}${filename}`;
  const headers = {};
  if (process.env.HF_TOKEN) {
    headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  }
  const resp = await fetch(url, { headers, redirect: "follow" });
  if (!resp.ok) {
    if (allowMissing && resp.status === 404) {
      return false;
    }
    throw new Error(`Remote artifact fetch failed (${resp.status}) for ${filename}`);
  }
  const buf = Buffer.from(await resp.arrayBuffer());
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  fs.writeFileSync(destPath, buf);
  return true;
}

export class FieldDetector {
  constructor(rootDir) {
    this.rootDir = rootDir;
    this.session = null;
    this.modelConfig = null;
    this.loadedKey = null;
    this.loadAttempted = false;
  }

  getSamplesDir() {
    const cfg = loadAppConfig(this.rootDir);
    return path.resolve(this.rootDir, cfg.samples.dir || "samples");
  }

  listSamples() {
    const samplesDir = this.getSamplesDir();
    if (!fs.existsSync(samplesDir)) return [];

    let demoMeta = {};
    const demoPath = path.join(samplesDir, "demo_annotations.json");
    if (fs.existsSync(demoPath)) {
      try {
        demoMeta = JSON.parse(fs.readFileSync(demoPath, "utf-8"));
      } catch {
        demoMeta = {};
      }
    }

    const entries = fs.readdirSync(samplesDir, { withFileTypes: true });
    return entries
      .filter((e) => e.isFile() && SUPPORTED_EXTENSIONS.has(path.extname(e.name).toLowerCase()))
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((e) => {
        const meta = demoMeta[e.name];
        const fallbackTitle = path
          .basename(e.name, path.extname(e.name))
          .replace(/[_-]+/g, " ")
          .replace(/\b\w/g, (ch) => ch.toUpperCase());
        return {
          id: e.name,
          name: meta?.name || fallbackTitle,
          url: `/samples/${encodeURIComponent(e.name)}`,
        };
      });
  }

  async ensureModelLoaded() {
    const cfg = loadAppConfig(this.rootDir);
    const m = cfg.model;
    const key = JSON.stringify(m);

    if (this.session && this.loadedKey === key) {
      return true;
    }

    let onnxPath = null;
    let configPath = null;

    if (m.source === "huggingface" && m.hf_repo) {
      const safeRepo = m.hf_repo.replace(/[^a-zA-Z0-9._-]/g, "_");
      const safeSub = (m.hf_subpath || "root").replace(/[^a-zA-Z0-9._-]/g, "_");
      const cacheDir = path.join(this.rootDir, ".model-cache", safeRepo, safeSub);
      const onnxName = m.onnx_file || "model.onnx";
      const configName = m.config_file || "config.json";
      onnxPath = path.join(cacheDir, onnxName);
      configPath = path.join(cacheDir, configName);
      const onnxBundle = listHfOnnxArtifacts(onnxName);
      const missingBundleFiles = [configPath, ...onnxBundle.map((artifactName) => path.join(cacheDir, artifactName))].filter(
        (filePath) => !fs.existsSync(filePath)
      );

      if (missingBundleFiles.length > 0) {
        try {
          if (!fs.existsSync(configPath)) {
            await downloadFromHf(m.hf_repo, m.hf_subpath, m.hf_revision, configName, configPath);
          }

          for (const artifactName of onnxBundle) {
            const destPath = path.join(cacheDir, artifactName);
            if (fs.existsSync(destPath)) continue;
            const downloaded = await downloadFromHf(m.hf_repo, m.hf_subpath, m.hf_revision, artifactName, destPath, {
              allowMissing: artifactName !== onnxName,
            });
            if (artifactName === onnxName && !downloaded) {
              throw new Error(`Missing required ONNX artifact: ${artifactName}`);
            }
          }

          if (!fs.existsSync(onnxPath)) {
            throw new Error(`Missing required ONNX artifact: ${onnxName}`);
          }
        } catch {
          this.session = null;
          this.modelConfig = null;
          this.loadedKey = key;
          return false;
        }
      }
    } else {
      const modelDir = path.resolve(this.rootDir, m.dir || "model");
      if (fs.existsSync(modelDir)) {
        const preferredOnnx = path.join(modelDir, m.onnx_file || "model.onnx");
        if (fs.existsSync(preferredOnnx)) {
          onnxPath = preferredOnnx;
        } else {
          const anyOnnx = fs.readdirSync(modelDir).find((f) => f.toLowerCase().endsWith(".onnx"));
          if (anyOnnx) onnxPath = path.join(modelDir, anyOnnx);
        }
        const preferredCfg = path.join(modelDir, m.config_file || "config.json");
        if (fs.existsSync(preferredCfg)) {
          configPath = preferredCfg;
        }
      }
    }

    if (!onnxPath || !fs.existsSync(onnxPath) || !configPath || !fs.existsSync(configPath)) {
      this.session = null;
      this.modelConfig = null;
      this.loadedKey = key;
      return false;
    }

    const rawCfg = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    this.modelConfig = {
      idx2label: rawCfg.idx2label || {},
      conf_thre: typeof rawCfg.conf_thre === "number" ? rawCfg.conf_thre : 0.5,
      nms_thre: typeof rawCfg.nms_thre === "number" ? rawCfg.nms_thre : 0.45,
      test_size: Array.isArray(rawCfg.test_size) && rawCfg.test_size.length === 2 ? rawCfg.test_size : [640, 640],
    };

    try {
      this.session = await ort.InferenceSession.create(onnxPath, {
        executionProviders: ["cpu"],
      });
    } catch (error) {
      console.error("Failed to initialize ONNX model:", error);
      this.session = null;
      this.modelConfig = null;
      this.loadedKey = key;
      return false;
    }

    this.loadedKey = key;
    return true;
  }

  async getPublicConfig() {
    const ready = await this.ensureModelLoaded();
    if (ready && this.modelConfig) {
      return {
        modelReady: true,
        mode: "live",
        defaultConfThre: this.modelConfig.conf_thre,
        labels: Object.values(this.modelConfig.idx2label),
      };
    }

    const demoLabels = new Set();
    const demoPath = path.join(this.getSamplesDir(), "demo_annotations.json");
    if (fs.existsSync(demoPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(demoPath, "utf-8"));
        for (const item of Object.values(data)) {
          for (const d of item.detections || []) {
            demoLabels.add(d.label);
          }
        }
      } catch {
        // ignore malformed demo annotations
      }
    }

    return {
      modelReady: false,
      mode: "demo",
      defaultConfThre: 0.5,
      labels: [...demoLabels],
    };
  }

  async detectFromBuffer(imageBuffer, sampleId = null) {
    const started = performance.now();
    const imgSharp = sharp(imageBuffer, { failOn: "none" }).rotate();
    const meta = await imgSharp.metadata();
    const origW = meta.width || 1;
    const origH = meta.height || 1;

    let ready = false;
    try {
      ready = await this.ensureModelLoaded();
    } catch (error) {
      console.error("Model initialization failed during detection:", error);
      ready = false;
    }

    if (!ready || !this.session || !this.modelConfig) {
      if (sampleId) {
        const demoPath = path.join(this.getSamplesDir(), "demo_annotations.json");
        if (fs.existsSync(demoPath)) {
          const demoData = JSON.parse(fs.readFileSync(demoPath, "utf-8"));
          const entry = demoData[sampleId];
          if (entry && Array.isArray(entry.detections)) {
            return {
              bboxes: entry.detections.map((d) => d.bbox),
              label: entry.detections.map((d) => d.label),
              scores: entry.detections.map((d) => d.score),
              imageSize: { width: origW, height: origH },
              defaultConfThre: 0.5,
              inferenceMs: Math.round(performance.now() - started),
              mode: "demo",
              notice: "Showing precomputed sample detections. Connect a live model in config.ini to run inference on custom uploads.",
            };
          }
        }
      }

      return {
        bboxes: [],
        label: [],
        scores: [],
        imageSize: { width: origW, height: origH },
        defaultConfThre: 0.5,
        inferenceMs: Math.round(performance.now() - started),
        mode: "unconfigured",
        notice: "No live detection model is currently configured on the server. Try the sample IDs above or configure a model source in config.ini.",
      };
    }

    const [testH, testW] = this.modelConfig.test_size;
    const ratio = Math.min(testH / origH, testW / origW);
    const resizedW = Math.max(1, Math.min(testW, Math.round(origW * ratio)));
    const resizedH = Math.max(1, Math.min(testH, Math.round(origH * ratio)));

    const rgbBuffer = await imgSharp
      .removeAlpha()
      .toColourspace("srgb")
      .resize(resizedW, resizedH, { fit: "fill" })
      .extend({
        top: 0,
        left: 0,
        bottom: testH - resizedH,
        right: testW - resizedW,
        background: { r: 114, g: 114, b: 114 },
      })
      .raw()
      .toBuffer();

    const planeSize = testH * testW;
    const chw = new Float32Array(3 * planeSize);
    for (let i = 0; i < planeSize; i++) {
      const idx = i * 3;
      chw[i] = rgbBuffer[idx];
      chw[planeSize + i] = rgbBuffer[idx + 1];
      chw[2 * planeSize + i] = rgbBuffer[idx + 2];
    }

    const inputName = this.session.inputNames[0];
    const inputTensor = new ort.Tensor("float32", chw, [1, 3, testH, testW]);
    const outputs = await this.session.run({ [inputName]: inputTensor });
    const outputTensor = outputs[this.session.outputNames[0]];
    const data = outputTensor.data;
    const dims = outputTensor.dims;

    const numAnchors = dims.length === 3 ? dims[1] : dims[0];
    const stride = dims.length === 3 ? dims[2] : dims[1];
    const numClasses = stride - 5;
    const floorConf = Math.min(0.15, this.modelConfig.conf_thre);

    const candidates = [];
    for (let i = 0; i < numAnchors; i++) {
      const offset = i * stride;
      const objConf = data[offset + 4];
      if (objConf < floorConf) continue;

      let bestCls = 0;
      let bestClsProb = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const p = data[offset + 5 + c];
        if (p > bestClsProb) {
          bestClsProb = p;
          bestCls = c;
        }
      }

      const score = objConf * bestClsProb;
      if (score < floorConf) continue;

      const cx = data[offset];
      const cy = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];

      const x1 = Math.max(0, Math.min(origW, (cx - w / 2) / ratio));
      const y1 = Math.max(0, Math.min(origH, (cy - h / 2) / ratio));
      const x2 = Math.max(0, Math.min(origW, (cx + w / 2) / ratio));
      const y2 = Math.max(0, Math.min(origH, (cy + h / 2) / ratio));

      candidates.push({
        box: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
        clsId: bestCls,
        score: Number(score.toFixed(4)),
      });
    }

    const kept = nmsClassAgnostic(candidates, this.modelConfig.nms_thre);
    const idx2label = this.modelConfig.idx2label;

    return {
      bboxes: kept.map((k) => k.box),
      label: kept.map((k) => idx2label[String(k.clsId)] || `field_${k.clsId}`),
      scores: kept.map((k) => k.score),
      imageSize: { width: origW, height: origH },
      defaultConfThre: this.modelConfig.conf_thre,
      inferenceMs: Math.round(performance.now() - started),
      mode: "live",
      notice: null,
    };
  }
}
