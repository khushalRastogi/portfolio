import express from "express";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import multer from "multer";
import { FieldDetector, SUPPORTED_EXTENSIONS } from "./lib/detector.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const envPath = path.join(__dirname, ".env");
if (fs.existsSync(envPath) && typeof process.loadEnvFile === "function") {
  process.loadEnvFile(envPath);
}

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const HOST = "0.0.0.0";

// Vercel may not expose VERCEL at runtime; its functions always run from /var/task on Lambda.
const ON_VERCEL = Boolean(
  process.env.VERCEL ||
    process.env.LAMBDA_TASK_ROOT ||
    process.env.AWS_LAMBDA_FUNCTION_NAME ||
    __dirname.startsWith("/var/task")
);
const BACKEND_URL = process.env.RENDER
  ? ""
  : process.env.API_BACKEND_URL || ON_VERCEL
    ? (process.env.API_BACKEND_URL || "https://portfolio-mkk7.onrender.com").replace(/\/$/, "")
    : "";

async function proxyToBackend(req, res) {
  try {
    const headers = {};
    for (const name of ["content-type", "accept"]) {
      if (req.headers[name]) headers[name] = req.headers[name];
    }
    let body;
    if (!["GET", "HEAD"].includes(req.method)) {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      body = Buffer.concat(chunks);
    }
    const upstream = await fetch(`${BACKEND_URL}${req.originalUrl}`, {
      method: req.method,
      headers,
      body,
      signal: AbortSignal.timeout(120000),
    });
    res.status(upstream.status);
    res.setHeader("x-served-by", "render-via-vercel");
    for (const name of ["content-type", "cache-control", "etag"]) {
      const v = upstream.headers.get(name);
      if (v) res.setHeader(name, v);
    }
    res.send(Buffer.from(await upstream.arrayBuffer()));
  } catch {
    res.status(502).json({ error: "Detection backend is unavailable. Please retry in a minute." });
  }
}

const detector = new FieldDetector(__dirname);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype && file.mimetype.startsWith("image/")) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are supported."));
    }
  },
});

if (BACKEND_URL) {
  app.use(["/api", "/samples"], proxyToBackend);
}

app.use(express.json({ limit: "1mb" }));

app.use((req, res, next) => {
  const origin = req.get("Origin");
  if (origin) {
    try {
      const hostname = new URL(origin).hostname;
      if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1") {
        res.setHeader("Access-Control-Allow-Origin", origin);
        res.setHeader("Vary", "Origin");
        res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
        res.setHeader("Access-Control-Allow-Headers", "Content-Type");
      }
    } catch {
      // Ignore malformed Origin headers.
    }
  }
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// Security headers for browser responses
app.use((req, res, next) => {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
  next();
});

// Block static access to internal configuration, weights, scripts, and server files
const BLOCKED_PREFIXES = ["/config.ini", "/model", "/scripts", "/lib", "/server.js", "/package.json", "/package-lock.json", "/.env", "/.git", "/.aistudio", "/.model-cache"];
const BLOCKED_EXTS = new Set([".ini", ".py", ".onnx", ".pth", ".pt", ".json"]);

app.use((req, res, next) => {
  const reqPath = req.path.toLowerCase();
  if (reqPath.startsWith("/api/")) return next();
  if (BLOCKED_PREFIXES.some((p) => reqPath === p || reqPath.startsWith(`${p}/`))) {
    return res.status(404).end();
  }
  const ext = path.extname(reqPath);
  if (BLOCKED_EXTS.has(ext) && !reqPath.startsWith("/fonts/")) {
    return res.status(404).end();
  }
  next();
});

app.use(
  "/fonts",
  express.static(path.join(__dirname, "fonts"), {
    maxAge: "365d",
    immutable: true,
  })
);

// Serve sample ID images safely from configured samples directory
app.get("/samples/:filename", (req, res) => {
  const safeName = path.basename(req.params.filename);
  const ext = path.extname(safeName).toLowerCase();
  if (!SUPPORTED_EXTENSIONS.has(ext)) {
    return res.status(404).end();
  }
  const filePath = path.join(detector.getSamplesDir(), safeName);
  if (!fs.existsSync(filePath)) {
    return res.status(404).end();
  }
  res.sendFile(filePath);
});

app.get("/api/workbench-config", async (_req, res) => {
  try {
    const cfg = await detector.getPublicConfig();
    res.json(cfg);
  } catch {
    res.status(500).json({ error: "Unable to initialize detection configuration." });
  }
});

app.get("/api/samples", (_req, res) => {
  try {
    res.json({ samples: detector.listSamples() });
  } catch {
    res.status(500).json({ error: "Unable to list sample images." });
  }
});

app.post("/api/detect", (req, res) => {
  upload.single("image")(req, res, async (uploadErr) => {
    if (uploadErr) {
      return res.status(400).json({ error: uploadErr.message || "Invalid image upload." });
    }
    try {
      if (req.file && req.file.buffer) {
        const result = await detector.detectFromBuffer(req.file.buffer, null);
        return res.json(result);
      }

      const sampleId = req.body?.sampleId ? path.basename(String(req.body.sampleId)) : null;
      if (!sampleId) {
        return res.status(400).json({ error: "Provide an uploaded image or a sampleId." });
      }

      const ext = path.extname(sampleId).toLowerCase();
      if (!SUPPORTED_EXTENSIONS.has(ext)) {
        return res.status(400).json({ error: "Unsupported sample image format." });
      }

      const samplePath = path.join(detector.getSamplesDir(), sampleId);
      if (!fs.existsSync(samplePath)) {
        return res.status(404).json({ error: "Sample image not found." });
      }

      const buf = fs.readFileSync(samplePath);
      const result = await detector.detectFromBuffer(buf, sampleId);
      return res.json(result);
    } catch {
      return res.status(500).json({ error: "Field detection failed for this image." });
    }
  });
});

app.get("/try-it-out", (_req, res) => {
  res.sendFile(path.join(__dirname, "try-it-out.html"));
});

app.use(
  express.static(__dirname, {
    extensions: ["html"],
    index: "index.html",
  })
);

app.get("/{*splat}", (_req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(PORT, HOST, () => {
  console.log(`Server listening on http://${HOST}:${PORT}`);
});
