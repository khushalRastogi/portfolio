import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const samplesDir = path.join(rootDir, "samples");
const outputPath = path.join(samplesDir, "demo_annotations.json");
const baseUrl = (process.env.API_BACKEND_URL || process.argv[2] || "https://portfolio-mkk7.onrender.com").replace(/\/$/, "");
const supported = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

const sampleFiles = fs.readdirSync(samplesDir, { withFileTypes: true })
  .filter((entry) => entry.isFile() && supported.has(path.extname(entry.name).toLowerCase()))
  .map((entry) => entry.name)
  .sort((a, b) => a.localeCompare(b));

if (sampleFiles.length === 0) throw new Error("No sample images found.");

const previous = fs.existsSync(outputPath)
  ? JSON.parse(fs.readFileSync(outputPath, "utf8"))
  : {};
const results = {};

for (const [index, sampleId] of sampleFiles.entries()) {
  const response = await fetch(`${baseUrl}/api/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sampleId }),
    signal: AbortSignal.timeout(180000),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(`Failed for ${sampleId}: ${data.error || response.statusText}`);
  if (data.mode !== "live") throw new Error(`Expected live inference for ${sampleId}, received ${data.mode}.`);
  if (!Array.isArray(data.bboxes) || !Array.isArray(data.label) || !Array.isArray(data.scores)) {
    throw new Error(`Incomplete detection response for ${sampleId}.`);
  }

  results[sampleId] = {
    name: previous[sampleId]?.name || sampleId
      .replace(/\.[^.]+$/, "")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (character) => character.toUpperCase()),
    imageSize: data.imageSize,
    defaultConfThre: data.defaultConfThre,
    detections: data.bboxes.map((bbox, detectionIndex) => ({
      label: data.label[detectionIndex] || `field_${detectionIndex}`,
      score: data.scores[detectionIndex],
      bbox,
    })),
  };

  console.log(`[${index + 1}/${sampleFiles.length}] Cached ${sampleId} (${results[sampleId].detections.length} detections)`);
}

fs.writeFileSync(outputPath, `${JSON.stringify(results, null, 2)}\n`);
console.log(`Wrote ${sampleFiles.length} sample results to ${outputPath}`);
