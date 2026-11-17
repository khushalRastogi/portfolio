import test from 'node:test';
import assert from 'node:assert/strict';

import { listHfOnnxArtifacts } from './detector.js';

test('HF ONNX bundle includes external-data sidecars', () => {
  const files = listHfOnnxArtifacts('model.onnx');
  assert.deepEqual(files[0], 'model.onnx');
  assert.ok(files.includes('model.onnx.data'));
  assert.ok(files.includes('model.onnx.data_0'));
  assert.ok(files.includes('model.onnx.data_1'));
});