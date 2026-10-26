// Prompt2Rig — tiny zero-dependency server.
// Serves ./public and proxies fal.ai so the API key never reaches the browser.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, 'public');
const TAKES = path.join(ROOT, 'takes'); // exported takes for tools/render_blender.py

// --- .env -------------------------------------------------------------------
const envFile = path.join(ROOT, '.env');
if (fs.existsSync(envFile)) {
  for (const line of fs.readFileSync(envFile, 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
}
const FAL_KEY = process.env.FAL_KEY;
const PORT = Number(process.env.PORT || 5177);
const HQ_URL = process.env.HQ_URL || 'http://127.0.0.1:5178';
if (!FAL_KEY) console.warn('[prompt2rig] FAL_KEY is not set — generation will fail. Put it in .env');

const VIDEO_MODELS = {
  // MiniMax H3 Max Turbo — faster than real time (~0.5s GPU for 5s @ 480P)
  h3: {
    t2v: 'minimax/h3-max-turbo/text-to-video',
    i2v: 'minimax/h3-max-turbo/image-to-video',
    input: (b, i2v) => ({
      duration: Math.min(15, Math.max(1, Number(b.duration) || 5)),
      resolution: ['480P', '768P', '1080P'].includes(b.resolution) ? b.resolution : '480P',
      prompt_expansion_mode: 'disabled',
      ...(i2v ? {} : { aspect_ratio: b.aspect_ratio || '16:9' }),
    }),
  },
  // FLUX 3 Max — higher fidelity, ~real time (5s @ 720p ≈ 4.5s GPU, ~9s wall)
  flux3: {
    t2v: 'blackforestlabs/flux-3-max/text-to-video',
    i2v: 'blackforestlabs/flux-3-max/image-to-video',
    input: (b, i2v) => ({
      duration: Math.min(20, Math.max(5, Math.round(Number(b.duration) || 5))),
      resolution: b.resolution === '1080P' ? '1080p' : '720p',
      generate_audio: false,
      reasoning_level: b.reasoning_level === 'high' ? 'high' : 'low',
      aspect_ratio: i2v ? 'auto' : (b.aspect_ratio || '16:9'),
    }),
  },
};
const IMAGE_MODEL = 'fal-ai/flux/schnell';

async function falRun(model, input, keyOverride) {
  const t0 = performance.now();
  const key = keyOverride || FAL_KEY;
  if (!key) throw Object.assign(new Error('FAL_KEY is not set. Please provide it in Settings or .env'), { status: 401 });
  const res = await fetch(`https://fal.run/${model}`, {
    method: 'POST',
    headers: { Authorization: `Key ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = { raw: text }; }
  if (!res.ok) {
    const detail = json?.detail ? JSON.stringify(json.detail) : text.slice(0, 500);
    throw Object.assign(new Error(`fal ${model} → ${res.status}: ${detail}`), { status: res.status });
  }
  return { ...json, _elapsed: (performance.now() - t0) / 1000 };
}

// --- request handling ----------------------------------------------------------
function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0; const chunks = [];
    req.on('data', (c) => { size += c.length; if (size > 25e6) reject(new Error('body too large')); else chunks.push(c); });
    req.on('end', () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString() || '{}')); } catch (e) { reject(e); } });
    req.on('error', reject);
  });
}

function send(res, status, body) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': '*',
  });
  res.end(JSON.stringify(body));
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css',
  '.json': 'application/json', '.glb': 'model/gltf-binary', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime', '.woff2': 'font/woff2',
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');

  // Handle CORS Preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': '*',
    });
    return res.end();
  }

  const clientFalKey = req.headers['x-fal-key'] || (req.headers.authorization || '').replace(/^Bearer\s+|^Key\s+/i, '');

  try {
    // System telemetry endpoint
    if (url.pathname === '/api/system') {
      let tunnelUrl = process.env.TUNNEL_URL || '';
      const tunnelFile = path.join(ROOT, '.tunnel_url');
      if (!tunnelUrl && fs.existsSync(tunnelFile)) {
        try { tunnelUrl = fs.readFileSync(tunnelFile, 'utf8').trim(); } catch {}
      }
      let workerOk = false;
      try {
        const r = await fetch(`${HQ_URL}/health`, { signal: AbortSignal.timeout(1000) });
        const j = await r.json();
        workerOk = !!j.ok;
      } catch {}
      return send(res, 200, {
        status: 'online',
        tunnel_url: tunnelUrl,
        worker_ready: workerOk,
        has_fal_key: !!(FAL_KEY || clientFalKey),
        platform: process.platform,
        node_version: process.version,
      });
    }

    // Video generation proxy
    if (req.method === 'POST' && url.pathname === '/api/video') {
      const b = await readJson(req);
      const falKey = clientFalKey || b.fal_key || FAL_KEY;
      const model = VIDEO_MODELS[b.model] ? b.model : 'h3';
      const mode = b.image_url ? 'i2v' : 't2v';
      const input = { prompt: String(b.prompt || '').slice(0, 4000), ...VIDEO_MODELS[model].input(b, mode === 'i2v') };
      if (b.seed != null && b.seed !== '') input.seed = Number(b.seed);
      if (mode === 'i2v') input.image_url = b.image_url;
      const out = await falRun(VIDEO_MODELS[model][mode], input, falKey);
      console.log(`[video] ${model} ${mode} ${input.duration}s ${input.resolution} in ${out._elapsed.toFixed(1)}s (gpu ${out.timings?.inference?.toFixed?.(2) ?? '?'}s)`);
      return send(res, 200, { model, mode, video_url: out.video?.url, timings: out.timings, elapsed: out._elapsed });
    }

    // Image generation proxy
    if (req.method === 'POST' && url.pathname === '/api/image') {
      const b = await readJson(req);
      const falKey = clientFalKey || b.fal_key || FAL_KEY;
      const out = await falRun(IMAGE_MODEL, {
        prompt: String(b.prompt || '').slice(0, 4000),
        image_size: b.image_size || 'landscape_16_9',
        num_inference_steps: 4,
        enable_safety_checker: true,
      }, falKey);
      console.log(`[image] in ${out._elapsed.toFixed(1)}s`);
      return send(res, 200, { image_url: out.images?.[0]?.url, elapsed: out._elapsed });
    }

    // File upload endpoint for direct video inputs
    if (req.method === 'POST' && url.pathname === '/api/upload') {
      const filename = (url.searchParams.get('name') || `upload_${Date.now()}.mp4`).replace(/[^a-z0-9_.-]+/gi, '_');
      const uploadDir = path.join(TAKES, 'uploads');
      fs.mkdirSync(uploadDir, { recursive: true });
      const dest = path.join(uploadDir, filename);
      const writeStream = fs.createWriteStream(dest);
      for await (const chunk of req) writeStream.write(chunk);
      writeStream.end();
      console.log(`[upload] saved takes/uploads/${filename}`);
      return send(res, 200, { ok: true, video_url: `/takes/uploads/${filename}`, filename });
    }

    // Save an exported take (GLB body, ?name=…&source=<video url>) into ./takes for the Blender renderer.
    if (req.method === 'POST' && url.pathname === '/api/takes') {
      const name = (url.searchParams.get('name') || 'take').replace(/[^a-z0-9_-]+/gi, '_').slice(0, 80);
      const chunks = []; let size = 0;
      for await (const c of req) { size += c.length; if (size > 100e6) throw Object.assign(new Error('take too large'), { status: 413 }); chunks.push(c); }
      fs.mkdirSync(TAKES, { recursive: true });
      fs.writeFileSync(path.join(TAKES, name + '.glb'), Buffer.concat(chunks));
      const meta = { name, source: url.searchParams.get('source') || null, prompt: url.searchParams.get('prompt') || '', saved: new Date().toISOString() };
      fs.writeFileSync(path.join(TAKES, name + '.json'), JSON.stringify(meta, null, 2));
      console.log(`[take] saved takes/${name}.glb`);
      return send(res, 200, { path: `takes/${name}.glb` });
    }

    // HQ pass: forward to the local GVHMR worker (hq/worker.py).
    if (url.pathname.startsWith('/api/hq/job/')) {
      const key = url.pathname.slice('/api/hq/job/'.length);
      try {
        const r = await fetch(`${HQ_URL}/api/hq/job/${key}`);
        const text = await r.text();
        res.writeHead(r.status, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
        return res.end(text);
      } catch {
        return send(res, 200, { status: 'not_found' });
      }
    }
    if (url.pathname === '/api/hq/health') {
      try {
        const r = await fetch(`${HQ_URL}/health`, { signal: AbortSignal.timeout(2000) });
        return send(res, 200, await r.json());
      } catch { return send(res, 200, { ok: false, error: 'worker offline' }); }
    }
    if (req.method === 'POST' && url.pathname === '/api/hq') {
      const b = await readJson(req);
      const r = await fetch(`${HQ_URL}/process`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: b.video_url, ...(b.hands ? { hands: b.hands } : {}) }),
      });
      const text = await r.text();
      res.writeHead(r.status, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      return res.end(text);
    }

    // Serve static files from takes/ directory
    if (req.method === 'GET' && url.pathname.startsWith('/takes/')) {
      const sub = decodeURIComponent(url.pathname.slice('/takes/'.length));
      const file = path.normalize(path.join(TAKES, sub));
      if (!file.startsWith(TAKES) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404); return res.end('take not found');
      }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache',
      });
      return fs.createReadStream(file).pipe(res);
    }

    // Serve static files from public/ directory
    if (req.method === 'GET') {
      let p = decodeURIComponent(url.pathname);
      if (p === '/') p = '/index.html';
      const file = path.normalize(path.join(PUBLIC, p));
      if (!file.startsWith(PUBLIC) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404); return res.end('not found');
      }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache',
      });
      return fs.createReadStream(file).pipe(res);
    }

    res.writeHead(405); res.end();
  } catch (err) {
    console.error('[error]', err.message);
    send(res, err.status && err.status < 600 ? err.status : 500, { error: err.message });
  }
});

server.listen(PORT, () => console.log(`Prompt2Rig → http://localhost:${PORT}`));

