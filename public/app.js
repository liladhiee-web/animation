// Prompt2Rig — Modern Cyber-Studio Frontend
// Prompt → H3 Max Turbo / FLUX 3 Max video → live pose (MediaPipe) → Mixamo rig,
// with GVHMR 3D body + WiLoR finger motion on GPU (Kaggle / Local) that swaps in when ready.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { createStage } from './stage.js';
import { HAND_LIMITS, Rig, OneEuro, Track, trackToClip, blendSnapshots, smplGlobals, aaToQuat } from './retarget.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// Mixamo-rigged characters loaded from three.js CDN
const THREE_MODELS = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r169/examples/models/gltf/';
const CHARACTERS = {
  xbot: { name: 'X Bot', url: THREE_MODELS + 'Xbot.glb' },
  michelle: { name: 'Michelle', url: THREE_MODELS + 'Michelle.glb' },
  soldier: { name: 'Soldier', url: THREE_MODELS + 'Soldier.glb' },
};
const VIDEO_MODEL_LABEL = { h3: 'H3 Max Turbo', flux3: 'FLUX 3 Max' };
const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MP_MODEL = (v) => `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${v}/float16/latest/pose_landmarker_${v}.task`;

const PERFORMER_PROMPT =
  'Full-body photograph of one young adult standing in a relaxed neutral pose facing the camera, arms relaxed at the sides, ' +
  'wearing a fitted plain light-grey t-shirt, black slim pants and white sneakers. The whole body is visible from head to feet ' +
  'with empty space above the head and below the feet, the person is centered. Plain seamless ' +
  'light-grey studio backdrop and floor, flat even soft lighting, sharp focus, 35mm photo.';

const PRESETS = [
  'does an energetic hip hop dance with wide, open arm movements',
  'throws boxing jabs and hooks toward the camera',
  'does a fast martial arts combo: two punches, a front kick, then a roundhouse kick',
  'does a Michael Jackson style dance with sharp kicks and a dramatic final pose',
  'celebrates a victory, jumping and pumping both fists in the air',
  'does a powerful kung fu form with deep horse stances and a crane kick',
  'does star jumps, then a big tuck jump with knees to the chest',
  'does a sword fighting routine with big slashes, thrusts and a lunge',
  'plays a wild air guitar solo, jumping and headbanging',
  'does a superhero power-up: crouches low, then explodes upward with fists raised',
];

const DYNAMIC_PROMPTS = [
  'does an explosive hip hop routine with jumps, kicks and big arm swings',
  'does a fast martial arts combo: two punches, a front kick, then a high roundhouse kick',
  'does a celebratory touchdown dance with a ball spike and a shimmy',
  'does an Irish step dance with fast kicking footwork and arms held straight down',
  'does a karate sequence: front kick, side kick, then a jumping punch',
  'does boxing footwork, bobbing and weaving, then a fast flurry of hooks and uppercuts',
  'does a high-energy aerobics routine with punches, side steps and knee lifts',
  'does a haka-style war dance with stomps, chest slaps and wide powerful stances',
  'does a breakdance toprock with quick crossing footwork, staying on their feet',
  'does a ballet sequence: plié, relevé, then an arabesque with one leg extended back',
  'does a powerful kung fu form with deep horse stances, palm strikes and a crane kick',
  'does a squat kick dance, alternating legs kicking out from a deep squat',
  'does a superhero power-up: crouches low, then explodes upward with fists raised',
  'dances the Charleston with swinging arms and kicking feet',
  'does a Matrix-style slow-motion dodge, leaning far back',
  'does a gymnastics finish: arms up, a jump with a half turn, then a big salute',
];

const IDLE_PROMPT = 'stands in place in a relaxed idle pose, breathing and gently shifting weight from foot to foot';
const MAX_IDLE_STREAK = 3;
const chainOn = () => settings.chain || settings.stream;
const waitOn = () => settings.hq && hqOnline && (settings.waitHq || settings.stream);

const STAGING = 'They stay roughly in place facing the camera, with both arms clearly visible and held away from the torso, no turning around, no flips. ' +
  'Their entire body, including both feet, stays visible in frame the whole time. Static locked-off camera, no camera movement, no zoom, no cuts.';
const videoPrompt = (action, fromImage) => fromImage
  ? `The person ${action}. ${STAGING} Plain studio background.`
  : `Wide full-body shot of one person wearing a plain grey t-shirt and dark pants in an empty light-grey studio. The person ${action}. ${STAGING}`;

// ---- Settings ------------------------------------------------------------------
const DEFAULTS = {
  stream: false, source: 'performer', chain: false, duration: 5, resolution: '480P', hq: true, waitHq: false,
  rootMotion: true, poseModel: 'heavy', smoothing: 0.5, showSkel: true, orientation: 'portrait', performerUrls: {},
  character: 'xbot', videoModel: 'h3', swap: false, cinematic: false, hands: true, falKey: '', lightingTheme: 'cyber',
};
const settings = (() => { try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('p2r') || '{}') }; } catch { return { ...DEFAULTS }; } })();
const save = () => { try { localStorage.setItem('p2r', JSON.stringify(settings)); } catch {} };

function bindSetting(id, kind = 'value', onChange) {
  const el = $('#' + id);
  if (!el) return;
  if (kind === 'checked') el.checked = !!settings[id]; else el.value = settings[id];
  el.addEventListener('input', () => {
    settings[id] = kind === 'checked' ? el.checked : (el.type === 'range' ? Number(el.value) : el.value);
    save(); onChange?.(settings[id]);
  });
  onChange?.(settings[id]);
}

// ---- Log / Banner Helpers ------------------------------------------------------
function log(msg, cls = '') {
  const el = $('#log');
  if (!el) return;
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  const timeStr = new Date().toLocaleTimeString([], { hour12: false });
  entry.innerHTML = `<span class="log-time">${timeStr}</span><span class="log-msg ${cls}">${msg}</span>`;
  el.appendChild(entry);
  el.scrollTop = el.scrollHeight;
}

function banner(text) {
  const b = $('#banner');
  if (!b) return;
  b.textContent = text || '';
  b.classList.toggle('show', !!text);
}

// API request helper with X-Fal-Key header
async function api(path, body) {
  const headers = { 'Content-Type': 'application/json' };
  if (settings.falKey) headers['X-Fal-Key'] = settings.falKey;
  const res = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.detail || json.error || `${path} → ${res.status}`);
  return json;
}

// ---- 3D Stage Setup ------------------------------------------------------------
const stage = createStage($('#stage'));
const { scene } = stage;

// Apply initial lighting theme
stage.setLightingTheme(settings.lightingTheme || 'cyber');

// Keep characters centered when video PIP resizes
new ResizeObserver(() => {
  const v = $('#vidwrap');
  if (v) stage.setInset(v.offsetLeft + v.offsetWidth);
}).observe($('#vidwrap'));

// ---- Character Management ------------------------------------------------------
let model = null, rig = null;

async function loadModel(src, name) {
  const isFbx = /\.fbx$/i.test(name);
  const obj = await (isFbx ? new FBXLoader() : new GLTFLoader()).loadAsync(src);
  const root = isFbx ? obj : obj.scene;
  if (isFbx) root.scale.setScalar(0.01);
  root.updateMatrixWorld(true);

  const armX = (side) => {
    let b; root.traverse((o) => { if (!b && o.isBone && o.name.replace(/^mixamorig\d*[:_]?/i, '') === side + 'Arm') b = o; });
    return b?.getWorldPosition(new THREE.Vector3()).x;
  };
  const lx = armX('Left'), rx = armX('Right');
  if (lx != null && rx != null && lx < rx) { root.rotation.y += Math.PI; root.updateMatrixWorld(true); }
  const box = new THREE.Box3().setFromObject(root);
  root.position.y -= box.min.y;
  root.position.x -= (box.min.x + box.max.x) / 2;
  root.position.z -= (box.min.z + box.max.z) / 2;
  root.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true; o.receiveShadow = true;
      if (o.material) o.material.roughness = Math.max(o.material.roughness, 0.45);
    }
  });
  return { root, rig: new Rig(root) };
}

const cast = {};
let onStage = null;

async function ensureCast(key) {
  if (cast[key]) return cast[key];
  const char = CHARACTERS[key];
  if (!char) throw new Error('unknown character: ' + key);
  const loaded = await loadModel(char.url, char.name);
  loaded.root.visible = false;
  scene.add(loaded.root);
  stage.contactShadow(loaded.root);
  cast[key] = loaded;
  return loaded;
}

function showOnStage(key) {
  if (!key) {
    for (const [k, c] of Object.entries(cast)) c.root.visible = false;
    if (model) { model.visible = true; rig.visible = true; }
    onStage = null; return;
  }
  for (const [k, c] of Object.entries(cast)) c.root.visible = k === key;
  if (model) model.visible = false;
  onStage = key;
}

const CHAR_ORDER = ['xbot', 'michelle', 'soldier'];
async function swapCharacter() {
  const currentKey = onStage || settings.character;
  const nextKey = CHAR_ORDER[(CHAR_ORDER.indexOf(currentKey) + 1) % CHAR_ORDER.length];
  await ensureCast(nextKey);
  const fromRig = onStage ? cast[onStage].rig : rig;
  const toRig = cast[nextKey].rig;
  toRig.copyPoseFrom(fromRig);
  showOnStage(nextKey);
  $('#pill-rig-text').textContent = `Rig: ${CHARACTERS[nextKey].name}`;
}

async function loadCharacter(src, name) {
  const loaded = await loadModel(src, name);
  if (model) { scene.remove(model); stage.removeContactShadow(model); }
  model = loaded.root; rig = loaded.rig;
  scene.add(model);
  stage.contactShadow(model);
  stage.setFraming(1.4);
  showOnStage(null);
  $('#pill-rig-text').textContent = `Rig: ${name.replace(/\.[^.]+$/, '')}`;
  log(`Loaded character ${name}`, 'ok');
}

// ---- Pose Detection Setup (MediaPipe) -------------------------------------------
let vision = null, landmarker = null;
let mpTs = 0;

async function initPose(modelVariant = 'heavy') {
  if (!vision) vision = await FilesetResolver.forVisionTasks(MP_WASM);
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MP_MODEL(modelVariant), delegate: 'GPU' },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  log(`MediaPipe Pose loaded (${modelVariant})`, 'ok');
}

// ---- Takes Management ----------------------------------------------------------
let takeSeq = 0;
class Take {
  constructor(prompt) {
    this.id = ++takeSeq;
    this.prompt = prompt;
    this.status = 'queued';
    this.videoUrl = null;
    this.lastFrame = null;
    this.live = null;
    this.liveDone = false;
    this.hq = null;
    this.hqStatus = 'idle';
    this.hqAt = 0;
    this.timings = {};
    this.handsMethod = settings.hands ? 'wilor' : 'off';
    this.started = false;
  }
}

const takes = [];

function renderTakes() {
  const drawer = $('#takes');
  if (!drawer) return;
  const cards = drawer.querySelectorAll('.take-card');
  cards.forEach((c) => c.remove());

  takes.slice().reverse().forEach((take) => {
    const card = document.createElement('div');
    card.className = 'take-card' + (current === take ? ' playing' : '');
    const isHqReady = take.hqStatus === 'done';
    const isRunningHq = take.hqStatus === 'running';

    const statusTag = isHqReady
      ? `<span class="take-status-tag hq-ready">HQ READY</span>`
      : isRunningHq
      ? `<span class="take-status-tag running">HQ RUNNING...</span>`
      : `<span class="take-status-tag live-only">LIVE PREVIEW</span>`;

    card.innerHTML = `
      <div class="take-header">
        <span class="take-id">#${take.id}</span>
        ${statusTag}
      </div>
      <div class="take-prompt">${take.prompt}</div>
      <div class="take-footer">
        <span class="take-time">${take.timings.gen ? `${take.timings.gen.toFixed(1)}s gen` : ''}</span>
        <div class="take-actions">
          <button class="take-btn" data-action="play">Play</button>
          <button class="take-btn" data-action="glb" title="Export 3D Animation (GLB)">↓ GLB</button>
        </div>
      </div>
    `;

    card.querySelector('[data-action="play"]').onclick = (e) => {
      e.stopPropagation();
      playTake(take);
    };
    card.querySelector('[data-action="glb"]').onclick = (e) => {
      e.stopPropagation();
      exportTakeGlb(take);
    };
    card.onclick = () => playTake(take);

    drawer.appendChild(card);
  });
}

// ---- Performer Setup -----------------------------------------------------------
async function ensurePerformer(force = false) {
  const o = settings.orientation;
  if (!force && settings.performerUrls[o]) return settings.performerUrls[o];
  banner('Generating performer reference frame...');
  log(`Generating performer (${o})...`);
  const out = await api('/api/image', {
    prompt: PERFORMER_PROMPT,
    image_size: o === 'portrait' ? 'portrait_16_9' : 'landscape_16_9',
  });
  if (!out.image_url) throw new Error('image generation returned no url');
  settings.performerUrls[o] = out.image_url;
  save();
  showPerformer();
  banner('');
  log('Performer reference updated', 'ok');
  return out.image_url;
}

function showPerformer() {
  const img = $('#performer-img');
  if (!img) return;
  const url = settings.performerUrls[settings.orientation];
  img.classList.toggle('portrait', settings.orientation === 'portrait');
  if (url) img.src = url; else img.removeAttribute('src');
  if (!video.videoWidth) $('#vidwrap')?.classList.toggle('portrait', settings.orientation === 'portrait');
}

// ---- Video Generation Pipeline -------------------------------------------------
let genBusy = 0;

async function submitPrompt(prompt) {
  if (!prompt.trim()) return;
  const take = new Take(prompt);
  takes.push(take);
  renderTakes();
  const queuePos = genBusy;
  genBusy++;

  try {
    let imgUrl = null;
    if (settings.source === 'performer') {
      const prev = takes[takes.indexOf(take) - 1];
      if (chainOn() && prev?.lastFrame) {
        imgUrl = await prev.lastFrame;
      }
      if (!imgUrl) imgUrl = await ensurePerformer();
    }

    banner(queuePos ? `Waiting in queue (#${take.id})...` : `#${take.id}: Generating video (${VIDEO_MODEL_LABEL[settings.videoModel] || 'H3'})...`);
    log(`#${take.id}: Generating video with ${VIDEO_MODEL_LABEL[settings.videoModel] || 'H3'}...`);

    const out = await api('/api/video', {
      prompt: videoPrompt(prompt, !!imgUrl),
      model: settings.videoModel,
      duration: settings.duration,
      resolution: settings.resolution,
      aspect_ratio: settings.orientation === 'portrait' ? '9:16' : '16:9',
      image_url: imgUrl,
    });

    take.videoUrl = out.video_url;
    take.status = 'ready';
    take.timings.gen = out.elapsed;
    log(`#${take.id} video generated in ${out.elapsed.toFixed(1)}s`, 'ok');

    if (chainOn()) take.lastFrame = lastFrameDataUrl(take.videoUrl).catch(() => null);
    banner('');

    if (settings.hq && hqOnline) runHQ(take);
    renderTakes();

    if (waitOn() && !(settings.stream && held)) {
      take.awaitingHq = true;
      if (!current) banner(`#${take.id}: Waiting for HQ GPU motion pass...`);
      take.hqPromise?.finally(() => {
        banner('');
        if (take.awaitingHq && take.status !== 'skipped') {
          take.awaitingHq = false;
          enqueue(take);
        }
      });
    } else {
      enqueue(take);
    }
  } catch (e) {
    take.status = 'error';
    renderTakes();
    banner('');
    log(`#${take.id} generation failed: ${e.message}`, 'err');
    if (e.message.includes('FAL_KEY')) {
      $('#modal-settings').classList.add('open');
      banner('Please configure your FAL API Key in Settings');
    }
  } finally {
    genBusy = Math.max(0, genBusy - 1);
  }
}

async function lastFrameDataUrl(url) {
  const v = document.createElement('video');
  v.crossOrigin = 'anonymous'; v.muted = true; v.preload = 'auto'; v.src = url;
  await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('cannot load video frame')); });
  v.currentTime = Math.max(0, v.duration - 0.05);
  await new Promise((res) => (v.onseeked = res));
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  v.removeAttribute('src'); v.load();
  return c.toDataURL('image/jpeg', 0.92);
}

// Upload direct local video file
async function uploadLocalVideo(file) {
  if (!file) return;
  banner(`Uploading ${file.name}...`);
  log(`Uploading video file ${file.name}...`);
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || 'Upload failed');

    const take = new Take(`(file) ${file.name}`);
    take.videoUrl = json.video_url;
    take.status = 'ready';
    take.timings.gen = 0;
    takes.push(take);
    renderTakes();
    banner('');
    log(`Video uploaded successfully (${(file.size / (1024*1024)).toFixed(1)} MB)`, 'ok');

    if (settings.hq && hqOnline) runHQ(take);
    enqueue(take);
  } catch (e) {
    banner('');
    log(`Video upload failed: ${e.message}`, 'err');
  }
}

// ---- HQ Motion (GVHMR + WiLoR) ------------------------------------------------
let hqOnline = false;

async function checkHQ() {
  try {
    const r = await api('/api/hq/health');
    hqOnline = !!r.ok;
    const dev = (r.device || 'GPU Ready').replace(/^NVIDIA GeForce /, '').toLowerCase();
    $('#pill-hq-text').textContent = r.ok ? `GVHMR: ${dev}` : `GVHMR: ${r.error ? 'Loading...' : 'Offline'}`;
    $('#pill-hq').className = `status-pill ${r.ok ? 'active' : 'hq'}`;
  } catch {
    hqOnline = false;
    $('#pill-hq-text').textContent = 'GVHMR: Offline';
    $('#pill-hq').className = 'status-pill';
  }
}

function runHQ(take) {
  const t0 = performance.now();
  take.hqStatus = 'running';
  renderTakes();

  take.hqPromise = api('/api/hq', {
    video_url: take.videoUrl,
    poll: true,
    ...(take.handsMethod ? { hands: take.handsMethod } : {}),
  })
    .then((resp) => {
      // If immediate finished result returned (e.g. cached)
      if (resp.n && resp.fps) {
        handleHQResult(take, resp, t0);
        return;
      }
      // If async job returned for polling (prevents Cloudflare 100s timeout on long clips!)
      if (resp.status === 'processing' && resp.key) {
        return pollHQJob(take, resp.key, t0);
      }
      throw new Error(resp.message || 'Unexpected response from HQ worker');
    })
    .catch((e) => {
      take.hqStatus = 'error';
      banner('');
      log(`#${take.id} HQ motion failed: ${e.message}`, 'err');
      renderTakes();
    });
}

function pollHQJob(take, key, t0) {
  return new Promise((resolve, reject) => {
    banner(`#${take.id}: Processing HQ motion on GPU...`);
    const interval = setInterval(async () => {
      try {
        const job = await api(`/api/hq/job/${key}`);
        if (job.status === 'processing') {
          const sec = job.elapsed ? `${job.elapsed}s` : `${((performance.now() - t0) / 1000).toFixed(0)}s`;
          banner(`#${take.id}: Processing HQ motion on GPU (${sec})...`);
          take.hqStatus = 'running';
          renderTakes();
        } else if (job.status === 'done' && job.result) {
          clearInterval(interval);
          handleHQResult(take, job.result, t0);
          resolve(job.result);
        } else if (job.status === 'error') {
          clearInterval(interval);
          take.hqStatus = 'error';
          banner('');
          log(`#${take.id} HQ motion failed: ${job.error}`, 'err');
          renderTakes();
          reject(new Error(job.error));
        }
      } catch (err) {
        // Network jitter or temporary connection reset: keep polling!
      }
    }, 2000);
  });
}

function handleHQResult(take, r, t0) {
  take.hqRaw = r;
  take.hq = bakeSmpl(r);
  take.hqAt = performance.now();
  take.hqStatus = 'done';
  take.timings.hq = (performance.now() - t0) / 1000;
  banner('');
  const handsTag = r.hands?.method ? ` · hands: ${r.hands.method}` : '';
  log(`#${take.id} HQ motion ready in ${take.timings.hq.toFixed(1)}s (${r.n} frames on ${r.device || 'GPU'}${handsTag})`, 'hq');
  renderTakes();
  if (take.awaitingHq) {
    take.awaitingHq = false;
    enqueue(take);
  }
}

const _kp = Array.from({ length: 21 }, () => new THREE.Vector3());
function applyHandsFrame(hands, f, yawFix) {
  HAND_LIMITS.enabled = hands?.method !== 'wilor';
  for (const [src, side] of [['R', 'Right'], ['L', 'Left']]) {
    const h = hands?.[src];
    if (settings.hands !== false && h?.valid?.[f]) {
      const o = f * 63;
      for (let j = 0; j < 21; j++) _kp[j].set(h.kp[o + j * 3], h.kp[o + j * 3 + 1], h.kp[o + j * 3 + 2]).applyQuaternion(yawFix);
      rig.applyHand(side, _kp);
    } else rig.relaxHand(side);
  }
}

function bakeSmpl(r) {
  const { n, fps, global_orient: GO, body_pose: BP, transl: T, leg_length } = r;
  const fWorld = new THREE.Vector3(0, 0, 1).applyQuaternion(aaToQuat(GO[0], GO[1], GO[2]));
  let targetYaw = 0;
  if (r.incam_global_orient0) {
    const [a, b, c] = r.incam_global_orient0;
    const fc = new THREE.Vector3(0, 0, 1).applyQuaternion(aaToQuat(a, b, c));
    targetYaw = Math.atan2(fc.x, -fc.z);
  }
  const yawFix = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), targetYaw - Math.atan2(fWorld.x, fWorld.z));
  const scale = rig ? rig.legLength / (leg_length || 0.85) : 1.0;
  const track = new Track(rig.order.length);
  const lows = [];
  const d = new THREE.Vector3();

  for (let f = 0; f < n; f++) {
    rig.reset();
    rig.applySmplGlobal(smplGlobals(GO.subarray ? GO.subarray(f * 3, f * 3 + 3) : GO.slice(f * 3, f * 3 + 3), BP.slice(f * 63, f * 63 + 63), yawFix));
    d.set(T[f * 3] - T[0], T[f * 3 + 1] - T[1], T[f * 3 + 2] - T[2]).applyQuaternion(yawFix).multiplyScalar(scale);
    const p = rig.rest.Hips.wp.clone().add(d);
    rig.setHipsWorld(p);
    applyHandsFrame(r.hands, f, yawFix);
    lows.push(rig.lowestFoot());
    track.push(f / fps, rig.snapshot());
  }

  const sorted = [...lows].sort((a, b) => a - b);
  const offset = rig.floorY - sorted[Math.floor(sorted.length * 0.08)];
  track.p.forEach((p, f) => { p[1] += offset + Math.max(0, rig.floorY - (lows[f] + offset)); });
  rig.reset();
  return track;
}

// ---- Playback & Sequencing -----------------------------------------------------
const video = $('#video');
const overlay = $('#overlay');
const octx = overlay.getContext('2d');
let current = null;
const queue = [];
let liveSnap = null;
let liveState = null;
const worldFilter = new OneEuro(99);
const hipFilter = new OneEuro(2);
let lastRenderedHips = new THREE.Vector3();
let switchFrom = null, switchAt = 0;

function setSmoothing(s) {
  worldFilter.minCutoff = hipFilter.minCutoff = THREE.MathUtils.lerp(6, 0.4, s);
  worldFilter.beta = hipFilter.beta = THREE.MathUtils.lerp(1.2, 0.15, s);
  const badge = $('#smoothing-v');
  if (badge) badge.textContent = Math.round(s * 100) + '%';
}

let held = false;

function enqueue(take) {
  queue.push(take);
  if (capturing) return;
  if (!current || held || (current.liveDone && !chainOn())) playNext();
}

function playNext() {
  const t = queue.shift();
  if (t) playTake(t);
}

function playTake(take) {
  if (!take.videoUrl) return;
  if (current && rig) { switchFrom = rig.snapshot(); switchAt = performance.now(); }
  held = false; take.started = true;
  if (take.hq) take.hqAt = -1e9;
  const prev = current;
  current = take;
  const qi = queue.indexOf(take); if (qi >= 0) queue.splice(qi, 1);

  if (!take.rootOffset) {
    take.rootOffset = new THREE.Vector3();
    if (chainOn() && prev && rig) {
      take.rootOffset.set(lastRenderedHips.x - rig.rest.Hips.wp.x, 0, lastRenderedHips.z - rig.rest.Hips.wp.z);
      take.rootOffset.clampLength(0, 2.5);
    }
  }

  if (!take.liveDone) {
    take.live = new Track(rig.order.length);
    liveState = { hip0: null, baseY: null, mpp: null, scale: null };
    worldFilter.reset(); hipFilter.reset(); liveSnap = null;
  }

  $('#vid-empty').style.display = 'none';
  video.style.display = 'block';
  video.src = take.videoUrl;
  video.onloadedmetadata = () => $('#vidwrap').classList.toggle('portrait', video.videoHeight > video.videoWidth);
  lastDetectT = -1;
  video.play().catch(() => {});
  renderTakes();
}

video.addEventListener('ended', () => {
  if (settings.swap && rig) swapCharacter().catch((e) => log('swap failed: ' + e.message, 'err'));
  if (current && !current.liveDone) {
    current.liveDone = current.live.length > 3;
    log(`#${current.id} live pass complete (${current.live.length} frames)`);
    renderTakes();
  }
  if (queue.length) playNext();
  else if (settings.stream) { held = true; }
  else { video.currentTime = 0; video.play().catch(() => {}); }
});

// Per decoded frame: MediaPipe pose detection + overlay
let lastDetectT = -1;
let lastPose2d = null;

function detectIfNewFrame() {
  if (!landmarker || !rig || !current || video.readyState < 2 || video.seeking) return;
  const mediaTime = video.currentTime;
  if (mediaTime === lastDetectT) return;
  if (mediaTime < lastDetectT) { worldFilter.reset(); hipFilter.reset(); }
  lastDetectT = mediaTime;

  try {
    const ts = Math.max(mpTs + 1, performance.now());
    mpTs = ts;
    const res = landmarker.detectForVideo(video, ts);
    drawOverlay(res);
    if (!current.liveDone && res.worldLandmarks?.[0]) {
      liveSnap = solveLive(res, mediaTime);
      current.live.push(mediaTime, liveSnap);
    }
  } catch (e) {
    console.error(e);
  }
}

function drawOverlay(res) {
  if (res.landmarks?.[0]) lastPose2d = res.landmarks[0];
  overlay.width = video.videoWidth || 320;
  overlay.height = video.videoHeight || 240;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!settings.showSkel || !res.landmarks?.[0]) return;
  const du = new DrawingUtils(octx);
  du.drawConnectors(res.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: '#06b6d4', lineWidth: 3 });
  du.drawLandmarks(res.landmarks[0], { color: '#ec4899', fillColor: '#090710', lineWidth: 2, radius: 3 });
}

const _lm = Array.from({ length: 33 }, () => new THREE.Vector3());
function solveLive(res, t) {
  const w = res.worldLandmarks[0], n2 = res.landmarks[0];
  const raw = new Float32Array(99);
  w.forEach((p, i) => { raw[i * 3] = p.x; raw[i * 3 + 1] = -p.y; raw[i * 3 + 2] = -p.z; });
  const f = worldFilter.filter(raw, t);
  for (let i = 0; i < 33; i++) _lm[i].set(f[i * 3], f[i * 3 + 1], f[i * 3 + 2]);
  const vis = w.map((p) => p.visibility ?? 1);

  const W = video.videoWidth, H = video.videoHeight;
  const px = (i) => new THREE.Vector2(n2[i].x * W, n2[i].y * H);
  const hipPx = px(23).add(px(24)).multiplyScalar(0.5);
  const shPx = px(11).add(px(12)).multiplyScalar(0.5);
  const hf = hipFilter.filter([hipPx.x, hipPx.y], t);
  const midHip = _lm[23].clone().add(_lm[24]).multiplyScalar(0.5);
  const midSh = _lm[11].clone().add(_lm[12]).multiplyScalar(0.5);
  const mpp = midSh.distanceTo(midHip) / Math.max(1, shPx.distanceTo(hipPx));
  const leg = (_lm[23].distanceTo(_lm[25]) + _lm[25].distanceTo(_lm[27]) + _lm[24].distanceTo(_lm[26]) + _lm[26].distanceTo(_lm[28])) / 2;

  const S = liveState;
  S.mpp = S.mpp == null ? mpp : S.mpp + 0.05 * (mpp - S.mpp);
  S.scale = S.scale == null ? rig.legLength / leg : S.scale + 0.05 * (rig.legLength / leg - S.scale);
  if (!S.hip0) S.hip0 = [hf[0], hf[1]];

  rig.reset();
  rig.applyLandmarks(_lm, vis);
  const p = rig.rest.Hips.wp.clone();
  if (settings.rootMotion) {
    p.x += (hf[0] - S.hip0[0]) * S.mpp * S.scale;
    p.y += -(hf[1] - S.hip0[1]) * S.mpp * S.scale;
  }
  rig.setHipsWorld(p);
  if (S.baseY == null) S.baseY = rig.floorY - rig.lowestFoot();
  p.y += S.baseY; rig.setHipsWorld(p);
  rig.clampToFloor();
  return rig.snapshot();
}

// ---- Render Loop & HUD Telemetry -----------------------------------------------
const bufA = { q: null, p: new Float32Array(3) };
const bufB = { q: null, p: new Float32Array(3) };
const bufC = { q: null, p: new Float32Array(3) };

let lastRaf = 0;
let capturing = false;

function frame() {
  requestAnimationFrame(frame);
  lastRaf = performance.now();
  if (!capturing) tick();
}

setInterval(() => {
  if (!capturing && performance.now() - lastRaf > 100) tick();
}, 33);

function formatTime(s) {
  if (isNaN(s) || !isFinite(s)) return '00:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 100);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
}

function tick() {
  detectIfNewFrame();
  if (rig && current) {
    const n = rig.order.length * 4;
    for (const b of [bufA, bufB, bufC]) if (!b.q || b.q.length !== n) b.q = new Float32Array(n);
    const t = video.currentTime;
    const live = current.liveDone ? current.live.sample(t, bufA) : liveSnap;
    const hq = current.hq ? current.hq.sample(t, bufB) : null;
    let snap = hq || live;
    let src = hq ? 'HQ · GVHMR (GPU)' : live ? 'LIVE · MediaPipe' : '—';

    if (hq && live) {
      const w = Math.min(1, (performance.now() - current.hqAt) / 600);
      if (w < 1) {
        snap = blendSnapshots(live.q, live.p, hq.q, hq.p, w, bufC);
        src = 'LIVE → HQ BLEND';
      }
    }

    if (snap && switchFrom) {
      const w = Math.min(1, (performance.now() - switchAt) / 250);
      if (w < 1) snap = blendSnapshots(switchFrom.q, switchFrom.p, snap.q, snap.p, w, snap === bufC ? bufA : bufC);
      else switchFrom = null;
    }

    if (snap) {
      rig.applySnapshot(snap);
      if (current.rootOffset?.lengthSq()) {
        const h = rig.hipsWorld().add(current.rootOffset);
        rig.setHipsWorld(h);
      }
      rig.hipsWorld(lastRenderedHips);
    }

    // Update Floating HUD Playback Dock
    const hudSrc = $('#hud-src');
    if (hudSrc) {
      hudSrc.textContent = src;
      hudSrc.className = 'dock-motion-badge ' + (hq ? 'hq' : 'live');
    }

    const scrubber = $('#scrubber');
    if (scrubber && !scrubberDragging && video.duration) {
      scrubber.value = ((video.currentTime / video.duration) * 100).toFixed(1);
    }

    const dockTime = $('#dock-time');
    if (dockTime) {
      dockTime.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
    }
  }

  // Update Three.js rendering
  const activeRig = onStage ? cast[onStage].rig : rig;
  if (activeRig) stage.update(activeRig.hipsWorld());
  else stage.update(null);
}

// ---- GLB Export ----------------------------------------------------------------
async function exportTakeGlb(take) {
  if (!rig) return;
  banner(`Exporting #${take.id} to GLB animation...`);
  log(`Exporting #${take.id} to GLB animation...`);
  const track = take.hq || take.live;
  if (!track || !track.length) {
    banner('');
    log(`Cannot export #${take.id}: no recorded motion track`, 'err');
    return;
  }

  const clip = trackToClip(rig, track, take.prompt);
  const root = onStage ? cast[onStage].root : model;
  const prevPose = rig.snapshot();
  rig.reset();

  const exporter = new GLTFExporter();
  const glb = await new Promise((res, rej) => {
    exporter.parse(root, res, rej, { binary: true, animations: [clip] });
  });

  rig.applySnapshot(prevPose);
  banner('');

  const blob = new Blob([glb], { type: 'model/gltf-binary' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  const safePrompt = take.prompt.replace(/[^a-z0-9_-]+/gi, '_').slice(0, 32);
  const name = `take_${take.id}_${safePrompt}`;
  a.download = `${name}.glb`;
  a.click();

  // Save copy to ./takes
  const q = new URLSearchParams({ name, prompt: take.prompt, ...(take.videoUrl?.startsWith('http') ? { source: take.videoUrl } : {}) });
  fetch('/api/takes?' + q, { method: 'POST', body: glb }).catch(() => {});
  log(`Exported ${name}.glb`, 'ok');
}

// ---- UI Event Wiring & Interactions --------------------------------------------
let scrubberDragging = false;

// Play / Pause Dock Button
const dockPlayBtn = $('#dock-play');
if (dockPlayBtn) {
  dockPlayBtn.onclick = () => {
    if (video.paused) video.play(); else video.pause();
  };
  video.addEventListener('play', () => {
    dockPlayBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
  });
  video.addEventListener('pause', () => {
    dockPlayBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
  });
}

// Timeline Scrubber
const scrubberEl = $('#scrubber');
if (scrubberEl) {
  scrubberEl.addEventListener('mousedown', () => { scrubberDragging = true; });
  scrubberEl.addEventListener('touchstart', () => { scrubberDragging = true; });
  scrubberEl.addEventListener('input', () => {
    if (video.duration) {
      video.currentTime = (Number(scrubberEl.value) / 100) * video.duration;
    }
  });
  const stopDrag = () => { scrubberDragging = false; };
  window.addEventListener('mouseup', stopDrag);
  window.addEventListener('touchend', stopDrag);
}

// Speed Chips
$$('.dock-chip[data-speed]').forEach((chip) => {
  chip.onclick = () => {
    $$('.dock-chip[data-speed]').forEach((c) => c.classList.remove('active'));
    chip.classList.add('active');
    video.playbackRate = Number(chip.dataset.speed);
  };
});

// Camera Presets
$$('.cam-btn[data-cam]').forEach((btn) => {
  btn.onclick = () => stage.setCameraPreset(btn.dataset.cam);
});

// Lighting Themes
$$('#lighting-themes .preset-chip[data-theme]').forEach((chip) => {
  chip.onclick = () => {
    $$('#lighting-themes .preset-chip').forEach((c) => c.style.borderColor = '');
    chip.style.borderColor = 'var(--neon-pink)';
    settings.lightingTheme = chip.dataset.theme;
    save();
    stage.setLightingTheme(chip.dataset.theme);
  };
});

// Character Selector Cards
$$('.char-card[data-char]').forEach((card) => {
  card.onclick = () => {
    $$('.char-card').forEach((c) => c.classList.remove('active'));
    card.classList.add('active');
    const charKey = card.dataset.char;
    settings.character = charKey;
    save();
    if (CHARACTERS[charKey]) {
      loadCharacter(CHARACTERS[charKey].url, CHARACTERS[charKey].name).catch((e) => log('character error: ' + e.message, 'err'));
    }
  };
});

// Tab Bar Navigation
$$('.tab-btn').forEach((btn) => {
  btn.onclick = () => {
    $$('.tab-btn').forEach((b) => b.classList.remove('active'));
    $$('.tab-pane').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    const targetPane = $('#' + btn.dataset.tab);
    if (targetPane) targetPane.classList.add('active');
  };
});

// Surprise Me Prompt Generator
$('#btn-surprise')?.addEventListener('click', () => {
  const allPrompts = [...PRESETS, ...DYNAMIC_PROMPTS];
  const rand = allPrompts[Math.floor(Math.random() * allPrompts.length)];
  $('#prompt').value = rand;
  $('#prompt').focus();
});

$('#btn-clear-prompt')?.addEventListener('click', () => {
  $('#prompt').value = '';
  $('#prompt').focus();
});

$('#btn-clear-log')?.addEventListener('click', () => {
  const el = $('#log');
  if (el) el.innerHTML = '';
});

// Populate Preset Chips
const chipsContainer = $('#chips');
if (chipsContainer) {
  PRESETS.forEach((p) => {
    const chip = document.createElement('div');
    chip.className = 'preset-chip';
    chip.textContent = p;
    chip.onclick = () => {
      $('#prompt').value = p;
      submitPrompt(p);
    };
    chipsContainer.appendChild(chip);
  });
}

// Generate Button & Keyboard Shortcuts
$('#go')?.addEventListener('click', () => submitPrompt($('#prompt').value));
$('#prompt')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    submitPrompt($('#prompt').value);
  }
});
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target.tagName !== 'TEXTAREA' && e.target.tagName !== 'INPUT') {
    e.preventDefault();
    if (video.paused) video.play(); else video.pause();
  }
});

// Drag & Drop Video Upload
const dropZone = $('#drop-zone');
if (dropZone) {
  dropZone.onclick = () => $('#file-vid').click();
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files?.[0]) uploadLocalVideo(e.dataTransfer.files[0]);
  });
}
$('#file-vid')?.addEventListener('change', (e) => {
  if (e.target.files?.[0]) uploadLocalVideo(e.target.files[0]);
});

// Custom Character Mixamo File Upload
$('#btn-char')?.addEventListener('click', () => $('#file-char').click());
$('#file-char')?.addEventListener('change', (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  loadCharacter(url, f.name).catch((err) => {
    log('Character load failed: ' + err.message, 'err');
    $('#pill-rig-text').textContent = 'Rig: Failed';
  });
});

// Performer Generation
$('#performer-new')?.addEventListener('click', () => {
  ensurePerformer(true).catch((e) => {
    banner('');
    log('Performer generation failed: ' + e.message, 'err');
  });
});

// Settings & Cloudflare Modal
const settingsModal = $('#modal-settings');
$('#btn-settings')?.addEventListener('click', () => {
  $('#cfg-fal-key').value = settings.falKey || '';
  settingsModal.classList.add('open');
  fetchSystemTelemetry();
});

const closeModal = () => settingsModal.classList.remove('open');
$('#modal-close')?.addEventListener('click', closeModal);
$('#btn-close-modal')?.addEventListener('click', closeModal);
settingsModal?.addEventListener('click', (e) => {
  if (e.target === settingsModal) closeModal();
});

$('#btn-save-key')?.addEventListener('click', () => {
  settings.falKey = $('#cfg-fal-key').value.trim();
  save();
  log('FAL API Key updated', 'ok');
  banner('FAL API Key saved successfully');
  setTimeout(() => banner(''), 3000);
});

$('#btn-copy-cf')?.addEventListener('click', () => {
  const url = $('#cfg-cf-url').value;
  if (url && url !== 'Detecting...') {
    navigator.clipboard.writeText(url);
    banner('Cloudflare Tunnel URL copied to clipboard!');
    setTimeout(() => banner(''), 3000);
  }
});

// Video PIP Minimizing & Skeleton Toggle
let pipMinimized = false;
$('#pip-min')?.addEventListener('click', () => {
  pipMinimized = !pipMinimized;
  const body = $('.pip-body');
  if (body) body.style.display = pipMinimized ? 'none' : 'grid';
  $('#pip-min').textContent = pipMinimized ? '□' : '—';
});

$('#pip-toggle-skel')?.addEventListener('click', () => {
  settings.showSkel = !settings.showSkel;
  $('#showSkel').checked = settings.showSkel;
  save();
  $('#pip-toggle-skel').style.opacity = settings.showSkel ? '1' : '0.4';
});

// Fullscreen Toggle
$('#btn-fullscreen')?.addEventListener('click', () => {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen();
  else document.exitFullscreen();
});

// Form Bindings
bindSetting('stream', 'checked', (on) => {
  for (const id of ['chain', 'waitHq']) {
    const el = $('#' + id);
    if (el) { el.disabled = on; el.closest('.switch-control').style.opacity = on ? '0.5' : '1'; }
  }
});
bindSetting('source');
bindSetting('chain', 'checked');
bindSetting('resolution');
bindSetting('videoModel', 'value', (m) => {
  const d = $('#duration');
  if (d) {
    d.min = m === 'flux3' ? 5 : 2;
    if (settings.duration < d.min) {
      settings.duration = Number(d.min);
      d.value = d.min;
      $('#duration-v').textContent = d.min + 's';
      save();
    }
  }
});
bindSetting('swap', 'checked', (on) => { if (!on && model) showOnStage(null); });
bindSetting('cinematic', 'checked', (on) => stage.setCinematic(on));
bindSetting('duration', 'value', (v) => { $('#duration-v').textContent = v + 's'; });
bindSetting('hq', 'checked');
bindSetting('hands', 'checked', () => { if (rig) for (const t of takes) if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); });
bindSetting('waitHq', 'checked');
bindSetting('rootMotion', 'checked');
bindSetting('showSkel', 'checked');
bindSetting('smoothing', 'value', setSmoothing);
bindSetting('poseModel', 'value', (v) => { if (vision) initPose(v); });
bindSetting('orientation', 'value', showPerformer);

// System Telemetry Polling (GPU & Cloudflare Tunnel)
async function fetchSystemTelemetry() {
  try {
    const info = await api('/api/system');
    if (info.gpu) {
      $('#pill-gpu-text').textContent = info.cuda ? `GPU: ${info.gpu}` : 'CPU Mode';
    }
    if (info.tunnel_url) {
      $('#pill-cf-text').textContent = 'Cloudflare: Live';
      $('#pill-cf').className = 'status-pill tunnel active';
      $('#cfg-cf-url').value = info.tunnel_url;
    } else {
      $('#pill-cf-text').textContent = 'Localhost';
      $('#pill-cf').className = 'status-pill tunnel';
      $('#cfg-cf-url').value = window.location.origin;
    }
    const sysEl = $('#cfg-sys-info');
    if (sysEl) {
      sysEl.innerHTML = `
        <strong>GPU:</strong> ${info.gpu} (${info.vram_used_gb || 0} / ${info.vram_total_gb || 0} GB VRAM)<br>
        <strong>GVHMR Engine:</strong> ${info.worker_ready ? '<span style="color:#10b981;">Ready on GPU</span>' : `<span style="color:#f59e0b;">${info.worker_error || 'Loading...'}</span>`}<br>
        <strong>Platform:</strong> ${info.platform} · Python ${info.python_version || '3.x'}<br>
        <strong>FAL API Key:</strong> ${info.has_fal_key || settings.falKey ? '<span style="color:#10b981;">Configured</span>' : '<span style="color:#f43f5e;">Missing (add above)</span>'}
      `;
    }
  } catch {}
}

// Debug API handle for console & tests
window.p2r = {
  get current() { return current; },
  get rig() { return rig; },
  takes, video, stage, HAND_LIMITS,
  submit: submitPrompt,
  upload: uploadLocalVideo,
};

// ---- Boot Application ----------------------------------------------------------
frame();
const startChar = CHARACTERS[settings.character] || CHARACTERS.xbot;
loadCharacter(startChar.url, startChar.name).catch((e) => {
  log('Character load failed: ' + e.message, 'err');
  $('#pill-rig-text').textContent = 'Rig: Failed';
});
initPose(settings.poseModel).catch((e) => log('Pose detector init: ' + e.message, 'err'));
checkHQ();
fetchSystemTelemetry();
setInterval(checkHQ, 8000);
setInterval(fetchSystemTelemetry, 10000);

log('Prompt2Rig Studio initialized and ready.');
