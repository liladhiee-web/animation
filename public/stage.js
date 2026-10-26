// The 3D stage: studio lighting, reflective floor, contact shadows, cinematic camera, view presets.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Reflector } from 'three/addons/objects/Reflector.js';

const FLOOR_COLOR = 0x120f1b;
const FOG_COLOR = 0x090710;

function gradientTexture(top, bottom) {
  const c = document.createElement('canvas'); c.width = 4; c.height = 512;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, top); grad.addColorStop(0.62, bottom); grad.addColorStop(1, bottom);
  g.fillStyle = grad; g.fillRect(0, 0, 4, 512);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function blobTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(0,0,0,0.85)'); grad.addColorStop(0.45, 'rgba(0,0,0,0.45)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

export function createStage(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = gradientTexture('#1c172e', '#090710');
  scene.fog = new THREE.Fog(FOG_COLOR, 9, 28);

  // Studio ambient reflections
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.55;

  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 100);
  camera.position.set(0.4, 1.45, 5.4);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.95, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.495; // stay above the floor

  // Lights: Key light, Rim light, Fill light, Hemi light
  const hemi = new THREE.HemisphereLight(0xddd6fe, 0x0f0b18, 0.6);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xfdfbf7, 2.8);
  key.position.set(-2.8, 5.5, 4.2); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02;
  Object.assign(key.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 0.5, far: 20 });
  scene.add(key);

  const rim = new THREE.DirectionalLight(0xc084fc, 1.35); // Purple neon rim
  rim.position.set(3.5, 4, -4.5);
  scene.add(rim);

  const fill = new THREE.DirectionalLight(0x38bdf8, 0.45); // Subtle cyan fill
  fill.position.set(3, 1.5, 4);
  scene.add(fill);

  // Floor: mirror with matte overlay
  const mirror = new Reflector(new THREE.CircleGeometry(40, 64), {
    color: 0x5a5468,
    textureWidth: Math.round(container.clientWidth * renderer.getPixelRatio() * 0.5) || 1024,
    textureHeight: Math.round(container.clientHeight * renderer.getPixelRatio() * 0.5) || 1024,
  });
  mirror.rotation.x = -Math.PI / 2; mirror.position.y = -0.001;
  scene.add(mirror);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(40, 64),
    new THREE.MeshStandardMaterial({ color: FLOOR_COLOR, roughness: 0.88, metalness: 0.05, transparent: true, opacity: 0.92 }),
  );
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
  scene.add(floor);

  const grid = new THREE.GridHelper(40, 80, 0x4c3875, 0x1f1730);
  grid.material.transparent = true; grid.material.opacity = 0.4; grid.position.y = 0.002;
  scene.add(grid);

  // Contact shadows
  const blobTex = blobTexture();
  const blobs = new Map();
  function contactShadow(model) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false, opacity: 0.85 }));
    m.rotation.x = -Math.PI / 2; m.position.y = 0.004; m.renderOrder = 1;
    scene.add(m); blobs.set(model, m);
    return m;
  }
  function removeContactShadow(model) { const m = blobs.get(model); if (m) { scene.remove(m); blobs.delete(model); } }
  function placeContactShadow(model, hips, floorY, size) {
    const m = blobs.get(model); if (!m) return;
    const lift = Math.max(0, hips.y - floorY - size * 0.52);
    const k = Math.max(0.25, 1 - lift * 1.6);
    m.position.set(hips.x, floorY + 0.004, hips.z);
    m.scale.setScalar(size * 0.55 * (0.8 + 0.2 * k));
    m.material.opacity = 0.85 * k;
  }

  // Camera state & cinematic follow
  const state = { cinematic: false, width: 1.4, inset: 0, pausedUntil: 0, t0: performance.now(), focus: new THREE.Vector3(0, 0.95, 0) };
  controls.addEventListener('start', () => { state.pausedUntil = performance.now() + 6000; });

  const baseDistance = () => {
    const w = container.clientWidth - state.inset, h = container.clientHeight || 1;
    const vHalf = THREE.MathUtils.degToRad(camera.fov / 2);
    const hHalf = Math.atan(Math.tan(vHalf) * Math.max(0.3, w / h));
    return Math.max(4.6, 1.2 / Math.tan(vHalf), (state.width / 2 + 0.5) / Math.tan(hHalf));
  };

  function applyInset() {
    const w = container.clientWidth || 1;
    const f = state.inset / 2 / w;
    camera.filmOffset = -f * camera.getFilmWidth() * camera.aspect * 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    camera.updateProjectionMatrix();
  }

  function reframe() {
    if (state.cinematic) return;
    const dir = camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(controls.target).addScaledVector(dir, baseDistance());
  }

  function fitFog() { const d = baseDistance(); scene.fog.near = d + 3; scene.fog.far = d + 22; }

  function update(focus) {
    if (state.lock) { controls.target.copy(state.lock); controls.update(); renderer.render(scene, camera); return; }
    if (focus) state.focus.lerp(new THREE.Vector3(focus.x, 0.95, focus.z), 0.04);
    const now = performance.now();
    if (state.cinematic && now > state.pausedUntil) {
      const t = (now - state.t0) / 1000;
      const az = 0.5 * Math.sin((t / 26) * Math.PI * 2);
      const dist = baseDistance() * (0.86 + 0.14 * Math.cos((t / 19) * Math.PI * 2));
      const h = 1.15 + 0.3 * Math.sin((t / 31) * Math.PI * 2);
      const want = new THREE.Vector3(state.focus.x + Math.sin(az) * dist, h, state.focus.z + Math.cos(az) * dist);
      camera.position.lerp(want, 0.03);
      controls.target.lerp(state.focus, 0.05);
    } else {
      const delta = state.focus.clone().sub(controls.target).multiplyScalar(0.04);
      controls.target.add(delta); camera.position.add(delta);
    }
    controls.update();
    renderer.render(scene, camera);
  }

  function resize() {
    const { clientWidth: w, clientHeight: h } = container;
    renderer.setSize(w, h); camera.aspect = w / h; applyInset();
    mirror.getRenderTarget().setSize(Math.round(w * renderer.getPixelRatio() * 0.5), Math.round(h * renderer.getPixelRatio() * 0.5));
  }
  new ResizeObserver(resize).observe(container);
  resize();

  // Camera presets
  function setCameraPreset(preset) {
    state.cinematic = false;
    const dist = baseDistance();
    const tgt = new THREE.Vector3(0, 0.95, 0);
    controls.target.copy(tgt);

    switch (preset) {
      case 'front':
        camera.position.set(0, 1.25, dist);
        break;
      case 'side':
        camera.position.set(dist, 1.25, 0);
        break;
      case 'threeQuarter':
        camera.position.set(dist * 0.72, 1.4, dist * 0.72);
        break;
      case 'top':
        camera.position.set(0, dist * 1.2, 0.01);
        break;
      case 'reset':
      default:
        camera.position.set(0.4, 1.45, dist);
        break;
    }
    controls.update();
  }

  // Lighting themes
  function setLightingTheme(theme) {
    switch (theme) {
      case 'cyber':
        key.color.setHex(0xf8fafc); key.intensity = 2.6;
        rim.color.setHex(0xf43f5e); rim.intensity = 2.0; // Cyber magenta
        fill.color.setHex(0x06b6d4); fill.intensity = 1.1; // Cyber cyan
        scene.background = gradientTexture('#181028', '#07050d');
        break;
      case 'studio':
        key.color.setHex(0xffffff); key.intensity = 2.9;
        rim.color.setHex(0xe2e8f0); rim.intensity = 0.8;
        fill.color.setHex(0xffffff); fill.intensity = 0.6;
        scene.background = gradientTexture('#201c29', '#0d0b13');
        break;
      case 'sunset':
        key.color.setHex(0xfbbf24); key.intensity = 2.8; // Amber
        rim.color.setHex(0x8b5cf6); rim.intensity = 1.6; // Purple
        fill.color.setHex(0xf97316); fill.intensity = 0.7; // Orange
        scene.background = gradientTexture('#251522', '#0c070f');
        break;
      case 'noir':
        key.color.setHex(0xffffff); key.intensity = 3.4;
        rim.color.setHex(0x94a3b8); rim.intensity = 0.4;
        fill.color.setHex(0x000000); fill.intensity = 0.0;
        scene.background = gradientTexture('#111116', '#050507');
        break;
    }
  }

  return {
    renderer, scene, camera, controls, update, contactShadow, removeContactShadow, placeContactShadow,
    setCinematic(on) { state.cinematic = on; state.t0 = performance.now(); },
    setFocusLock(p) { state.lock = p ? p.clone() : null; },
    setFraming(widthMeters) { state.width = widthMeters; reframe(); fitFog(); },
    setInset(px) { if (Math.abs(px - state.inset) > 2) { state.inset = px; applyInset(); reframe(); fitFog(); } },
    setCameraPreset,
    setLightingTheme,
    toggleGrid(show) { grid.visible = !!show; },
  };
}
