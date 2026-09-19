import * as THREE from "../../lib/three.module.min.js";
import { Animator } from "./tween.js";
import { makeTextures } from "./textures.js";

// The room: renderer, lights, the one render loop and the one clock. Everything else in stage/ adds
// meshes to `scene`, animates through `anim`, and asks for per-frame work with onFrame().
const QUALITY = [
  { shadows: 2048, pixelRatio: 2 },
  { shadows: 1024, pixelRatio: 2 },
  { shadows: 0, pixelRatio: 1.5 },
  { shadows: 0, pixelRatio: 1 },
];

export class Stage {
  constructor(container, { reducedMotion = false, fourColour = false } = {}) {
    this.container = container;
    this.anim = new Animator();
    this.anim.reduced = reducedMotion;
    this.frameFns = new Set();
    this.resizeFns = new Set();
    this.onContextRestored = null;

    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.12;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap; // soft-filtered in this three.js release; the old PCFSoft name is gone
    r.setClearColor(0x050607);
    r.domElement.style.cssText = "display:block;width:100%;height:100%;touch-action:none";
    r.domElement.setAttribute("aria-hidden", "true"); // what the canvas shows is also said by the DOM labels and panels
    container.appendChild(r.domElement);

    const scene = (this.scene = new THREE.Scene());
    scene.fog = new THREE.Fog(0x050607, 46, 110);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 300);
    this.textures = makeTextures(r, { fourColour });

    // One warm lamp over the table does the work; the fill keeps shadows from going dead black and the
    // cool rim from behind picks out card edges and chip sides.
    const lamp = (this.lamp = new THREE.SpotLight(0xffdcae, 2900, 90, 0.72, 0.8, 1.9));
    lamp.position.set(0, 26, 3);
    lamp.target.position.set(0, 0, 0.5);
    lamp.castShadow = true;
    lamp.shadow.bias = -0.0004;
    lamp.shadow.normalBias = 0.03;
    lamp.shadow.radius = 5;
    lamp.shadow.camera.near = 8;
    lamp.shadow.camera.far = 40;
    scene.add(lamp, lamp.target);
    scene.add((this.fill = new THREE.HemisphereLight(0x8a93a6, 0x0b0805, 0.3)));
    const rim = (this.rim = new THREE.DirectionalLight(0xa9c2ff, 0.32));
    rim.position.set(0, 10, -30);
    scene.add(rim);

    // Something for polished wood, brass and chip edges to reflect: a dark room with a warm panel where the lamp
    // is and a cool glow behind. Without an environment, metal renders nearly black and gloss has nothing to catch.
    const pmrem = new THREE.PMREMGenerator(r);
    const room = new THREE.Scene();
    room.background = new THREE.Color(0x020303);
    const glow = (hex, strength, w, h, pos) => {
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(hex).multiplyScalar(strength), side: THREE.DoubleSide })
      );
      m.position.set(...pos);
      m.lookAt(0, 0, 0);
      room.add(m);
    };
    glow(0xffd9a8, 14, 16, 10, [0, 26, 3]);
    glow(0x8fb4ff, 1.6, 60, 14, [0, 9, -40]);
    glow(0xffc890, 0.7, 50, 12, [0, 6, 42]);
    scene.environment = pmrem.fromScene(room, 0.03).texture;
    scene.environmentIntensity = 0.55;
    pmrem.dispose();

    this.quality = -1;
    this.setQuality(0);
    this.slowSeconds = 0;
    this.fpsFrames = 0;
    this.fpsTime = 0;

    r.domElement.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.stop();
    });
    r.domElement.addEventListener("webglcontextrestored", () => {
      this.start();
      this.onContextRestored?.();
    });
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
  }

  static supported() {
    try {
      const c = document.createElement("canvas");
      return !!(c.getContext("webgl2") || c.getContext("webgl"));
    } catch {
      return false;
    }
  }

  onResized(fn) {
    this.resizeFns.add(fn);
  }
  onFrame(fn) {
    this.frameFns.add(fn);
    return () => this.frameFns.delete(fn);
  }

  setQuality(level) {
    level = Math.max(0, Math.min(QUALITY.length - 1, level));
    if (level === this.quality) return;
    this.quality = level;
    const q = QUALITY[level];
    this.lamp.castShadow = q.shadows > 0;
    if (q.shadows) {
      this.lamp.shadow.mapSize.set(q.shadows, q.shadows);
      this.lamp.shadow.map?.dispose();
      this.lamp.shadow.map = null;
    }
    this.renderer.shadowMap.needsUpdate = true;
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth || 1,
      h = this.container.clientHeight || 1;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, QUALITY[Math.max(0, this.quality)].pixelRatio));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.portrait = w / h < 1; // a region taller than it is wide gets the quarter-turned table
    for (const fn of this.resizeFns) fn(w, h);
    // Resizing clears the canvas. Draw again at once, or the table is black until the next frame: a flash on a fast
    // device, and a long one on a slow device (exactly where the quality steps that cause resizes happen).
    this.renderer.render(this.scene, this.camera);
  }

  start() {
    if (this.raf) return;
    let last = performance.now();
    const tick = (now) => {
      this.raf = requestAnimationFrame(tick);
      const dt = Math.min(100, now - last); // a background tab must not come back as one giant step
      last = now;
      this.frame(dt);
      this.measure(dt);
    };
    this.raf = requestAnimationFrame(tick);
  }
  stop() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }
  // One frame of animation and drawing. Also called directly for screenshots and tests.
  frame(dt = 16) {
    this.anim.step(dt);
    for (const fn of this.frameFns) fn(dt);
    this.renderer.render(this.scene, this.camera);
  }

  // Step quality down after three slow seconds in a row. It never steps back up: a device that struggled once will again.
  measure(dt) {
    this.fpsFrames++;
    this.fpsTime += dt;
    if (this.fpsTime < 1000) return;
    const fps = (this.fpsFrames * 1000) / this.fpsTime;
    this.fpsFrames = this.fpsTime = 0;
    this.slowSeconds = fps < 40 ? this.slowSeconds + 1 : 0;
    if (this.slowSeconds >= 3) {
      this.slowSeconds = 0;
      this.setQuality(this.quality + 1);
    }
  }

  // World position -> CSS pixels inside the container (or null when behind the camera).
  project(x, y, z) {
    const v = new THREE.Vector3(x, y, z).project(this.camera);
    if (v.z > 1) return null;
    return { x: ((v.x + 1) / 2) * this.container.clientWidth, y: ((1 - v.y) / 2) * this.container.clientHeight };
  }

  dispose() {
    this.stop();
    this.observer.disconnect();
    this.textures.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
