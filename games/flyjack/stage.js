/* FlyJack stage (three.js r128): the felt table under a warm lamp, a card shoe, the real dealt cards and chips, the
   NeuroMechFly body replaying physics-recorded motion clips, and the fly's brain (magnified) replaying the decision
   trial's spikes. Scene units are millimetres, y up. The dealer sits across the table (-z); the fly stands at the
   player spot facing the dealer. */
(function () {
  'use strict';

  const TAU_MS = 20;
  const CARD = { w: 1.25, h: 1.75, t: 0.022 };
  const CHIP = { r: 0.36, t: 0.07 };
  const COLUMN = 20;              // chips per stack
  const MAX_BANK_CHIPS = 140;     // drawing cap; the HUD always shows the true bankroll
  const FELT = { x: 0, z: -4, r: 18 };
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const LAYOUT = {
    fly: V(0, 0, 4),
    player: V(-0.35, 0, 0.35), playerStep: V(0.5, 0, -0.08),
    dealer: V(-1.6, 0, -5.4), dealerStep: V(1.02, 0, 0.04),
    shoe: V(6.8, 0, -7.6), discard: V(-6.8, 0, -7.4), rack: V(0, 0, -9.6),
    bet: V(2.8, 0, 2.9), bank: V(-4.4, 0, 0.8), win: V(3.55, 0, 2.6),
    brain: V(0, 3.7, 3.4),
  };
  const SHOTS = {
    overview: { pos: V(9, 13, 17), target: V(0, 0, -2.5) },
    table: { pos: V(6.5, 8.2, 12.5), target: V(0.4, 0.2, -1.2) },
    think: { pos: V(5.8, 4.4, -6.2), target: V(0.3, 2.9, 3.2) },
    dealer: { pos: V(3.5, 7.2, 4.0), target: V(0.4, 0, -5.2) },
    result: { pos: V(6.8, 5.2, 9.8), target: V(1.4, 0.5, 2.9) },
  };
  // Three hues carry group identity (a 3D scatter puts every pair side by side); others share a neutral tone.
  const GROUP_COLORS = { 'central': '#3987e5', 'mushroom body': '#d95926', 'DN': '#199e70',
                         'photoreceptor': '#c3c2b7', 'optic': '#c3c2b7', 'other': '#c3c2b7' };

  const ease = {
    linear: (t) => t,
    inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    out: (t) => 1 - Math.pow(1 - t, 3),
  };

  class Animator {
    constructor() {
      this.tweens = new Set();
      this.speed = 1;
      this.now = performance.now();
    }
    tween(seconds, update, easing = ease.inOut) {
      return new Promise((resolve) => {
        const tw = { start: null, ms: Math.max(1, seconds * 1000), update, easing, resolve };
        this.tweens.add(tw);
        if (seconds <= 0) this.finish(tw);
      });
    }
    wait(seconds) { return this.tween(seconds, () => {}, ease.linear); }
    finish(tw) {
      tw.update(tw.easing(1));
      this.tweens.delete(tw);
      tw.resolve();
    }
    step(now, dt) {
      this.now = now;
      for (const tw of [...this.tweens]) {
        if (tw.start === null) { tw.start = 0; }
        tw.start += dt * this.speed;
        const t = Math.min(1, tw.start / tw.ms);
        if (t >= 1) this.finish(tw); else tw.update(tw.easing(t));
      }
    }
    skip() { for (const tw of [...this.tweens]) this.finish(tw); }
  }

  // ---------------------------------------------------------------- canvas textures
  function canvasTexture(w, h, draw, renderer) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    draw(c.getContext('2d'), w, h);
    const tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  }

  function roundRect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
  }

  function feltTexture(renderer) {
    const S = 2048;
    const toCanvas = (x, z) => [((x - FELT.x) / FELT.r + 1) * S / 2, ((z - FELT.z) / FELT.r + 1) * S / 2];
    const perMm = S / (2 * FELT.r);
    return canvasTexture(S, S, (g) => {
      const grad = g.createRadialGradient(S / 2, S * 0.62, S * 0.05, S / 2, S / 2, S * 0.55);
      grad.addColorStop(0, '#11703c');
      grad.addColorStop(1, '#063a1d');
      g.fillStyle = grad;
      g.fillRect(0, 0, S, S);
      for (let i = 0; i < 160000; i++) {                     // felt grain
        g.fillStyle = Math.random() < 0.5 ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.035)';
        g.fillRect(Math.random() * S, Math.random() * S, 1.5, 1.5);
      }
      const ink = 'rgba(236, 214, 160, 0.78)';
      const arcText = (text, cx, cz, radius, sizeMm, spacing) => {
        const [x0, y0] = toCanvas(cx, cz);
        const r = radius * perMm;
        g.font = `600 ${sizeMm * perMm}px Georgia, "Times New Roman", serif`;
        g.fillStyle = ink;
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        const widths = [...text].map((ch) => g.measureText(ch).width * spacing);
        const total = widths.reduce((a, b) => a + b, 0);
        let a = Math.PI / 2 + total / r / 2;                 // start left of the arc's lowest point, read left to right
        [...text].forEach((ch, i) => {
          const da = widths[i] / r;
          const mid = a - da / 2;
          g.save();
          g.translate(x0 + r * Math.cos(mid), y0 + r * Math.sin(mid));
          g.rotate(mid - Math.PI / 2);
          g.fillText(ch, 0, 0);
          g.restore();
          a -= da;
        });
      };
      arcText('DEALER MUST STAND ON 17', LAYOUT.dealer.x + 2, -12.5, 9.6, 0.62, 1.12);
      arcText('BLACKJACK-V1  ·  NO DOUBLING  ·  NO SPLITTING', LAYOUT.dealer.x + 2, -12.5, 8.4, 0.34, 1.1);
      g.strokeStyle = ink;
      g.lineWidth = 0.05 * perMm;
      const [bx, by] = toCanvas(LAYOUT.bet.x, LAYOUT.bet.z);
      g.beginPath();
      g.arc(bx, by, 0.72 * perMm, 0, Math.PI * 2);
      g.stroke();
      const card = (pos) => {
        const [x, y] = toCanvas(pos.x, pos.z);
        g.setLineDash([0.12 * perMm, 0.1 * perMm]);
        roundRect(g, x - 0.72 * perMm, y - 0.98 * perMm, 1.44 * perMm, 1.96 * perMm, 0.12 * perMm);
        g.stroke();
        g.setLineDash([]);
      };
      card(LAYOUT.player.clone().add(LAYOUT.playerStep.clone().multiplyScalar(0.5)));
      g.font = `600 ${0.42 * perMm}px Georgia, serif`;
      g.fillStyle = ink;
      g.textAlign = 'center';
      const [wx, wy] = toCanvas(0, -15.2);
      g.fillText('F L Y J A C K', wx, wy);
    }, renderer);
  }

  const SUIT_COLOR = { '♥': '#b3261e', '♦': '#b3261e', '♠': '#161616', '♣': '#161616' };
  function cardFace(renderer, rank, suit) {
    return canvasTexture(256, 358, (g, w, h) => {
      g.fillStyle = '#fbfaf5';
      roundRect(g, 0, 0, w, h, 22);
      g.fill();
      g.strokeStyle = '#cfcac0';
      g.lineWidth = 4;
      roundRect(g, 3, 3, w - 6, h - 6, 20);
      g.stroke();
      g.fillStyle = SUIT_COLOR[suit];
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      const corner = () => {
        g.font = 'bold 62px Georgia, serif';
        g.fillText(rank, 40, 46);
        g.font = '48px Georgia, serif';
        g.fillText(suit, 40, 100);
      };
      corner();
      g.save();
      g.translate(w, h);
      g.rotate(Math.PI);
      corner();
      g.restore();
      g.font = rank.length === 1 && 'JQK'.includes(rank) ? 'bold 150px Georgia, serif' : '150px Georgia, serif';
      g.fillText(/[JQK]/.test(rank) ? rank : suit, w / 2, h / 2 + 6);
    }, renderer);
  }
  function cardBack(renderer) {
    return canvasTexture(256, 358, (g, w, h) => {
      g.fillStyle = '#fbfaf5';
      roundRect(g, 0, 0, w, h, 22);
      g.fill();
      g.fillStyle = '#7a1420';
      roundRect(g, 16, 16, w - 32, h - 32, 12);
      g.fill();
      g.save();
      roundRect(g, 16, 16, w - 32, h - 32, 12);
      g.clip();
      g.strokeStyle = 'rgba(255, 210, 170, 0.35)';
      g.lineWidth = 4;
      for (let k = -h; k < w + h; k += 20) {
        g.beginPath(); g.moveTo(k, 0); g.lineTo(k + h, h); g.stroke();
        g.beginPath(); g.moveTo(k + h, 0); g.lineTo(k, h); g.stroke();
      }
      g.restore();
    }, renderer);
  }
  function chipTextures(renderer, color) {
    const top = canvasTexture(128, 128, (g, w) => {
      g.fillStyle = color;
      g.beginPath(); g.arc(w / 2, w / 2, w / 2, 0, Math.PI * 2); g.fill();
      g.strokeStyle = '#f4efe4';
      g.lineWidth = 9;
      g.setLineDash([14, 12]);
      g.beginPath(); g.arc(w / 2, w / 2, w / 2 - 8, 0, Math.PI * 2); g.stroke();
      g.setLineDash([]);
      g.lineWidth = 3;
      g.beginPath(); g.arc(w / 2, w / 2, w / 2 - 26, 0, Math.PI * 2); g.stroke();
    }, renderer);
    const side = canvasTexture(256, 16, (g, w, h) => {
      g.fillStyle = color;
      g.fillRect(0, 0, w, h);
      g.fillStyle = '#f4efe4';
      for (let i = 0; i < 8; i++) g.fillRect(i * 32 + 8, 0, 14, h);
    }, renderer);
    return { top, side };
  }

  // ---------------------------------------------------------------- the stage
  class Stage {
    constructor(canvas) {
      this.canvas = canvas;
      this.anim = new Animator();
      this.reducedMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.cards = [];
      this.chips = { bet: [], bank: [], moving: [] };
      this.onSound = () => {};
      this.onTick = () => {};
      this.quality = 0;
      this.fps = 60;
      this.clip = null;
      this.userOrbit = false;
    }

    async init({ cfg, neuronsBuf, meshJson, meshBuf, clipsJson, clipsBuf }) {
      const r = this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, powerPreference: 'high-performance' });
      r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      r.outputEncoding = THREE.sRGBEncoding;
      r.toneMapping = THREE.ACESFilmicToneMapping;
      r.toneMappingExposure = 1.05;
      r.shadowMap.enabled = true;
      r.shadowMap.type = THREE.PCFSoftShadowMap;
      const scene = this.scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050607);
      scene.fog = new THREE.Fog(0x050607, 30, 70);
      this.camera = new THREE.PerspectiveCamera(35, 1, 0.05, 300);
      this.camTarget = SHOTS.overview.target.clone();
      this.camera.position.copy(SHOTS.overview.pos);
      this.buildLights();
      this.buildTable();
      this.buildFly(meshJson, meshBuf, clipsJson, clipsBuf);
      this.buildBrain(cfg, neuronsBuf);
      this.rebuildBank(100);
      this.bindInput();
      const resize = () => {
        const el = this.canvas.parentElement;
        const w = el.clientWidth, h = Math.max(1, el.clientHeight);
        r.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
      };
      new ResizeObserver(resize).observe(this.canvas.parentElement);
      resize();
      this.last = performance.now();
      this.fpsWindow = { t: this.last, n: 0, low: 0 };
      requestAnimationFrame((t) => this.tick(t));
    }

    buildLights() {
      const lamp = this.lamp = new THREE.SpotLight(0xffd6a0, 2.6, 90, 0.62, 0.75, 1.1);
      lamp.position.set(0.5, 34, -1.5);
      lamp.target.position.set(0.3, 0, -1.8);
      lamp.castShadow = true;
      lamp.shadow.mapSize.set(2048, 2048);
      lamp.shadow.camera.near = 20;
      lamp.shadow.camera.far = 45;
      lamp.shadow.bias = -0.0004;
      lamp.shadow.radius = 3;
      this.scene.add(lamp, lamp.target);
      this.ambient = new THREE.HemisphereLight(0x6f7a8c, 0x0b0805, 0.32);
      this.scene.add(this.ambient);
      const rim = this.rim = new THREE.DirectionalLight(0x8fb4ff, 0.35);
      rim.position.set(-8, 7, -12);
      this.scene.add(rim);
      // a faint lamp-shade glow high above the table
      const shade = new THREE.Mesh(new THREE.SphereGeometry(1.4, 24, 12),
        new THREE.MeshBasicMaterial({ color: 0xffc27a, transparent: true, opacity: 0.55, fog: false }));
      shade.position.copy(lamp.position);
      this.scene.add(shade);
    }

    buildTable() {
      const felt = new THREE.Mesh(new THREE.CircleGeometry(FELT.r, 128),
        new THREE.MeshStandardMaterial({ map: feltTexture(this.renderer), roughness: 0.95, metalness: 0 }));
      felt.rotation.x = -Math.PI / 2;
      felt.position.set(FELT.x, 0, FELT.z);
      felt.receiveShadow = true;
      this.scene.add(felt);
      const rail = new THREE.Mesh(new THREE.TorusGeometry(FELT.r + 0.55, 0.75, 20, 160),
        new THREE.MeshStandardMaterial({ color: 0x3b2314, roughness: 0.45, metalness: 0.05 }));
      rail.rotation.x = -Math.PI / 2;
      rail.position.set(FELT.x, 0.25, FELT.z);
      rail.receiveShadow = true;
      rail.castShadow = true;
      this.scene.add(rail);
      const wood = new THREE.Mesh(new THREE.CylinderGeometry(FELT.r + 1.4, FELT.r + 1.6, 1.6, 128),
        new THREE.MeshStandardMaterial({ color: 0x1c110a, roughness: 0.7 }));
      wood.position.set(FELT.x, -0.81, FELT.z);
      this.scene.add(wood);
      // card shoe: a dark angled box with a slot, and a discard tray
      const shoe = new THREE.Group();
      const body = new THREE.Mesh(new THREE.BoxGeometry(2.2, 1.0, 2.6),
        new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.35, metalness: 0.2 }));
      body.position.y = 0.5;
      body.castShadow = true;
      shoe.add(body);
      const lip = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.08, 0.5),
        new THREE.MeshStandardMaterial({ color: 0xb08d57, roughness: 0.3, metalness: 0.8 }));
      lip.position.set(0, 0.35, 1.3);
      shoe.add(lip);
      shoe.position.copy(LAYOUT.shoe);
      shoe.rotation.y = -0.5;
      this.scene.add(shoe);
      const tray = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.35, 2.3),
        new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.85, metalness: 0 }));
      tray.position.copy(LAYOUT.discard).setY(0.18);
      tray.rotation.y = 0.5;
      tray.castShadow = true;
      this.scene.add(tray);
      this.textures = { back: cardBack(this.renderer), faces: {}, chip: chipTextures(this.renderer, '#a4161a') };
      this.cardSide = new THREE.MeshStandardMaterial({ color: 0xf1eee6, roughness: 0.6 });
      this.cardBackMat = new THREE.MeshStandardMaterial({ map: this.textures.back, roughness: 0.55 });
      this.chipGeo = new THREE.CylinderGeometry(CHIP.r, CHIP.r, CHIP.t, 32);
      this.chipMats = [
        new THREE.MeshStandardMaterial({ map: this.textures.chip.side, roughness: 0.5 }),
        new THREE.MeshStandardMaterial({ map: this.textures.chip.top, roughness: 0.45 }),
        new THREE.MeshStandardMaterial({ map: this.textures.chip.top, roughness: 0.45 }),
      ];
    }

    // ---------------------------------------------------------------- fly
    buildFly(meshJson, meshBuf, clipsJson, clipsBuf) {
      const outer = this.flyOuter = new THREE.Group();          // places the fly on the table, facing the dealer
      outer.position.copy(LAYOUT.fly);
      outer.rotation.y = Math.PI / 2;                            // MuJoCo +x (fly forward) -> world -z
      const inner = new THREE.Group();                           // MuJoCo z-up -> three y-up
      inner.rotation.x = -Math.PI / 2;
      outer.add(inner);
      this.scene.add(outer);
      const posBase = 0, nrmBase = meshJson.position_bytes, idxBase = meshJson.position_bytes + meshJson.normal_bytes;
      this.flyParts = meshJson.parts.map((p) => {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(meshBuf, posBase + p.vert_offset * 12, p.vert_count * 3), 3));
        geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(meshBuf, nrmBase + p.vert_offset * 12, p.vert_count * 3), 3));
        let idx = new Uint32Array(meshBuf, idxBase + p.index_offset * 4, p.index_count);
        let maxIdx = 0;
        for (let i = 0; i < idx.length; i++) if (idx[i] > maxIdx) maxIdx = idx[i];
        if (maxIdx >= p.vert_count) idx = idx.map((v) => v - p.vert_offset);   // indices stored globally
        geo.setIndex(new THREE.BufferAttribute(idx, 1));
        const [cr, cg, cb, ca] = p.rgba;
        const color = new THREE.Color().setRGB(cr, cg, cb).convertSRGBToLinear();
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05,
                                                     transparent: ca < 0.999, opacity: ca, depthWrite: ca >= 0.999,
                                                     side: ca < 0.999 ? THREE.DoubleSide : THREE.FrontSide });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = ca >= 0.5;
        mesh.matrixAutoUpdate = true;
        inner.add(mesh);
        return mesh;
      });
      this.clips = clipsJson;
      this.clipData = new Float32Array(clipsBuf);
      this.nParts = meshJson.parts.length;
      const idle = clipsJson.clips.idle;
      this.setFlyFrame(idle.offset, idle.offset, 0);             // posed before the first animation frame
      this.playClip('idle');
    }

    setFlyFrame(frameA, frameB, u) {
      const d = this.clipData, P = this.nParts, qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
      for (let i = 0; i < P; i++) {
        const a = (frameA * P + i) * 7, b = (frameB * P + i) * 7;
        const mesh = this.flyParts[i];
        mesh.position.set(d[a] + (d[b] - d[a]) * u, d[a + 1] + (d[b + 1] - d[a + 1]) * u, d[a + 2] + (d[b + 2] - d[a + 2]) * u);
        qa.set(d[a + 4], d[a + 5], d[a + 6], d[a + 3]);
        qb.set(d[b + 4], d[b + 5], d[b + 6], d[b + 3]);
        mesh.quaternion.copy(qa).slerp(qb, u);
      }
    }

    /** Play a motion clip; resolves when a non-looping clip ends (the fly then returns to idle). */
    playClip(name) {
      const c = this.clips.clips[name];
      if (!c) return Promise.resolve();
      if (this.clip && this.clip.resolve) this.clip.resolve();
      return new Promise((resolve) => {
        this.clip = { name, c, t: 0, resolve: c.loop ? null : resolve };
        if (c.loop) resolve();
      });
    }

    updateFly(dt) {
      if (!this.clip) return;
      const { c } = this.clip;
      this.clip.t += dt * this.anim.speed / 1000;
      let f = this.clip.t * this.clips.fps;
      if (c.loop) f %= (c.frames - 1);
      else if (f >= c.frames - 1) {
        this.setFlyFrame(c.offset + c.frames - 1, c.offset + c.frames - 1, 0);
        const done = this.clip.resolve;
        this.clip = null;
        this.playClip('idle');
        if (done) done();
        return;
      }
      const f0 = Math.floor(f);
      this.setFlyFrame(c.offset + f0, c.offset + Math.min(f0 + 1, c.frames - 1), f - f0);
    }

    // ---------------------------------------------------------------- brain
    buildBrain(cfg, buf) {
      const N = this.N = cfg.neurons.N;
      const raw = new Float32Array(buf, 0, 3 * N);
      this.groups = new Uint8Array(buf, 12 * N, N);
      const span = cfg.neurons.bbox_um[1][0] - cfg.neurons.bbox_um[0][0];
      const s = 8.0 / span;                                      // ~8 mm wide: about 10x the real brain
      this.brainMagnification = 8.0 / (span / 1000);
      const pos = new Float32Array(3 * N);
      for (let i = 0; i < N; i++) {                              // FlyWire x lateral, y ventral, z depth
        pos[3 * i] = raw[3 * i] * s;
        pos[3 * i + 1] = -raw[3 * i + 1] * s;
        pos[3 * i + 2] = raw[3 * i + 2] * s;
      }
      const stim = new Float32Array(N);
      for (const i of cfg.stimulated_idx) stim[i] = 1;
      const g = this.brainGeo = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('aGroup', new THREE.BufferAttribute(Float32Array.from(this.groups), 1));
      g.setAttribute('aTLast', new THREE.BufferAttribute(new Float32Array(N).fill(-1e9), 1).setUsage(THREE.DynamicDrawUsage));
      g.setAttribute('aHighlight', new THREE.BufferAttribute(new Float32Array(N), 1).setUsage(THREE.DynamicDrawUsage));
      g.setAttribute('aStim', new THREE.BufferAttribute(stim, 1));
      const hex = (h) => new THREE.Vector3(...[1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16) / 255));
      this.brainMat = new THREE.ShaderMaterial({
        uniforms: { uColors: { value: cfg.neurons.groups.map((name) => hex(GROUP_COLORS[name] || '#c3c2b7')) },
                    uVisible: { value: cfg.neurons.groups.map(() => 1) }, uTime: { value: 0 }, uTau: { value: TAU_MS },
                    uHighlightOn: { value: 0 }, uSize: { value: 0.010 }, uPx: { value: 800 }, uOpacity: { value: 0 } },
        vertexShader: `
          attribute float aGroup; attribute float aTLast; attribute float aHighlight; attribute float aStim;
          uniform vec3 uColors[6]; uniform float uVisible[6]; uniform float uTime; uniform float uTau;
          uniform float uHighlightOn; uniform float uSize; uniform float uPx; uniform float uOpacity;
          varying vec3 vColor; varying float vAlpha;
          void main() {
            int g = int(aGroup + 0.5);
            float dt = uTime - aTLast;
            float b = dt >= 0.0 ? exp(-dt / uTau) : 0.0;
            vec3 hot = aStim > 0.5 ? vec3(1.0) : uColors[g];
            vColor = mix(vec3(0.16, 0.19, 0.24), hot, min(1.0, 1.5 * b));
            float a = 0.035 + 0.965 * b;
            float size = uSize * (1.0 + 2.4 * b);
            if (uHighlightOn > 0.5) {
              if (aHighlight > 0.5) { a = max(a, 0.9); size = max(size, 3.0 * uSize); vColor = mix(vColor, vec3(1.0), 0.5); }
              else { a *= 0.3; }
            }
            vAlpha = a * uVisible[g] * uOpacity;
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = clamp(size * uPx / -mv.z, 1.0, 14.0);
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: `
          varying vec3 vColor; varying float vAlpha;
          void main() {
            vec2 c = gl_PointCoord - vec2(0.5);
            float r2 = dot(c, c);
            if (r2 > 0.25 || vAlpha < 0.003) discard;
            gl_FragColor = vec4(vColor * vAlpha * (1.0 - 3.0 * r2), 1.0);
          }`,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
      });
      this.brain = new THREE.Points(g, this.brainMat);
      this.brain.position.copy(LAYOUT.brain);
      this.brain.frustumCulled = false;
      this.brain.visible = false;
      this.scene.add(this.brain);
      this.frames = null;
      this.applied = -1;
      this.frameMs = cfg.frame_ms;
      this.onFrames = () => {};
    }

    loadFrames(buf) {
      const a = new Int32Array(buf);
      const n = a[0];
      this.frames = { n, offsets: a.subarray(1, n + 2), idx: a.subarray(n + 2) };
      this.seekBrain(0);
    }

    seekBrain(tMs) {
      this.applied = -1;
      this.brainGeo.attributes.aTLast.array.fill(-1e9);
      this.brainGeo.attributes.aTLast.needsUpdate = true;
      this.brainMat.uniforms.uTime.value = tMs;
      if (this.frames) this.applyBrainTo(tMs);
    }

    applyBrainTo(tMs) {
      this.brainMat.uniforms.uTime.value = tMs;
      if (!this.frames) return;
      const f = Math.min(Math.floor(tMs / this.frameMs), this.frames.n - 1);
      if (f <= this.applied) return;
      const t = this.brainGeo.attributes.aTLast.array;
      for (let k = this.applied + 1; k <= f; k++) {
        const lo = this.frames.offsets[k], hi = this.frames.offsets[k + 1];
        for (let j = lo; j < hi; j++) t[this.frames.idx[j]] = k * this.frameMs;
        this.onFrames(this.frames.idx.subarray(lo, hi), k);
      }
      this.applied = f;
      this.brainGeo.attributes.aTLast.needsUpdate = true;
    }

    showBrain(on, seconds = 0.8) {
      const u = this.brainMat.uniforms.uOpacity;
      const from = u.value, to = on ? 1 : 0;
      const lampFrom = this.lamp.intensity, lampTo = on ? 1.1 : 2.6;
      this.brain.visible = true;
      return this.anim.tween(this.reducedMotion ? 0 : seconds, (t) => {
        u.value = from + (to - from) * t;
        this.lamp.intensity = lampFrom + (lampTo - lampFrom) * t;
      }).then(() => { if (!on) this.brain.visible = false; });
    }

    setGroupVisible(i, on) { this.brainMat.uniforms.uVisible.value[i] = on ? 1 : 0; }

    highlight(indices) {
      const a = this.brainGeo.attributes.aHighlight;
      a.array.fill(0);
      for (const i of indices) a.array[i] = 1;
      a.needsUpdate = true;
      this.brainMat.uniforms.uHighlightOn.value = indices.length ? 1 : 0;
    }

    // ---------------------------------------------------------------- cards and chips
    faceMaterial(card) {
      const key = card.rank + card.suit;
      if (!this.textures.faces[key]) {
        this.textures.faces[key] = new THREE.MeshStandardMaterial({ map: cardFace(this.renderer, card.rank, card.suit), roughness: 0.55 });
      }
      return this.textures.faces[key];
    }

    cardSlot(owner, index) {
      const base = owner === 'player' ? LAYOUT.player : LAYOUT.dealer;
      const step = owner === 'player' ? LAYOUT.playerStep : LAYOUT.dealerStep;
      return base.clone().add(step.clone().multiplyScalar(index)).setY(CARD.t / 2 + 0.004 + 0.03 * index);
    }

    /** Deal one card from the shoe to its slot. card = null deals an unknown face-down card (the dealer's hole card,
        which the server reveals only when the hand ends); revealHole(card) gives it a face. */
    async dealCard(owner, card, faceUp = true) {
      const index = this.cards.filter((c) => c.owner === owner).length;
      // box faces: +x, -x, +y (face), -y (back), +z, -z
      const face = card ? this.faceMaterial(card) : this.cardBackMat;
      const mats = [this.cardSide, this.cardSide, face, this.cardBackMat, this.cardSide, this.cardSide];
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(CARD.w, CARD.t, CARD.h), mats);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const from = LAYOUT.shoe.clone().add(V(-0.6, 0.9, 1.4));
      const to = this.cardSlot(owner, index);
      const yaw = (Math.random() - 0.5) * 0.08;               // card tops face away from the camera: faces read upright
      mesh.position.copy(from);
      mesh.rotation.set(0, yaw - 0.6, Math.PI);
      this.scene.add(mesh);
      const entry = { mesh, owner, faceUp: false, rank: card ? card.rank : null, suit: card ? card.suit : null,
                      value: card ? card.value : null };
      this.cards.push(entry);
      this.onSound('slide');
      const dur = this.reducedMotion ? 0.2 : 0.55;
      await this.anim.tween(dur, (t) => {
        mesh.position.lerpVectors(from, to, t);
        mesh.position.y += Math.sin(Math.PI * t) * (this.reducedMotion ? 0.3 : 1.6);
        mesh.rotation.y = yaw - 0.6 * (1 - t);
      }, ease.out);
      if (faceUp) await this.flipCard(entry);
      return entry;
    }

    async flipCard(entry) {
      const { mesh } = entry;
      const y0 = mesh.position.y;
      this.onSound('flip');
      await this.anim.tween(this.reducedMotion ? 0.05 : 0.3, (t) => {
        mesh.rotation.z = Math.PI * (1 - t);
        mesh.position.y = y0 + Math.sin(Math.PI * t) * 0.7;
      });
      entry.faceUp = true;
    }

    async revealHole(card) {
      const hole = this.cards.find((c) => c.owner === 'dealer' && !c.faceUp);
      if (!hole) return;
      if (card) {
        hole.mesh.material[2] = this.faceMaterial(card);
        Object.assign(hole, { rank: card.rank, suit: card.suit, value: card.value });
      }
      await this.flipCard(hole);
    }

    makeChip() {
      const m = new THREE.Mesh(this.chipGeo, this.chipMats);
      m.castShadow = true;
      m.receiveShadow = true;
      this.scene.add(m);
      return m;
    }

    /** One chip is one unit. Stacks hold COLUMN chips, like a casino rail; columns run to the fly's left. */
    stackPos(base, i) {
      const col = Math.floor(i / COLUMN), row = i % COLUMN;
      return base.clone().add(V(-0.85 * col + Math.sin(i * 1.7) * 0.02, 0, Math.cos(i * 2.3) * 0.02))
        .setY(CHIP.t / 2 + row * CHIP.t + 0.001);
    }

    rebuildBank(bankroll) {
      for (const m of this.chips.bank) this.scene.remove(m);
      const n = Math.max(0, Math.min(MAX_BANK_CHIPS, Math.round(bankroll)));   // 1 chip = 1 unit, capped for drawing
      this.chips.bank = Array.from({ length: n }, (_, i) => {
        const m = this.makeChip();
        m.position.copy(this.stackPos(LAYOUT.bank, i));
        m.rotation.y = i * 0.9;
        return m;
      });
    }

    async moveChips(chips, target, stagger = 0.06) {
      const moves = chips.map((m, i) => {
        const from = m.position.clone();
        const to = this.stackPos(target, i);
        return this.anim.wait(i * stagger).then(() => {
          this.onSound('chip');
          return this.anim.tween(this.reducedMotion ? 0.1 : 0.45, (t) => {
            m.position.lerpVectors(from, to, t);
            m.position.y += Math.sin(Math.PI * t) * 0.8;
          }, ease.out);
        });
      });
      await Promise.all(moves);
    }

    async placeBet(units = 1) {
      const n = Math.max(1, Math.min(units, this.chips.bank.length));
      const chips = this.chips.bank.splice(this.chips.bank.length - n, n);
      this.chips.bet = chips;
      await this.moveChips(chips, LAYOUT.bet);
    }

    /** outcome +1: dealer pays a matching stack; -1: dealer takes the bet; 0: push, the bet comes back. */
    async settle(outcome, bankroll) {
      const bet = this.chips.bet;
      if (outcome > 0) {
        const paid = bet.map(() => {
          const m = this.makeChip();
          m.position.copy(this.stackPos(LAYOUT.rack, 0));
          return m;
        });
        await this.moveChips(paid, LAYOUT.win);
        await this.anim.wait(0.25);
        await this.moveChips([...bet, ...paid], LAYOUT.bank.clone().add(V(0.9, 0, 0)));
        [...bet, ...paid].forEach((m) => this.scene.remove(m));
      } else if (outcome < 0) {
        await this.moveChips(bet, LAYOUT.rack);
        bet.forEach((m) => this.scene.remove(m));
      } else {
        await this.moveChips(bet, LAYOUT.bank.clone().add(V(0.9, 0, 0)));
        bet.forEach((m) => this.scene.remove(m));
      }
      this.chips.bet = [];
      this.rebuildBank(bankroll);
    }

    async clearTable() {
      const cards = this.cards;
      this.cards = [];
      await Promise.all(cards.map((c, i) => {
        const from = c.mesh.position.clone();
        const to = LAYOUT.discard.clone().setY(0.4 + i * 0.02);
        return this.anim.wait(i * 0.05).then(() => this.anim.tween(this.reducedMotion ? 0.1 : 0.45, (t) => {
          c.mesh.position.lerpVectors(from, to, t);
          c.mesh.position.y += Math.sin(Math.PI * t) * 0.8;
          c.mesh.rotation.z = c.faceUp ? Math.PI * t : Math.PI;
        })).then(() => this.scene.remove(c.mesh));
      }));
      for (const m of this.chips.bet) this.scene.remove(m);
      this.chips.bet = [];
    }

    // ---------------------------------------------------------------- camera
    shot(name, seconds = 1.2) {
      const s = SHOTS[name];
      this.userOrbit = false;
      const p0 = this.camera.position.clone(), t0 = this.camTarget.clone();
      return this.anim.tween(this.reducedMotion ? 0 : seconds, (t) => {
        this.camera.position.lerpVectors(p0, s.pos, t);
        this.camTarget.lerpVectors(t0, s.target, t);
      });
    }

    bindInput() {
      let drag = null;
      const el = this.canvas;
      el.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); });
      el.addEventListener('pointerup', () => { drag = null; });
      el.addEventListener('pointermove', (e) => {
        if (!drag) return;
        const off = this.camera.position.clone().sub(this.camTarget);
        const sph = new THREE.Spherical().setFromVector3(off);
        sph.theta -= (e.clientX - drag.x) * 0.005;
        sph.phi = Math.min(1.5, Math.max(0.15, sph.phi - (e.clientY - drag.y) * 0.005));
        this.camera.position.copy(this.camTarget).add(new THREE.Vector3().setFromSpherical(sph));
        drag = { x: e.clientX, y: e.clientY };
        this.userOrbit = true;
      });
      el.addEventListener('wheel', (e) => {
        e.preventDefault();
        const off = this.camera.position.clone().sub(this.camTarget);
        const len = Math.min(45, Math.max(2.5, off.length() * Math.exp(e.deltaY * 0.001)));
        this.camera.position.copy(this.camTarget).add(off.setLength(len));
        this.userOrbit = true;
      }, { passive: false });
    }

    project(vec) {
      const v = vec.clone().project(this.camera);
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      return { x: (v.x + 1) / 2 * w, y: (1 - v.y) / 2 * h, visible: v.z < 1 && Math.abs(v.x) < 1.1 && Math.abs(v.y) < 1.1 };
    }

    anchors() {
      const top = (owner) => {
        const cards = this.cards.filter((c) => c.owner === owner);
        const last = cards.length ? cards[cards.length - 1].mesh.position : this.cardSlot(owner, 0);
        return this.project(last.clone().add(V(0.4, 0.3, owner === 'player' ? 1.3 : -1.3)));
      };
      return { player: top('player'), dealer: top('dealer'), brain: this.project(LAYOUT.brain.clone().add(V(0, 4.2, 0))),
               bank: this.project(LAYOUT.bank.clone().add(V(0, 1.6, 0))) };
    }

    debugCards() { return this.cards.map((c) => ({ owner: c.owner, rank: c.rank, suit: c.suit, faceUp: c.faceUp })); }

    setQuality(tier) {
      this.quality = tier;
      this.lamp.castShadow = tier < 2;
      if (tier === 1) { this.lamp.shadow.mapSize.set(1024, 1024); if (this.lamp.shadow.map) { this.lamp.shadow.map.dispose(); this.lamp.shadow.map = null; } }
      this.renderer.setPixelRatio(tier === 0 ? Math.min(window.devicePixelRatio || 1, 2) : tier === 1 ? 1.25 : 1);
      this.canvas.parentElement.dispatchEvent(new Event('resize'));
      const el = this.canvas.parentElement;
      this.renderer.setSize(el.clientWidth, el.clientHeight, false);
    }

    tick(now) {
      const dt = Math.min(100, now - this.last);
      this.last = now;
      this.ticks = (this.ticks || 0) + 1;
      this.anim.step(now, dt);
      this.updateFly(dt);
      this.onTick(dt);
      this.camera.lookAt(this.camTarget);
      const px = this.renderer.getDrawingBufferSize(new THREE.Vector2()).y / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
      this.brainMat.uniforms.uPx.value = px;
      this.renderer.render(this.scene, this.camera);
      const w = this.fpsWindow;
      w.n++;
      if (now - w.t >= 1000) {
        this.fps = Math.round(w.n * 1000 / (now - w.t));
        w.low = this.fps < 30 ? w.low + 1 : 0;
        if (w.low >= 3 && this.quality < 2) { this.setQuality(this.quality + 1); w.low = 0; }
        w.t = now;
        w.n = 0;
      }
      requestAnimationFrame((t) => this.tick(t));
    }
  }

  window.FlyStage = { Stage, LAYOUT, SHOTS, GROUP_COLORS, TAU_MS };
})();
