import * as THREE from "../../vendor/three.module.min.js";
import { TABLE, DENOMS } from "./layout.js";

// Everything on the table is painted here, into 2D canvases: no image files. The painters take a plain
// canvas context, so the DOM panels can draw exactly the same cards as the 3D table does.

const RANKS = "23456789TJQKA";
const INK = { black: "#17171a", red: "#b3261e", blue: "#1d5fbf", green: "#1d7a3a" };
const STOCK = "#fbfaf5";
export const BRASS = "#d8b36a";

// Suits are indexed as in core/cards.js: 0 spade, 1 heart, 2 diamond, 3 club.
export function suitColour(s, fourColour) {
  if (fourColour) return [INK.black, INK.red, INK.blue, INK.green][s];
  return s === 1 || s === 2 ? INK.red : INK.black;
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

// A suit pip as a path, centred on (cx, cy) and about `size` tall. Drawn rather than typeset so that it
// looks the same on every machine, whatever fonts are installed.
export function paintPip(g, s, cx, cy, size) {
  const u = size / 2;
  g.save();
  g.translate(cx, cy);
  g.beginPath();
  if (s === 2) {
    g.moveTo(0, -u);
    g.quadraticCurveTo(0.34 * u, -0.36 * u, 0.78 * u, 0);
    g.quadraticCurveTo(0.34 * u, 0.36 * u, 0, u);
    g.quadraticCurveTo(-0.34 * u, 0.36 * u, -0.78 * u, 0);
    g.quadraticCurveTo(-0.34 * u, -0.36 * u, 0, -u);
  } else if (s === 1) {
    g.moveTo(0, 0.95 * u);
    g.bezierCurveTo(-0.35 * u, 0.55 * u, -1.0 * u, 0.1 * u, -1.0 * u, -0.42 * u);
    g.bezierCurveTo(-1.0 * u, -0.98 * u, -0.22 * u, -1.12 * u, 0, -0.5 * u);
    g.bezierCurveTo(0.22 * u, -1.12 * u, 1.0 * u, -0.98 * u, 1.0 * u, -0.42 * u);
    g.bezierCurveTo(1.0 * u, 0.1 * u, 0.35 * u, 0.55 * u, 0, 0.95 * u);
  } else {
    if (s === 0) {
      g.moveTo(0, -1.0 * u);
      g.bezierCurveTo(0.3 * u, -0.55 * u, 1.0 * u, -0.15 * u, 1.0 * u, 0.32 * u);
      g.bezierCurveTo(1.0 * u, 0.86 * u, 0.3 * u, 0.92 * u, 0.06 * u, 0.5 * u);
      g.lineTo(-0.06 * u, 0.5 * u);
      g.bezierCurveTo(-0.3 * u, 0.92 * u, -1.0 * u, 0.86 * u, -1.0 * u, 0.32 * u);
      g.bezierCurveTo(-1.0 * u, -0.15 * u, -0.3 * u, -0.55 * u, 0, -1.0 * u);
    } else {
      const r = 0.44 * u;
      g.arc(0, -0.5 * u, r, 0, Math.PI * 2);
      g.moveTo(-0.5 * u + r, 0.16 * u);
      g.arc(-0.5 * u, 0.16 * u, r, 0, Math.PI * 2);
      g.moveTo(0.5 * u + r, 0.16 * u);
      g.arc(0.5 * u, 0.16 * u, r, 0, Math.PI * 2);
      g.moveTo(0.2 * u, 0);
      g.arc(0, 0, 0.2 * u, 0, Math.PI * 2);
    }
    // the stem that spades and clubs share
    g.moveTo(0, 0.3 * u);
    g.quadraticCurveTo(-0.04 * u, 0.8 * u, -0.34 * u, 1.0 * u);
    g.lineTo(0.34 * u, 1.0 * u);
    g.quadraticCurveTo(0.04 * u, 0.8 * u, 0, 0.3 * u);
  }
  g.fill();
  g.restore();
}

// Built to be read from across the table, where a card may be fifty pixels wide: one huge rank, one big pip,
// nothing else. (Mirrored corner indices and pip layouts turn to noise at that size.)
export function paintCardFace(g, w, h, card, { fourColour = false } = {}) {
  const rank = RANKS[card.r - 2] === "T" ? "10" : RANKS[card.r - 2];
  g.clearRect(0, 0, w, h);
  g.fillStyle = STOCK;
  roundRect(g, 0, 0, w, h, w * 0.085);
  g.fill();
  const shade = g.createLinearGradient(0, 0, w, h);
  shade.addColorStop(0, "rgba(255,255,255,0)");
  shade.addColorStop(1, "rgba(120,100,70,0.12)");
  g.fillStyle = shade;
  g.fill();
  g.strokeStyle = "rgba(60,50,40,0.3)";
  g.lineWidth = w * 0.012;
  roundRect(g, w * 0.012, w * 0.012, w * 0.976, h - w * 0.024, w * 0.078);
  g.stroke();

  g.fillStyle = suitColour(card.s, fourColour);
  g.textAlign = "center";
  g.textBaseline = "alphabetic";
  g.font = `700 ${w * (rank === "10" ? 0.6 : 0.74)}px Georgia, "Times New Roman", serif`;
  g.fillText(rank, w * (rank === "10" ? 0.37 : 0.33), h * 0.43);
  paintPip(g, card.s, w * 0.63, h * 0.71, w * 0.52);
}

export function paintCardBack(g, w, h) {
  g.clearRect(0, 0, w, h);
  g.fillStyle = STOCK;
  roundRect(g, 0, 0, w, h, w * 0.085);
  g.fill();
  const m = w * 0.06;
  roundRect(g, m, m, w - 2 * m, h - 2 * m, w * 0.05);
  const fill = g.createLinearGradient(0, 0, w, h);
  fill.addColorStop(0, "#16324a");
  fill.addColorStop(1, "#0c1d2e");
  g.fillStyle = fill;
  g.fill();
  g.save();
  g.clip();
  g.strokeStyle = "rgba(216,179,106,0.42)";
  g.lineWidth = w * 0.008;
  const step = w * 0.085;
  for (let k = -h; k < w + h; k += step) {
    g.beginPath();
    g.moveTo(k, 0);
    g.lineTo(k + h, h);
    g.stroke();
    g.beginPath();
    g.moveTo(k + h, 0);
    g.lineTo(k, h);
    g.stroke();
  }
  g.restore();
  g.strokeStyle = BRASS;
  g.lineWidth = w * 0.012;
  roundRect(g, m * 1.6, m * 1.6, w - 3.2 * m, h - 3.2 * m, w * 0.035);
  g.stroke();
  g.fillStyle = "#0c1d2e";
  g.beginPath();
  g.arc(w / 2, h / 2, w * 0.16, 0, Math.PI * 2);
  g.fill();
  g.stroke();
  g.fillStyle = BRASS;
  paintPip(g, 0, w / 2, h / 2 - w * 0.01, w * 0.18);
}

// Small fixed-seed generator so the felt grain (and so every screenshot) is identical from run to run.
function grain(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A racetrack with half-extents hx (across) and hz (down the canvas); the longer one carries the straight sides.
function racetrackPath(g, cx, cy, hx, hz) {
  const r = Math.min(hx, hz),
    L = Math.max(hx, hz) - r;
  g.beginPath();
  if (hx >= hz) {
    g.moveTo(cx - L, cy - r);
    g.lineTo(cx + L, cy - r);
    g.arc(cx + L, cy, r, -Math.PI / 2, Math.PI / 2);
    g.lineTo(cx - L, cy + r);
    g.arc(cx - L, cy, r, Math.PI / 2, (3 * Math.PI) / 2);
  } else {
    g.moveTo(cx + r, cy - L);
    g.lineTo(cx + r, cy + L);
    g.arc(cx, cy + L, r, 0, Math.PI);
    g.lineTo(cx - r, cy - L);
    g.arc(cx, cy - L, r, Math.PI, 2 * Math.PI);
  }
  g.closePath();
}

// The felt texture covers the whole felt: x in [-hx, hx] across, z in [-hz, hz] down. Portrait tables are tall.
export function paintFelt(g, w, h, portrait = false) {
  const hx = portrait ? TABLE.b : TABLE.a,
    hz = portrait ? TABLE.a : TABLE.b;
  const px = w / (2 * hx); // pixels per world unit
  const pool = g.createRadialGradient(w / 2, h * 0.52, Math.min(w, h) * 0.05, w / 2, h / 2, Math.max(w, h) * 0.56);
  pool.addColorStop(0, "#1b6a4d");
  pool.addColorStop(0.55, "#0f4634");
  pool.addColorStop(1, "#08261c");
  g.fillStyle = pool;
  g.fillRect(0, 0, w, h);
  const rnd = grain(7);
  for (let i = 0; i < (w * h) / 22; i++) {
    g.fillStyle = rnd() < 0.5 ? "rgba(0,0,0,0.085)" : "rgba(255,255,255,0.04)";
    g.fillRect(rnd() * w, rnd() * h, 1.6, 1.6);
  }
  // The betting line: chips pushed across it are committed.
  const line = portrait ? 2.75 : 3.15;
  g.strokeStyle = "rgba(216,179,106,0.55)";
  g.lineWidth = 0.07 * px;
  racetrackPath(g, w / 2, h / 2, (hx - line) * px, (hz - line) * px);
  g.stroke();
  g.strokeStyle = "rgba(216,179,106,0.16)";
  g.lineWidth = 0.03 * px;
  racetrackPath(g, w / 2, h / 2, (hx - 0.45) * px, (hz - 0.45) * px);
  g.stroke();

  // Lettering: widely spaced capitals, faint enough to sit under the cards without fighting them.
  const spaced = (text, z, size, alpha) => {
    g.font = `600 ${size * px}px Georgia, "Times New Roman", serif`;
    g.fillStyle = `rgba(216,179,106,${alpha})`;
    g.textAlign = "center";
    g.textBaseline = "middle";
    const gap = size * px * 0.62;
    const widths = [...text].map((ch) => g.measureText(ch).width + gap);
    let x = w / 2 - (widths.reduce((a, c) => a + c, 0) - gap) / 2;
    [...text].forEach((ch, i) => {
      g.fillText(ch, x + (widths[i] - gap) / 2, h / 2 + z * px);
      x += widths[i];
    });
  };
  if (portrait) {
    spaced("HOLD\u2019EM", 3.1, 0.46, 0.3);
    spaced("TRAINER", 4.2, 0.3, 0.24);
  } else {
    spaced("NO LIMIT HOLD\u2019EM", -2.55, 0.5, 0.3);
    spaced("TRAINER", 1.75, 0.34, 0.24);
  }
}

export function paintChipTop(g, size, denom) {
  const c = size / 2;
  g.clearRect(0, 0, size, size);
  g.fillStyle = denom.color;
  g.beginPath();
  g.arc(c, c, c, 0, Math.PI * 2);
  g.fill();
  g.strokeStyle = denom.edge;
  g.lineWidth = size * 0.11;
  g.setLineDash([size * 0.2, size * 0.2112]);
  g.beginPath();
  g.arc(c, c, c - size * 0.06, 0, Math.PI * 2);
  g.stroke();
  g.setLineDash([]);
  g.lineWidth = size * 0.018;
  g.beginPath();
  g.arc(c, c, c * 0.62, 0, Math.PI * 2);
  g.stroke();
  g.fillStyle = "rgba(255,255,255,0.06)";
  g.beginPath();
  g.arc(c, c, c * 0.6, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = denom.edge;
  g.textAlign = "center";
  g.textBaseline = "middle";
  const label = denom.value >= 1000 ? denom.value / 1000 + "K" : String(denom.value);
  g.font = `700 ${size * (label.length > 2 ? 0.27 : 0.36)}px Georgia, "Times New Roman", serif`;
  g.fillText(label, c, c + size * 0.02);
}

export function paintChipEdge(g, w, h, denom) {
  g.fillStyle = denom.color;
  g.fillRect(0, 0, w, h);
  g.fillStyle = denom.edge;
  for (let i = 0; i < 6; i++) g.fillRect((i + 0.25) * (w / 6), 0, w / 12, h);
  g.fillStyle = "rgba(0,0,0,0.25)";
  g.fillRect(0, 0, w, h * 0.12);
  g.fillRect(0, h * 0.88, w, h * 0.12);
}

export function paintButton(g, size) {
  const c = size / 2;
  g.clearRect(0, 0, size, size);
  g.fillStyle = "#f4efe4";
  g.beginPath();
  g.arc(c, c, c, 0, Math.PI * 2);
  g.fill();
  g.strokeStyle = BRASS;
  g.lineWidth = size * 0.05;
  g.beginPath();
  g.arc(c, c, c * 0.84, 0, Math.PI * 2);
  g.stroke();
  g.fillStyle = "#17171a";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.font = `700 ${size * 0.2}px Georgia, "Times New Roman", serif`;
  g.fillText("DEALER", c, c + size * 0.01);
}

function canvas(w, h, draw) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d"), w, h);
  return c;
}

// A card as a canvas element, for the DOM panels and lessons.
export const cardCanvas = (card, opts, w = 200) =>
  canvas(w, Math.round(w * 1.4), (g, cw, ch) => (card ? paintCardFace(g, cw, ch, card, opts) : paintCardBack(g, cw, ch)));

// GPU textures for the stage. Card faces are painted on first use and kept.
export function makeTextures(renderer, { fourColour = false } = {}) {
  const made = [];
  const tex = (w, h, draw) => {
    const t = new THREE.CanvasTexture(canvas(w, h, draw));
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    made.push(t);
    return t;
  };
  const faces = new Map();
  const felts = [];
  const chips = DENOMS.map((d) => {
    const edge = tex(256, 32, (g, w, h) => paintChipEdge(g, w, h, d));
    edge.wrapS = THREE.RepeatWrapping;
    return { top: tex(256, 256, (g, w) => paintChipTop(g, w, d)), edge };
  });
  const api = {
    fourColour,
    face(card) {
      const key = card.r * 4 + card.s;
      if (!faces.has(key))
        faces.set(
          key,
          tex(512, 716, (g, w, h) => paintCardFace(g, w, h, card, { fourColour: api.fourColour }))
        );
      return faces.get(key);
    },
    // Repaint the faces in place when the deck style changes, so meshes keep their textures.
    setFourColour(on) {
      api.fourColour = on;
      for (const [key, t] of faces) {
        paintCardFace(t.image.getContext("2d"), 512, 716, { r: key >> 2, s: key & 3 }, { fourColour: on });
        t.needsUpdate = true;
      }
    },
    back: tex(512, 716, paintCardBack),
    felt: (portrait) =>
      (felts[+portrait] ??= portrait
        ? tex(1182, 2048, (g, w, h) => paintFelt(g, w, h, true))
        : tex(2048, 1182, (g, w, h) => paintFelt(g, w, h, false))),
    chipTop: (i) => chips[i].top,
    chipEdge: (i) => chips[i].edge,
    button: tex(256, 256, (g, w) => paintButton(g, w)),
    dispose: () => made.forEach((t) => t.dispose()),
  };
  return api;
}
