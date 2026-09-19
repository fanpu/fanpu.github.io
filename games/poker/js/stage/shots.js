import * as THREE from "../../lib/three.module.min.js";
import { ease } from "./tween.js";

// Camera shots, and the rig that moves between them. A shot is a direction to look from, a field of view and
// a zoom; the distance is not fixed but solved, so that the whole table (rail, labels and all) fits whatever
// rectangle the stage has been given. The table can therefore share the screen with panels of any size: the
// top half of a phone, or a desktop window with a drawer open.
// `wide` is for landscape regions, `tall` for portrait ones (where the table itself is turned a quarter).
export const SHOTS = {
  table: { wide: { from: [0, 21.5, 24.5], fov: 35, zoom: 1 }, tall: { from: [0, 41, 20], fov: 46, zoom: 1 } },
  deal: { wide: { from: [0, 24, 22], fov: 35, zoom: 1 }, tall: { from: [0, 43, 18], fov: 46, zoom: 1 } },
  board: { wide: { from: [0, 21, 23], fov: 35, zoom: 0.93 }, tall: { from: [0, 41, 19], fov: 46, zoom: 0.96 } },
  showdown: { wide: { from: [0, 22.5, 23], fov: 35, zoom: 0.96 }, tall: { from: [0, 42, 19], fov: 46, zoom: 0.98 } },
  top: { wide: { from: [0, 44, 0.5], fov: 38, zoom: 1 }, tall: { from: [0, 62, 0.5], fov: 46, zoom: 1 } },
  lesson: { wide: { from: [-3, 21, 25], fov: 35, zoom: 1 }, tall: { from: [0, 41, 20], fov: 46, zoom: 1 } },
  idle: { wide: { from: [12, 17, 26], fov: 35, zoom: 1.05 }, tall: { from: [8, 40, 22], fov: 46, zoom: 1.05 } },
};

// Room to leave round the table, in CSS pixels: seat labels hang outside the rail, and action bubbles above them.
const PAD = { wide: { x: 64, top: 62, bottom: 30 }, tall: { x: 10, top: 66, bottom: 30 } };

export function createCameraRig(stage, table) {
  const cam = stage.camera;
  const probe = new THREE.PerspectiveCamera();
  let current = "table";
  // Orbit offsets the user adds by dragging; they ride on top of whichever shot is current.
  const orbit = { yaw: 0, pitch: 0, zoom: 1 };
  const base = { pos: new THREE.Vector3(), target: new THREE.Vector3(), fov: 35 };
  const v = new THREE.Vector3();

  // Where must the camera be for this shot, so that everything fits the stage's rectangle?
  function solve(name) {
    const tall = stage.portrait,
      shot = SHOTS[name][tall ? "tall" : "wide"],
      pad = PAD[tall ? "tall" : "wide"];
    const w = stage.container.clientWidth || 1,
      h = stage.container.clientHeight || 1;
    const dir = new THREE.Vector3().fromArray(shot.from).normalize();
    const pts = [...table.layout.outline(40).map((p) => [p.x, 0.8, p.z]), ...table.seats.map((s) => [s.label.x, 0.9, s.label.z])];
    probe.fov = shot.fov;
    probe.aspect = w / h;
    probe.updateProjectionMatrix();

    // For a look-at point (0, 0, tz) and distance d: does everything land inside the padded rectangle? Also reports how far off-centre it sits.
    const measure = (tz, d) => {
      probe.position.set(dir.x * d, dir.y * d, tz + dir.z * d);
      probe.lookAt(0, 0, tz);
      probe.updateMatrixWorld();
      let ok = true,
        lo = Infinity,
        hi = -Infinity;
      for (const p of pts) {
        v.set(p[0], p[1], p[2]).project(probe);
        const x = ((v.x + 1) / 2) * w,
          y = ((1 - v.y) / 2) * h;
        if (v.z > 1 || x < pad.x || x > w - pad.x || y < pad.top || y > h - pad.bottom) ok = false;
        lo = Math.min(lo, y);
        hi = Math.max(hi, y);
      }
      return { ok, offCentre: Math.abs((lo + hi) / 2 - (pad.top + h - pad.bottom) / 2) };
    };
    const nearest = (tz) => {
      let lo = 8,
        hi = 600;
      for (let i = 0; i < 22; i++) {
        const mid = (lo + hi) / 2;
        measure(tz, mid).ok ? (hi = mid) : (lo = mid);
      }
      return hi;
    };
    // Slide the look-at point along the table's depth. When height is the limit, the closest fit is already
    // centred; when width is the limit, many look-at points fit about equally well, and the centred one wins.
    let best = null;
    for (let tz = -14; tz <= 14; tz += 0.5) {
      const d = nearest(tz),
        score = d + measure(tz, d).offCentre * 0.12;
      if (!best || score < best.score) best = { tz, d, score };
    }
    const d = best.d * shot.zoom;
    return { pos: new THREE.Vector3(dir.x * d, dir.y * d, best.tz + dir.z * d), target: new THREE.Vector3(0, 0, best.tz), fov: shot.fov };
  }

  function place() {
    const off = base.pos.clone().sub(base.target);
    const sph = new THREE.Spherical().setFromVector3(off);
    sph.theta += orbit.yaw;
    sph.phi = Math.max(0.12, Math.min(1.45, sph.phi + orbit.pitch));
    sph.radius *= orbit.zoom;
    cam.position.copy(new THREE.Vector3().setFromSpherical(sph).add(base.target));
    cam.fov = base.fov;
    cam.updateProjectionMatrix();
    cam.lookAt(base.target);
  }
  function snap(name = current) {
    current = name;
    const s = solve(name);
    base.pos.copy(s.pos);
    base.target.copy(s.target);
    base.fov = s.fov;
    place();
  }
  function to(name, seconds = 0.9) {
    current = name;
    const s = solve(name);
    const from = { pos: base.pos.clone(), target: base.target.clone(), fov: base.fov };
    return stage.anim.tween(
      seconds,
      (k) => {
        base.pos.lerpVectors(from.pos, s.pos, k);
        base.target.lerpVectors(from.target, s.target, k);
        base.fov = from.fov + (s.fov - from.fov) * k;
        place();
      },
      ease.inOut
    );
  }
  function enableOrbit(dom) {
    let drag = null;
    dom.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, yaw: orbit.yaw, pitch: orbit.pitch };
      dom.setPointerCapture(e.pointerId);
    });
    dom.addEventListener("pointermove", (e) => {
      if (!drag) return;
      orbit.yaw = Math.max(-1.1, Math.min(1.1, drag.yaw - (e.clientX - drag.x) * 0.005));
      orbit.pitch = Math.max(-0.5, Math.min(0.5, drag.pitch - (e.clientY - drag.y) * 0.004));
      place();
    });
    const end = () => (drag = null);
    dom.addEventListener("pointerup", end);
    dom.addEventListener("pointercancel", end);
    dom.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        orbit.zoom = Math.max(0.6, Math.min(1.5, orbit.zoom * (1 + Math.sign(e.deltaY) * 0.06)));
        place();
      },
      { passive: false }
    );
    dom.addEventListener("dblclick", () => {
      Object.assign(orbit, { yaw: 0, pitch: 0, zoom: 1 });
      place();
    });
  }
  return {
    to,
    snap,
    refit: () => snap(current), // call after the stage's rectangle, the table's orientation or the number of seats changes
    enableOrbit,
    get current() {
      return current;
    },
  };
}
