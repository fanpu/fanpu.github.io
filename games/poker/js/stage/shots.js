import * as THREE from "../../vendor/three.module.min.js";
import { ease } from "./tween.js";

// Named camera positions, and the rig that moves between them. `wide` variants are for landscape
// screens; `tall` for portrait, where the camera climbs and pulls back so that every seat still fits.
export const SHOTS = {
  table: { wide: { pos: [0, 21.5, 26.5], target: [0, 0, 2.0], fov: 35 }, tall: { pos: [0, 41, 23.5], target: [0, 0, 3.4], fov: 50 } },
  deal: { wide: { pos: [0, 24, 27], target: [0, 0, 1.0], fov: 35 }, tall: { pos: [0, 43, 22], target: [0, 0, 3.2], fov: 50 } },
  board: { wide: { pos: [0, 19, 24], target: [0, 0, 0.2], fov: 33 }, tall: { pos: [0, 38, 20], target: [0, 0, 2.4], fov: 48 } },
  showdown: { wide: { pos: [0, 20.5, 25.5], target: [0, 0, 0.6], fov: 34 }, tall: { pos: [0, 39, 21.5], target: [0, 0, 2.8], fov: 49 } },
  top: { wide: { pos: [0, 44, 0.01], target: [0, 0, 0], fov: 38 }, tall: { pos: [0, 62, 0.01], target: [0, 0, 0], fov: 50 } },
  lesson: { wide: { pos: [-5.5, 20, 27], target: [-3.2, 0, 0.8], fov: 35 }, tall: { pos: [0, 41, 23.5], target: [0, 0, 3.4], fov: 50 } },
  idle: { wide: { pos: [14, 17, 30], target: [0, 0, 0], fov: 35 }, tall: { pos: [10, 46, 30], target: [0, 0, 1], fov: 50 } },
};

export function createCameraRig(stage) {
  const cam = stage.camera;
  const target = new THREE.Vector3();
  let current = "table";
  // Orbit offsets the user adds by dragging; they ride on top of whichever shot is current.
  const orbit = { yaw: 0, pitch: 0, zoom: 1 };
  const base = { pos: new THREE.Vector3(), target: new THREE.Vector3(), fov: 35 };

  const pick = (name) => SHOTS[name][stage.portrait ? "tall" : "wide"];
  function place() {
    const off = base.pos.clone().sub(base.target);
    const sph = new THREE.Spherical().setFromVector3(off);
    sph.theta += orbit.yaw;
    sph.phi = Math.max(0.12, Math.min(1.45, sph.phi + orbit.pitch));
    sph.radius *= orbit.zoom;
    cam.position.copy(new THREE.Vector3().setFromSpherical(sph).add(base.target));
    target.copy(base.target);
    cam.fov = base.fov;
    cam.updateProjectionMatrix();
    cam.lookAt(target);
  }
  function snap(name = current) {
    current = name;
    const s = pick(name);
    base.pos.fromArray(s.pos);
    base.target.fromArray(s.target);
    base.fov = s.fov;
    place();
  }
  function to(name, seconds = 0.9) {
    current = name;
    const s = pick(name);
    const from = { pos: base.pos.clone(), target: base.target.clone(), fov: base.fov };
    const toPos = new THREE.Vector3().fromArray(s.pos),
      toTarget = new THREE.Vector3().fromArray(s.target);
    return stage.anim.tween(
      seconds,
      (k) => {
        base.pos.lerpVectors(from.pos, toPos, k);
        base.target.lerpVectors(from.target, toTarget, k);
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
  stage.onResized(() => snap(current));
  snap("table");
  return {
    to,
    snap,
    enableOrbit,
    get current() {
      return current;
    },
  };
}
