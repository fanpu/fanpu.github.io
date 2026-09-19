import * as THREE from "../../vendor/three.module.min.js";

// Light touches: a burst of brass sparks when a pot is won, and a soft pool of light on whoever is to act.
export function createFx(stage) {
  const MAX = 160;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(MAX * 3);
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const sparks = new THREE.Points(
    geo,
    new THREE.PointsMaterial({ color: 0xffd98a, size: 0.2, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending })
  );
  sparks.frustumCulled = false;
  stage.scene.add(sparks);
  let vel = null,
    life = 0;

  // A soft-edged pool: a radial gradient, added to the felt's own colour.
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.45, "rgba(255,255,255,0.5)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const spot = new THREE.Mesh(
    new THREE.PlaneGeometry(8.5, 8.5).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({
      map: new THREE.CanvasTexture(c),
      color: 0xffe2a8,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
  spot.position.y = 0.006;
  stage.scene.add(spot);
  let spotTarget = 0,
    clock = 0;

  stage.onFrame((dt) => {
    clock += dt;
    const o = spot.material.opacity;
    spot.material.opacity = o + (spotTarget * (0.2 + 0.05 * Math.sin(clock / 420)) - o) * Math.min(1, dt / 160);
    if (life <= 0) return;
    life -= dt;
    const s = dt / 1000;
    for (let i = 0; i < MAX; i++) {
      vel[i * 3 + 1] -= 9 * s;
      for (let a = 0; a < 3; a++) pos[i * 3 + a] += vel[i * 3 + a] * s;
      if (pos[i * 3 + 1] < 0.05) {
        pos[i * 3 + 1] = 0.05;
        vel[i * 3 + 1] *= -0.3;
      }
    }
    geo.attributes.position.needsUpdate = true;
    sparks.material.opacity = Math.max(0, Math.min(1, life / 500));
  });

  return {
    burst(at) {
      if (stage.anim.reduced) return;
      vel = new Float32Array(MAX * 3);
      for (let i = 0; i < MAX; i++) {
        const a = Math.random() * Math.PI * 2,
          r = 1.5 + Math.random() * 3.5;
        pos.set([at.x, 0.3, at.z], i * 3);
        vel.set([Math.cos(a) * r, 4 + Math.random() * 5, Math.sin(a) * r], i * 3);
      }
      life = 1300;
    },
    // at: { x, z } to light, or null for nobody.
    spotlight(at) {
      spotTarget = at ? 1 : 0;
      if (at) spot.position.set(at.x, 0.006, at.z);
    },
  };
}
