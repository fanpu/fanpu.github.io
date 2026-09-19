import * as THREE from "../../vendor/three.module.min.js";
import { CHIP, DENOMS, chipBreakdown } from "./layout.js";
import { ease } from "./tween.js";

// Chips. A "stack" is an amount of money at a place, addressed by a key ("stack3", "bet3", "pot").
// All chips of one denomination are a single InstancedMesh, so nine players' stacks cost seven draw calls.
const COLUMN = 10; // chips per column before starting another
const SPOTS = [
  [0, 0],
  [0.84, 0.12],
  [-0.46, 0.74],
  [0.42, 0.86],
  [-0.9, -0.08],
  [1.28, 0.78],
  [-0.06, 1.5],
  [0.8, 1.6],
  [-1.34, 0.7],
];
const CAPACITY = 220;

function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 16777619);
  return h >>> 0;
}

export function createChips(stage) {
  const geo = new THREE.CylinderGeometry(CHIP.r, CHIP.r, CHIP.t, 28);
  const meshes = DENOMS.map((d, i) => {
    const edge = stage.textures.chipEdge(i);
    const m = new THREE.InstancedMesh(
      geo,
      [
        new THREE.MeshStandardMaterial({ map: edge, roughness: 0.55 }),
        new THREE.MeshStandardMaterial({ map: stage.textures.chipTop(i), roughness: 0.5 }),
        new THREE.MeshStandardMaterial({ color: d.color, roughness: 0.6 }),
      ],
      CAPACITY
    );
    m.count = 0;
    m.castShadow = m.receiveShadow = true;
    m.frustumCulled = false;
    stage.scene.add(m);
    return m;
  });
  const stacks = new Map(); // key -> { amount, x, z, lift }
  const dummy = new THREE.Object3D();

  // Rewrite every instance matrix from the stacks. Cheap (a few hundred chips), and it keeps the truth in one place.
  function rebuild() {
    const counts = DENOMS.map(() => 0);
    for (const [key, s] of stacks) {
      if (s.amount <= 0) continue;
      let seed = hash(key);
      let n = 0;
      const parts = chipBreakdown(s.amount);
      // Centre the cluster of columns on the stack's spot, however many columns it takes.
      const columns = Math.min(SPOTS.length, Math.ceil(parts.reduce((t, q) => t + q.count, 0) / COLUMN));
      const cx = SPOTS.slice(0, columns).reduce((t, q) => t + q[0], 0) / columns,
        cz = SPOTS.slice(0, columns).reduce((t, q) => t + q[1], 0) / columns;
      for (const part of parts) {
        const di = DENOMS.findIndex((d) => d.value === part.value);
        for (let c = 0; c < part.count; c++, n++) {
          const col = Math.floor(n / COLUMN) % SPOTS.length,
            level = n % COLUMN;
          seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
          const jitter = ((seed >>> 8) / 16777216 - 0.5) * 0.05;
          dummy.position.set(
            s.x + SPOTS[col][0] - cx + jitter,
            (s.lift || 0) + CHIP.t / 2 + 0.004 + level * (CHIP.t + 0.003),
            s.z + SPOTS[col][1] - cz - jitter
          );
          dummy.rotation.set(0, ((seed >>> 4) % 628) / 100, 0);
          dummy.updateMatrix();
          if (counts[di] < CAPACITY) meshes[di].setMatrixAt(counts[di]++, dummy.matrix);
        }
      }
    }
    meshes.forEach((m, i) => {
      m.count = counts[i];
      m.instanceMatrix.needsUpdate = true;
    });
  }

  function slide(key, at, seconds = 0.42) {
    const s = stacks.get(key);
    if (!s) return Promise.resolve();
    const from = { x: s.x, z: s.z };
    return stage.anim.tween(
      seconds,
      (k) => {
        s.x = from.x + (at.x - from.x) * k;
        s.z = from.z + (at.z - from.z) * k;
        s.lift = Math.sin(k * Math.PI) * 0.18;
        rebuild();
      },
      ease.inOut
    );
  }

  let temp = 0;
  const api = {
    amount: (key) => stacks.get(key)?.amount || 0,
    setStack(key, amount, at) {
      if (amount <= 0) stacks.delete(key);
      else stacks.set(key, { amount, x: at.x, z: at.z, lift: 0 });
      rebuild();
    },
    // Take `amount` out of one stack and slide it onto another (created at `at` if new): a bet, a call, a refund.
    async move(fromKey, toKey, amount, at, seconds) {
      const from = stacks.get(fromKey);
      if (!from || amount <= 0) return;
      amount = Math.min(amount, from.amount);
      const moving = "~" + temp++;
      from.amount -= amount;
      if (from.amount <= 0) stacks.delete(fromKey);
      stacks.set(moving, { amount, x: from.x, z: from.z, lift: 0 });
      await slide(moving, stacks.get(toKey) || at, seconds);
      stacks.delete(moving);
      const to = stacks.get(toKey);
      if (to) to.amount += amount;
      else stacks.set(toKey, { amount, x: at.x, z: at.z, lift: 0 });
      rebuild();
    },
    // Sweep several stacks into one: the bets going into the pot.
    async merge(fromKeys, toKey, at, seconds = 0.5) {
      const live = fromKeys.filter((k) => stacks.has(k));
      if (!live.length) return;
      let sum = 0;
      await Promise.all(live.map((k) => ((sum += stacks.get(k).amount), slide(k, at, seconds))));
      live.forEach((k) => stacks.delete(k));
      const to = stacks.get(toKey);
      if (to) to.amount += sum;
      else stacks.set(toKey, { amount: sum, x: at.x, z: at.z, lift: 0 });
      rebuild();
    },
    clear() {
      stacks.clear();
      rebuild();
    },
  };
  return api;
}
