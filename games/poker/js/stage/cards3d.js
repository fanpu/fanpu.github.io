import * as THREE from "../../vendor/three.module.min.js";
import { CARD, DECK, MUCK } from "./layout.js";
import { ease } from "./tween.js";

// Playing cards as objects on the table. A card is addressed by a string id chosen by the caller
// ("h3a" = seat 3's first hole card, "b0" = first board card) and sent to slots: { x, z, rot, y? }.
// Every motion returns a promise and ends exactly on its target, skipped or not.
function roundedRect(w, h, r) {
  const s = new THREE.Shape();
  const x = -w / 2,
    y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.absarc(x + w - r, y + r, r, -Math.PI / 2, 0, false);
  s.lineTo(x + w, y + h - r);
  s.absarc(x + w - r, y + h - r, r, 0, Math.PI / 2, false);
  s.lineTo(x + r, y + h);
  s.absarc(x + r, y + h - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(x, y + r);
  s.absarc(x + r, y + r, r, Math.PI, 1.5 * Math.PI, false);
  return s;
}

export function createCards(stage) {
  const { w, h, t } = CARD;
  const shape = roundedRect(w, h, w * 0.085);
  // Face and back are flat rounded sheets with their own textures; a thin extrusion between them is the card's edge.
  const sheet = new THREE.ShapeGeometry(shape, 6);
  const uv = sheet.attributes.uv,
    p = sheet.attributes.position;
  for (let i = 0; i < p.count; i++) uv.setXY(i, p.getX(i) / w + 0.5, p.getY(i) / h + 0.5);
  const faceGeo = sheet
    .clone()
    .rotateX(-Math.PI / 2)
    .translate(0, t / 2 + 0.001, 0);
  const backGeo = sheet
    .clone()
    .rotateX(Math.PI / 2)
    .translate(0, -t / 2 - 0.001, 0);
  const edgeGeo = new THREE.ExtrudeGeometry(shape, { depth: t, bevelEnabled: false, curveSegments: 6 }).rotateX(-Math.PI / 2).translate(0, -t / 2, 0);
  const glowGeo = new THREE.ShapeGeometry(roundedRect(w * 1.13, h * 1.095, w * 0.12), 6).rotateX(-Math.PI / 2);
  const edgeMat = new THREE.MeshStandardMaterial({ color: 0xf2efe6, roughness: 0.6 });
  const glowMat = new THREE.MeshBasicMaterial({ color: 0xffd98a, transparent: true, opacity: 0.95, depthWrite: false });

  const root = new THREE.Group();
  stage.scene.add(root);
  const cards = new Map();
  let spots = { deck: DECK, muck: MUCK }; // where cards come from and go to; changes with the table's orientation
  const REST = t / 2 + 0.012; // resting height of a card's centre above the felt

  function make(id) {
    const group = new THREE.Group(), // position and yaw
      inner = new THREE.Group(); // the flip, about the card's long axis
    const face = new THREE.Mesh(faceGeo, new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0 }));
    const back = new THREE.Mesh(backGeo, new THREE.MeshStandardMaterial({ map: stage.textures.back, roughness: 0.5 }));
    const edge = new THREE.Mesh(edgeGeo, edgeMat);
    const glow = new THREE.Mesh(glowGeo, glowMat);
    glow.position.y = -t / 2 - 0.004;
    glow.visible = false;
    edge.castShadow = true;
    face.receiveShadow = back.receiveShadow = true;
    inner.add(face, back, edge);
    group.add(inner, glow);
    root.add(group);
    const c = { id, group, inner, face, back, glow, faceUp: false, lifted: false, key: -1 };
    cards.set(id, c);
    return c;
  }
  const get = (id) => cards.get(id) || make(id);
  function setFace(c, card) {
    if (!card) return;
    c.key = card.r * 4 + card.s;
    c.face.material.map = stage.textures.face(card);
    c.face.material.needsUpdate = true;
  }
  const yOf = (slot, c) => (slot.y ?? 0) + REST + (c.lifted ? 0.42 : 0);
  function put(c, slot) {
    c.group.position.set(slot.x, yOf(slot, c), slot.z);
    c.group.rotation.y = slot.rot || 0;
    c.group.scale.setScalar(slot.scale || 1);
    c.slot = { ...slot };
  }

  // Move in an arc. `hop` is the height of the arc; spin adds a turn on the way, for dealt cards.
  function travel(c, slot, seconds, { hop = 0.5, spin = 0, easing = ease.out } = {}) {
    const from = c.group.position.clone(),
      fromRot = c.group.rotation.y;
    const to = new THREE.Vector3(slot.x, yOf(slot, c), slot.z);
    let dRot = (slot.rot || 0) - fromRot;
    dRot = Math.atan2(Math.sin(dRot), Math.cos(dRot)) + spin * Math.PI * 2; // the short way round
    c.slot = { ...slot };
    const s0 = c.group.scale.x,
      s1 = slot.scale || 1;
    return stage.anim.tween(
      seconds,
      (k) => {
        c.group.scale.setScalar(s0 + (s1 - s0) * k);
        c.group.position.lerpVectors(from, to, k);
        c.group.position.y += Math.sin(k * Math.PI) * hop;
        c.group.rotation.y = k === 1 ? slot.rot || 0 : fromRot + dRot * k;
      },
      easing
    );
  }

  const api = {
    setLayout: (layout) => (spots = { deck: layout.deck, muck: layout.muck }),
    has: (id) => cards.has(id),
    ids: () => [...cards.keys()],
    place(id, card, slot, { faceUp = false } = {}) {
      const c = get(id);
      setFace(c, card);
      c.faceUp = faceUp;
      c.inner.rotation.z = faceUp ? 0 : Math.PI;
      put(c, slot);
      return c;
    },
    // Deal from the deck (or any `from` point) to a slot, face down.
    async deal(id, card, slot, { from = spots.deck, delay = 0, seconds = 0.34 } = {}) {
      const c = api.place(id, card, { x: from.x, z: from.z, rot: slot.rot, y: 0.3 }, { faceUp: false });
      c.group.visible = false;
      if (delay) await stage.anim.wait(delay);
      c.group.visible = true;
      await travel(c, slot, seconds, { hop: 0.9, spin: 0.5 });
    },
    async flip(id, faceUp, card) {
      const c = cards.get(id);
      if (!c) return;
      setFace(c, card);
      if (c.faceUp === faceUp) return;
      c.faceUp = faceUp;
      const from = c.inner.rotation.z,
        to = faceUp ? 0 : Math.PI,
        y0 = c.group.position.y;
      await stage.anim.tween(
        0.36,
        (k) => {
          c.inner.rotation.z = from + (to - from) * k;
          c.group.position.y = y0 + Math.sin(k * Math.PI) * 0.75; // lift clear of the felt while turning
        },
        ease.inOut
      );
    },
    moveTo: (id, slot, seconds = 0.4) => (cards.has(id) ? travel(cards.get(id), slot, seconds) : Promise.resolve()),
    // Fold: turn face down if needed, slide to the muck and go.
    async muck(id) {
      const c = cards.get(id);
      if (!c) return;
      if (c.faceUp) await api.flip(id, false);
      await travel(
        c,
        { x: spots.muck.x + (Math.random() - 0.5) * 0.5, z: spots.muck.z + (Math.random() - 0.5) * 0.5, rot: c.group.rotation.y + 0.6 },
        0.42,
        {
          hop: 0.25,
          easing: ease.inOut,
        }
      );
      api.remove(id);
    },
    // Raise the cards that make the hand and rim them in brass.
    lift(ids, on = true) {
      return Promise.all(
        ids.map((id) => {
          const c = cards.get(id);
          if (!c || c.lifted === on) return null;
          c.lifted = on;
          c.glow.visible = on;
          const y0 = c.group.position.y,
            y1 = yOf(c.slot, c);
          return stage.anim.tween(0.32, (k) => (c.group.position.y = y0 + (y1 - y0) * k), ease.outBack);
        })
      );
    },
    dim(ids, on = true) {
      for (const id of ids) {
        const c = cards.get(id);
        if (!c) continue;
        const v = on ? 0.34 : 1;
        c.face.material.color.setScalar(v);
        c.back.material.color.setScalar(v);
      }
    },
    remove(id) {
      const c = cards.get(id);
      if (!c) return;
      root.remove(c.group);
      c.face.material.dispose();
      c.back.material.dispose();
      cards.delete(id);
    },
    clear() {
      for (const id of [...cards.keys()]) api.remove(id);
    },
  };
  return api;
}
