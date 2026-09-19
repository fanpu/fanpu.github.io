import * as THREE from "../../vendor/three.module.min.js";
import { TABLE, layoutFor } from "./layout.js";
import { ease } from "./tween.js";

// The furniture: felt, a lacquered walnut racetrack with brass inlay and cup holders, a stitched leather rail,
// the table's body, the floor the lamp falls on, and the dealer button.
// hx, hz are half-extents along x and z. Shapes are drawn in xy and then laid flat, which maps shape y to -z;
// the racetrack is symmetric, so that flip does not matter.
function racetrack(hx, hz) {
  const r = Math.min(hx, hz),
    L = Math.max(hx, hz) - r,
    s = new THREE.Shape();
  if (hx >= hz) {
    s.moveTo(-L, -r);
    s.lineTo(L, -r);
    s.absarc(L, 0, r, -Math.PI / 2, Math.PI / 2, false);
    s.lineTo(-L, r);
    s.absarc(-L, 0, r, Math.PI / 2, (3 * Math.PI) / 2, false);
  } else {
    s.moveTo(r, -L);
    s.lineTo(r, L);
    s.absarc(0, L, r, 0, Math.PI, false);
    s.lineTo(-r, -L);
    s.absarc(0, -L, r, Math.PI, 2 * Math.PI, false);
  }
  return s;
}
function ring(outerLen, outerWid, innerLen, innerWid) {
  const s = racetrack(outerLen, outerWid);
  s.holes.push(racetrack(innerLen, innerWid));
  return s;
}
const flat = (geometry) => geometry.rotateX(-Math.PI / 2); // shapes are drawn in xy; the table lies in xz

// Evenly spaced points along a racetrack outline, with the direction of travel at each: for stitching.
function along(hx, hz, spacing) {
  const pts = racetrack(hx, hz).getSpacedPoints(Math.round(racetrack(hx, hz).getLength() / spacing));
  return pts.slice(0, -1).map((p, i) => {
    const q = pts[i + 1];
    return { x: p.x, z: -p.y, angle: Math.atan2(q.y - p.y, q.x - p.x) };
  });
}
// Planar UVs from world position, so a veneer's grain runs straight across the whole table as if cut from one sheet.
function planarUV(geometry, scaleX, scaleZ) {
  const pos = geometry.attributes.position,
    uv = geometry.attributes.uv;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) / scaleX, pos.getZ(i) / scaleZ);
  return geometry;
}

export function buildTable(stage) {
  const group = new THREE.Group();
  let layout = null,
    furniture = null,
    cups = null;

  const leather = new THREE.MeshPhysicalMaterial({
    color: 0x17100d,
    roughness: 0.5,
    metalness: 0,
    bumpMap: stage.textures.leatherBump,
    bumpScale: 0.55,
    clearcoat: 0.25,
    clearcoatRoughness: 0.55,
    sheen: 0.4,
    sheenRoughness: 0.5,
    sheenColor: 0x6b4a36,
  });
  leather.bumpMap.repeat.set(1, 1);
  const walnut = new THREE.MeshPhysicalMaterial({ map: stage.textures.wood, roughness: 0.3, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.07 }); // piano lacquer
  const brass = new THREE.MeshStandardMaterial({ color: 0xd8b36a, roughness: 0.2, metalness: 1 });
  const thread = new THREE.MeshStandardMaterial({ color: 0xcaa874, roughness: 0.8 });

  // Felt, racetrack, rail and body depend on which way the table lies; they are rebuilt when the screen turns.
  function setOrientation(portrait) {
    if (layout?.portrait === portrait) return false;
    layout = layoutFor(portrait);
    if (furniture) {
      group.remove(furniture);
      furniture.traverse((o) => o.geometry?.dispose());
    }
    furniture = new THREE.Group();
    const a = layout.halfX,
      b = layout.halfZ,
      { track, rail } = layout;

    const feltGeo = flat(new THREE.ShapeGeometry(racetrack(a, b), 48));
    const pos = feltGeo.attributes.position,
      uv = feltGeo.attributes.uv;
    for (let i = 0; i < pos.count; i++) uv.setXY(i, (pos.getX(i) + a) / (2 * a), 1 - (pos.getZ(i) + b) / (2 * b));
    const felt = new THREE.Mesh(
      feltGeo,
      new THREE.MeshStandardMaterial({ map: stage.textures.felt(portrait), roughness: 0.96, metalness: 0, envMapIntensity: 0.25 })
    );
    felt.receiveShadow = true;

    // The racetrack: lacquered walnut between the felt and the rail, stood a little proud of the felt,
    // with a brass inlay line on each edge.
    const wood = new THREE.Mesh(
      planarUV(
        flat(
          new THREE.ExtrudeGeometry(ring(a + track, b + track, a, b), {
            depth: 0.09,
            bevelEnabled: true,
            bevelThickness: 0.025,
            bevelSize: 0.025,
            bevelSegments: 2,
            curveSegments: 48,
          })
        ),
        13,
        6.5
      ),
      walnut
    );
    wood.position.y = 0.025;
    wood.receiveShadow = true;
    const inlay = (inner) => {
      const m = new THREE.Mesh(
        flat(
          new THREE.ExtrudeGeometry(ring(a + inner + 0.075, b + inner + 0.075, a + inner, b + inner), {
            depth: 0.02,
            bevelEnabled: false,
            curveSegments: 48,
          })
        ),
        brass
      );
      m.position.y = 0.128;
      return m;
    };

    // The rail: a cushion. A thin slab with a deep round bevel gives the pillowed profile of upholstered leather.
    const pad = 0.46; // how far the cushion's shoulder rolls over
    const r0 = track + 0.04,
      r1 = track + rail;
    const cushion = new THREE.Mesh(
      planarUV(
        flat(
          new THREE.ExtrudeGeometry(ring(a + r1 - pad, b + r1 - pad, a + r0 + pad, b + r0 + pad), {
            depth: 0.06,
            bevelEnabled: true,
            bevelThickness: 0.44,
            bevelSize: pad,
            bevelSegments: 12,
            curveSegments: 48,
          })
        ),
        2.2,
        2.2
      ),
      leather
    );
    cushion.position.y = 0.3;
    cushion.castShadow = cushion.receiveShadow = true;
    const top = cushion.position.y + 0.06 + 0.44; // height of the cushion's crown

    // Contrast stitching along both shoulders of the cushion.
    const stitchGeo = new THREE.BoxGeometry(0.17, 0.03, 0.04);
    const lines = [along(a + r0 + pad * 0.62, b + r0 + pad * 0.62, 0.3), along(a + r1 - pad * 0.62, b + r1 - pad * 0.62, 0.3)];
    const stitches = new THREE.InstancedMesh(stitchGeo, thread, lines[0].length + lines[1].length);
    const d = new THREE.Object3D();
    let k = 0;
    for (const line of lines)
      for (const p of line) {
        d.position.set(p.x, top - 0.035, p.z);
        d.rotation.set(0, p.angle, 0);
        d.updateMatrix();
        stitches.setMatrixAt(k++, d.matrix);
      }

    const body = new THREE.Mesh(
      flat(new THREE.ExtrudeGeometry(racetrack(a + r1 + 0.12, b + r1 + 0.12), { depth: 1.7, bevelEnabled: false, curveSegments: 48 })),
      new THREE.MeshStandardMaterial({ color: 0x120c09, roughness: 0.7 })
    );
    body.position.y = -1.72;
    body.castShadow = body.receiveShadow = true; // without this the lamp shines straight through onto the floor

    furniture.add(felt, wood, inlay(0.0), inlay(track - 0.075), cushion, stitches, body);
    group.add(furniture);
    if (seats.length) setSeats(seats.length);
    return true;
  }

  // A brass cup holder let into the wood beside each seat.
  function setSeats(n) {
    seats = layout.seats(n);
    if (cups) {
      group.remove(cups);
      cups.traverse((o) => o.geometry?.dispose());
    }
    cups = new THREE.Group();
    const r = Math.min(0.4, layout.track * 0.42);
    const rimGeo = new THREE.TorusGeometry(r, 0.055, 12, 40).rotateX(Math.PI / 2),
      wellGeo = new THREE.CircleGeometry(r, 40).rotateX(-Math.PI / 2);
    const well = new THREE.MeshStandardMaterial({ color: 0x0a0806, roughness: 0.35, metalness: 0.9 });
    for (const s of seats) {
      const rim = new THREE.Mesh(rimGeo, brass),
        hole = new THREE.Mesh(wellGeo, well);
      rim.position.set(s.cup.x, 0.15, s.cup.z);
      hole.position.set(s.cup.x, 0.142, s.cup.z);
      rim.castShadow = true;
      cups.add(rim, hole);
    }
    group.add(cups);
    return seats;
  }

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(120, 64).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x0b0c0e, roughness: 0.95, envMapIntensity: 0 }) // the room stays a dark void
  );
  floor.position.y = -7.4;
  floor.receiveShadow = true;
  group.add(floor);

  const button = new THREE.Mesh(new THREE.CylinderGeometry(0.52, 0.52, 0.12, 40), [
    new THREE.MeshStandardMaterial({ color: 0xe9e4d8, roughness: 0.4 }),
    new THREE.MeshStandardMaterial({ map: stage.textures.button, roughness: 0.4 }),
    new THREE.MeshStandardMaterial({ color: 0xe9e4d8, roughness: 0.4 }),
  ]);
  button.castShadow = true;
  button.visible = false;
  group.add(button);

  stage.scene.add(group);
  let seats = [];
  setOrientation(stage.portrait);
  return {
    group,
    get seats() {
      return seats;
    },
    get layout() {
      return layout;
    },
    setOrientation,
    setSeats,
    // Slide the dealer button to a seat (in an arc, so it clears chips on the way).
    moveButton(seat, animate = true) {
      const to = seats[seat].button;
      const from = { x: button.position.x, z: button.position.z };
      const first = !button.visible;
      button.visible = true;
      if (first || !animate) {
        button.position.set(to.x, 0.06, to.z);
        return Promise.resolve();
      }
      return stage.anim.tween(
        0.55,
        (k) => {
          button.position.set(from.x + (to.x - from.x) * k, 0.06 + Math.sin(k * Math.PI) * 0.5, from.z + (to.z - from.z) * k);
          button.rotation.y = k * Math.PI;
        },
        ease.inOut
      );
    },
  };
}
