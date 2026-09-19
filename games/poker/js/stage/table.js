import * as THREE from "../../vendor/three.module.min.js";
import { TABLE, layoutFor } from "./layout.js";
import { ease } from "./tween.js";

// The furniture: felt, padded rail, brass trim, the table's body, the floor the lamp falls on, and the dealer button.
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

export function buildTable(stage) {
  const group = new THREE.Group();
  const { rail } = TABLE;
  let layout = null,
    furniture = null;

  // Felt, rail, trim and body depend on which way the table lies; they are rebuilt when the screen turns.
  function setOrientation(portrait) {
    if (layout?.portrait === portrait) return false;
    layout = layoutFor(portrait);
    if (furniture) {
      group.remove(furniture);
      furniture.traverse((o) => o.geometry?.dispose());
    }
    furniture = new THREE.Group();
    const a = layout.halfX,
      b = layout.halfZ;

    const feltGeo = flat(new THREE.ShapeGeometry(racetrack(a, b), 48));
    const pos = feltGeo.attributes.position,
      uv = feltGeo.attributes.uv;
    for (let i = 0; i < pos.count; i++) uv.setXY(i, (pos.getX(i) + a) / (2 * a), 1 - (pos.getZ(i) + b) / (2 * b));
    const felt = new THREE.Mesh(feltGeo, new THREE.MeshStandardMaterial({ map: stage.textures.felt(portrait), roughness: 0.96, metalness: 0 }));
    felt.receiveShadow = true;

    // Padded rail: a bevelled ring, dark leather with a soft sheen so the lamp draws a highlight along it.
    const railMesh = new THREE.Mesh(
      flat(
        new THREE.ExtrudeGeometry(ring(a + rail, b + rail, a + 0.12, b + 0.12), {
          depth: 0.34,
          bevelEnabled: true,
          bevelThickness: 0.3,
          bevelSize: 0.3,
          bevelSegments: 8,
          curveSegments: 48,
        })
      ),
      new THREE.MeshPhysicalMaterial({ color: 0x1c1411, roughness: 0.42, metalness: 0, clearcoat: 0.35, clearcoatRoughness: 0.5 })
    );
    railMesh.position.y = 0.02;
    railMesh.castShadow = railMesh.receiveShadow = true;

    const trim = new THREE.Mesh(
      flat(new THREE.ExtrudeGeometry(ring(a + 0.14, b + 0.14, a - 0.02, b - 0.02), { depth: 0.1, bevelEnabled: false, curveSegments: 48 })),
      new THREE.MeshStandardMaterial({ color: 0xd8b36a, roughness: 0.32, metalness: 0.85 })
    );

    const body = new THREE.Mesh(
      flat(new THREE.ExtrudeGeometry(racetrack(a + rail + 0.25, b + rail + 0.25), { depth: 1.6, bevelEnabled: false, curveSegments: 48 })),
      new THREE.MeshStandardMaterial({ color: 0x120c09, roughness: 0.7 })
    );
    body.position.y = -1.62;
    body.castShadow = body.receiveShadow = true; // without this the lamp shines straight through onto the floor

    furniture.add(felt, railMesh, trim, body);
    group.add(furniture);
    if (seats.length) seats = layout.seats(seats.length);
    return true;
  }

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(120, 64).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x0d0e10, roughness: 0.92 })
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
    setSeats(n) {
      seats = layout.seats(n);
      return seats;
    },
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
