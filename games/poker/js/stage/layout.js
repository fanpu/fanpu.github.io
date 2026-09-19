// Table geometry, as pure functions of the number of seats. No three.js here, so it is tested in node.
// Units: 1 = 10 cm. x runs along the table, the hero sits at +z (nearest the camera), y is up.
// Cards and chips are larger than life against the table, on purpose: they have to be read from across the room.
export const TABLE = { a: 13, b: 7.5, rail: 1.5 }; // felt half-length, felt half-width, rail width
export const CARD = { w: 1.26, h: 1.76, t: 0.02 };
export const CHIP = { r: 0.39, t: 0.066 };

const L = TABLE.a - TABLE.b; // half-length of the straight sides

// The felt is a racetrack: a rectangle with a semicircle on each end.
export function insideFelt(p, margin = 0) {
  return Math.hypot(Math.max(Math.abs(p.x) - L, 0), p.z) <= TABLE.b - margin;
}

// A point on the felt's edge. t in [0, 1) starts bottom centre and runs clockwise as seen from above
// (toward the hero's left), which is the direction the deal and the action travel.
// Returns the point, the outward normal and the direction of travel.
function edge(t) {
  const r = TABLE.b,
    P = 4 * L + 2 * Math.PI * r;
  let s = (((t % 1) + 1) % 1) * P;
  if (s < L) return { x: -s, z: r, nx: 0, nz: 1, tx: -1, tz: 0 };
  s -= L;
  if (s < Math.PI * r) {
    const th = Math.PI / 2 + s / r;
    return { x: -L + r * Math.cos(th), z: r * Math.sin(th), nx: Math.cos(th), nz: Math.sin(th), tx: -Math.sin(th), tz: Math.cos(th) };
  }
  s -= Math.PI * r;
  if (s < 2 * L) return { x: -L + s, z: -r, nx: 0, nz: -1, tx: 1, tz: 0 };
  s -= 2 * L;
  if (s < Math.PI * r) {
    const th = -Math.PI / 2 + s / r;
    return { x: L + r * Math.cos(th), z: r * Math.sin(th), nx: Math.cos(th), nz: Math.sin(th), tx: -Math.sin(th), tz: Math.cos(th) };
  }
  s -= Math.PI * r;
  return { x: L - s, z: r, nx: 0, nz: 1, tx: -1, tz: 0 };
}

// Everything that belongs to a seat, found by stepping in from the rail (inward) and along it (toward the next seat).
export function seatLayout(n) {
  const seats = [];
  for (let seat = 0; seat < n; seat++) {
    const t = seat / n,
      e = edge(t);
    const at = (inward, along = 0) => ({ x: e.x - e.nx * inward + e.tx * along, z: e.z - e.nz * inward + e.tz * along });
    const rot = Math.atan2(e.nx, e.nz); // turn about y so that a card's foot points at its player
    seats.push({
      seat,
      t,
      rot,
      pos: at(-TABLE.rail / 2),
      label: at(-TABLE.rail - 0.4),
      cards: [
        { ...at(2.1, 0.68), rot },
        { ...at(2.1, -0.68), rot },
      ],
      bet: at(3.9),
      stack: at(1.7, -2.3), // on the player's right
      button: at(3.0, 2.2), // on the player's left
    });
  }
  return seats;
}

export function boardSlots() {
  return [-2, -1, 0, 1, 2].map((i) => ({ x: i * 1.5, z: -0.4, rot: 0 }));
}
export const POT = { x: -5.5, z: -0.4 };
export const DECK = { x: 5.5, z: -0.4 };
export const MUCK = { x: 5.5, z: 1.65 };

// Chip colours by denomination, highest first. `edge` is the stripe colour on the rim.
export const DENOMS = [
  { value: 5000, color: "#6b4a2a", edge: "#f1dfb0" },
  { value: 1000, color: "#d8b36a", edge: "#1a1a19" },
  { value: 500, color: "#5b3a8c", edge: "#f4efe4" },
  { value: 100, color: "#1a1a19", edge: "#d8b36a" },
  { value: 25, color: "#1f7a4d", edge: "#f4efe4" },
  { value: 5, color: "#b3261e", edge: "#f4efe4" },
  { value: 1, color: "#e9e4d8", edge: "#2b5fa8" },
];

// How to show an amount in chips: exact, highest denominations first.
export function chipBreakdown(amount) {
  const out = [];
  let left = Math.max(0, Math.round(amount));
  for (const d of DENOMS) {
    const count = Math.floor(left / d.value);
    if (count) out.push({ value: d.value, count });
    left -= count * d.value;
  }
  return out;
}
