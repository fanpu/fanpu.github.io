// Table geometry, as pure functions of the number of seats. No three.js here, so it is tested in node.
// Units: 1 = 10 cm. x runs along the table, the hero sits at +z (nearest the camera), y is up.
// Cards and chips are larger than life against the table, on purpose: they have to be read from across the room.
export const TABLE = { a: 13, b: 7.5, rail: 1.5 }; // felt half-length, felt half-width, rail width
export const CARD = { w: 1.26, h: 1.76, t: 0.02 };
export const CHIP = { r: 0.39, t: 0.066 };
export const HERO_SCALE = 1.5;

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

// On a portrait screen the whole table turns a quarter: its long axis runs away from the viewer and the hero
// sits at the near end. The same racetrack, entered at the middle of its right-hand end and rotated so that
// end faces +z. Order round the table, and everything measured from the rail, are unchanged.
const T_END = 1 - (L + (Math.PI * TABLE.b) / 2) / (4 * L + 2 * Math.PI * TABLE.b);
function edgeFor(t, portrait) {
  if (!portrait) return edge(t);
  const e = edge(t + T_END);
  return { x: -e.z, z: e.x, nx: -e.nz, nz: e.nx, tx: -e.tz, tz: e.tx };
}

// Everything that belongs to a seat, found by stepping in from the rail (inward) and along it (toward the next seat).
export function seatLayout(n, portrait = false) {
  const seats = [];
  for (let seat = 0; seat < n; seat++) {
    const t = seat / n,
      e = edgeFor(t, portrait);
    const at = (inward, along = 0) => ({ x: e.x - e.nx * inward + e.tx * along, z: e.z - e.nz * inward + e.tz * along });
    const rot = Math.atan2(e.nx, e.nz); // turn about y so that a card's foot points at its player
    const hero = seat === 0;
    // A portrait table is narrow, so there cards sit nearer the rail and bets stay closer to their owners,
    // which leaves the middle free for the board.
    const inset = hero ? { cards: 2.25, bet: 4.3 } : portrait ? { cards: 1.55, bet: 3.3 } : { cards: 2.1, bet: 3.9 };
    const half = hero ? 1.0 : 0.68; // half the distance between the two cards' centres
    const mid = at(inset.cards);
    seats.push({
      seat,
      t,
      rot,
      pos: at(-TABLE.rail / 2),
      // On a phone the side labels are pulled in from the screen's edge, so they are slid along the rail to stay clear of the cards.
      label: at(-TABLE.rail - 0.4, portrait && !hero ? 2.7 : 0),
      // The hero's own two cards are the most important thing on the table, so they are drawn half as big again.
      cards: [
        { ...at(inset.cards, half), rot, ...(hero ? { scale: HERO_SCALE } : {}) },
        { ...at(inset.cards, -half), rot, ...(hero ? { scale: HERO_SCALE } : {}) },
      ],
      // Where an opponent's cards go when they are turned over at showdown: the same spot, but side by side and facing the hero.
      shown: [
        { x: mid.x - half, z: mid.z, rot: 0, ...(hero ? { scale: HERO_SCALE } : {}) },
        { x: mid.x + half, z: mid.z, rot: 0, ...(hero ? { scale: HERO_SCALE } : {}) },
      ],
      bet: at(inset.bet),
      stack: at(1.7, hero ? -3.3 : -2.3), // on the player's right
      button: at(3.0, hero ? 3.2 : 2.2), // on the player's left
    });
  }
  return seats;
}

export function boardSlots(portrait = false) {
  return [-2, -1, 0, 1, 2].map((i) => ({ x: i * (portrait ? 1.36 : 1.5), z: portrait ? 0.6 : -0.4, rot: 0 }));
}
export const POT = { x: -5.5, z: -0.4 };
export const DECK = { x: 5.5, z: -0.4 };
export const MUCK = { x: 5.5, z: 1.65 };

// One object with everything the stage needs to know about where things go, for either orientation.
export function layoutFor(portrait = false) {
  return {
    portrait,
    halfX: portrait ? TABLE.b : TABLE.a,
    halfZ: portrait ? TABLE.a : TABLE.b,
    seats: (n) => seatLayout(n, portrait),
    board: boardSlots(portrait),
    pot: portrait ? { x: 0, z: -2.3 } : POT,
    deck: portrait ? { x: -1.3, z: -5.2 } : DECK,
    muck: portrait ? { x: 1.3, z: -5.2 } : MUCK,
    inside: (p, margin = 0) => insideFelt(portrait ? { x: p.z, z: -p.x } : p, margin),
  };
}

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

// How to show an amount in chips: exact, highest denominations first, but always holding one chip of each
// denomination back to be made up in smaller ones, so that a bet looks like a stack and not a single chip.
export function chipBreakdown(amount) {
  const out = [];
  let left = Math.max(0, Math.round(amount));
  DENOMS.forEach((d, i) => {
    const last = i === DENOMS.length - 1;
    const count = Math.max(0, Math.floor(left / d.value) - (last ? 0 : 1));
    if (count) out.push({ value: d.value, count });
    left -= count * d.value;
  });
  return out;
}
