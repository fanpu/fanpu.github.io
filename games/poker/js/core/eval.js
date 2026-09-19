import { RANK_NAMES } from "./cards.js";

// Hand evaluation for 5 to 7 cards. A score is an int: category in the top bits (score >> 20, 0..8),
// then up to five 4-bit tiebreak ranks. A higher score wins; equal scores split.
export const CAT_NAMES = [
  "High card",
  "One pair",
  "Two pair",
  "Three of a kind",
  "Straight",
  "Flush",
  "Full house",
  "Four of a kind",
  "Straight flush",
];

// mask has bit r set for each rank present. Returns the top card of the best straight, or 0.
export function straightHigh(mask) {
  const m = mask | (((mask >> 14) & 1) << 1); // the ace also plays low
  for (let h = 14; h >= 5; h--) if (((m >> (h - 4)) & 31) === 31) return h;
  return 0;
}
function ranksFromMask(mask, n) {
  const out = [];
  for (let r = 14; r >= 2 && out.length < n; r--) if (mask & (1 << r)) out.push(r);
  return out;
}
function pack(arr) {
  let v = 0;
  for (let i = 0; i < 5; i++) v = (v << 4) | (arr[i] || 0);
  return v;
}

export function eval7(cards) {
  const cnt = new Array(15).fill(0),
    sc = [0, 0, 0, 0],
    sm = [0, 0, 0, 0];
  let mask = 0;
  for (const c of cards) {
    cnt[c.r]++;
    sc[c.s]++;
    sm[c.s] |= 1 << c.r;
    mask |= 1 << c.r;
  }
  for (let s = 0; s < 4; s++) {
    if (sc[s] >= 5) {
      const sf = straightHigh(sm[s]);
      if (sf) return (8 << 20) | pack([sf]);
      return (5 << 20) | pack(ranksFromMask(sm[s], 5));
    }
  }
  let quad = 0;
  const trips = [],
    pairs = [];
  for (let r = 14; r >= 2; r--) {
    if (cnt[r] === 4) quad = r;
    else if (cnt[r] === 3) trips.push(r);
    else if (cnt[r] === 2) pairs.push(r);
  }
  if (quad) return (7 << 20) | pack([quad, ranksFromMask(mask & ~(1 << quad), 1)[0]]);
  if (trips.length && (pairs.length || trips.length > 1)) return (6 << 20) | pack([trips[0], Math.max(trips[1] || 0, pairs[0] || 0)]);
  const st = straightHigh(mask);
  if (st) return (4 << 20) | pack([st]);
  if (trips.length) return (3 << 20) | pack([trips[0], ...ranksFromMask(mask & ~(1 << trips[0]), 2)]);
  if (pairs.length >= 2) {
    const [p1, p2] = pairs;
    return (2 << 20) | pack([p1, p2, ranksFromMask(mask & ~(1 << p1) & ~(1 << p2), 1)[0]]);
  }
  if (pairs.length === 1) return (1 << 20) | pack([pairs[0], ...ranksFromMask(mask & ~(1 << pairs[0]), 3)]);
  return pack(ranksFromMask(mask, 5));
}
export const evalCat = (score) => score >> 20;

export function describeScore(score) {
  const cat = score >> 20,
    k1 = (score >> 16) & 15,
    k2 = (score >> 12) & 15;
  const n = (r) => RANK_NAMES[r] || "";
  switch (cat) {
    case 8:
      return k1 === 14 ? "Royal flush" : "Straight flush, " + n(k1) + " high";
    case 7:
      return "Four " + n(k1) + "s";
    case 6:
      return "Full house, " + n(k1) + "s full of " + n(k2) + "s";
    case 5:
      return "Flush, " + n(k1) + " high";
    case 4:
      return "Straight, " + n(k1) + " high";
    case 3:
      return "Three " + n(k1) + "s";
    case 2:
      return "Two pair, " + n(k1) + "s and " + n(k2) + "s";
    case 1:
      return "Pair of " + n(k1) + "s";
    default:
      return n(k1) + " high";
  }
}

// The five cards that make the hand (what the table lifts and lights at showdown).
export function best5(cards) {
  const target = eval7(cards),
    n = cards.length;
  if (n === 5) return cards.slice();
  for (let i = 0; i < n; i++) {
    // With six cards j starts at i, so i === j drops a single card; with seven, every pair is dropped in turn.
    for (let j = n === 6 ? i : i + 1; j < n; j++) {
      const five = cards.filter((_, k) => k !== i && k !== j);
      if (five.length === 5 && eval7(five) === target) return five;
    }
  }
  throw new Error("best5: unreachable");
}
