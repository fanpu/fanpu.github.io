import { RANKS } from "./cards.js";

// Preflop hand strength as a percentile: PCT["AKs"] = 3.2 means "inside the top 3.2% of starting hands".
// Lower is stronger. Ranges throughout the trainer are "the top x%" of this ordering.

// Chen formula. Only breaks ties inside a tier (see RANGE_TIERS).
export function chenScore(r1, r2, suited) {
  const hv = (r) => (r === 14 ? 10 : r === 13 ? 8 : r === 12 ? 7 : r === 11 ? 6 : r / 2);
  let s = hv(r1);
  if (r1 === r2) return Math.ceil(Math.max(5, s * 2));
  if (suited) s += 2;
  const gap = r1 - r2 - 1;
  if (gap === 1) s -= 1;
  else if (gap === 2) s -= 2;
  else if (gap === 3) s -= 4;
  else if (gap >= 4) s -= 5;
  if (gap <= 1 && r1 < 12) s += 1;
  return Math.ceil(s);
}

export function handKey(c1, c2) {
  const hi = Math.max(c1.r, c2.r),
    lo = Math.min(c1.r, c2.r);
  if (hi === lo) return RANKS[hi - 2] + RANKS[lo - 2];
  return RANKS[hi - 2] + RANKS[lo - 2] + (c1.s === c2.s ? "s" : "o");
}

// Hand ordering. Tiers are simplified from published solver-based opening charts (8-max early seats,
// then 6-max cash LJ, HJ, CO, BTN, SB). A hand belongs to the first tier that lists it. Inside a tier,
// and for hands no chart opens, the Chen score only breaks ties.
export const RANGE_TIERS = [
  "TT+,AKs,AKo",
  "99+,AJs+,KQs,AQo+",
  "77+,A3s+,K9s+,QTs+,JTs,T9s,AQo+,KQo",
  "77+,A3s+,K8s+,QTs+,JTs,T9s,AJo+,KQo",
  "66+,A3s+,K8s+,Q9s+,J9s+,T9s,ATo+,KJo+,QJo",
  "55+,A2s+,K6s+,Q9s+,J9s+,T9s,98s,87s,76s,ATo+,KTo+,QTo+",
  "33+,A2s+,K3s+,Q6s+,J8s+,T7s+,97s+,87s,76s,A8o+,KTo+,QTo+,JTo",
  "22+,A2s+,K2s+,Q3s+,J4s+,T6s+,96s+,85s+,75s+,64s+,53s+,A4o+,K8o+,Q9o+,J9o+,T8o+,98o",
  "22+,A2s+,K2s+,Q2s+,J2s+,T3s+,94s+,84s+,74s+,63s+,53s+,43s,A2o+,K4o+,Q5o+,J7o+,T7o+,96o+,86o+,76o",
];

// "TT+,A9s+,KQo" -> Set of hand keys. "+" on a pair means every higher pair; on XYs/XYo it means
// every higher second card up to one below the first.
export function expandRange(spec) {
  const out = new Set();
  for (const tok of spec.split(",")) {
    const t = tok.trim();
    const plus = t.endsWith("+");
    const b = plus ? t.slice(0, -1) : t;
    if (b.length === 2) {
      const r = RANKS.indexOf(b[0]);
      for (let k = r; k <= (plus ? 12 : r); k++) out.add(RANKS[k] + RANKS[k]);
    } else {
      const hi = RANKS.indexOf(b[0]),
        lo = RANKS.indexOf(b[1]);
      for (let k = lo; k <= (plus ? hi - 1 : lo); k++) out.add(b[0] + RANKS[k] + b[2]);
    }
  }
  return out;
}

export const PCT = {};
export const TIER_END = [];
(function buildPct() {
  const tiers = RANGE_TIERS.map(expandRange);
  const tierOf = (k) => {
    for (let t = 0; t < tiers.length; t++) if (tiers[t].has(k)) return t;
    return tiers.length;
  };
  const list = [];
  for (let a = 14; a >= 2; a--)
    for (let b = a; b >= 2; b--) {
      const base = RANKS[a - 2] + RANKS[b - 2];
      if (a === b) list.push({ k: base, tier: tierOf(base), score: chenScore(a, b, false), combos: 6, tb: 3 });
      else {
        list.push({ k: base + "s", tier: tierOf(base + "s"), score: chenScore(a, b, true), combos: 4, tb: 2 });
        list.push({ k: base + "o", tier: tierOf(base + "o"), score: chenScore(a, b, false), combos: 12, tb: 1 });
      }
    }
  list.sort((x, y) => x.tier - y.tier || y.score - x.score || y.tb - x.tb || y.k.charCodeAt(0) - x.k.charCodeAt(0));
  let cum = 0;
  for (const h of list) {
    cum += h.combos;
    PCT[h.k] = (cum / 1326) * 100;
    TIER_END[h.tier] = PCT[h.k];
  }
})();

export const handPct = (c1, c2) => PCT[handKey(c1, c2)];
