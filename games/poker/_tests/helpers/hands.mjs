import { newDeck } from "../../js/core/cards.js";
import { randInt } from "../../js/core/rng.js";

// k distinct random cards (partial Fisher-Yates on a fresh deck).
export function randomCards(rng, k) {
  const d = newDeck();
  for (let i = 0; i < k; i++) {
    const j = i + randInt(rng, d.length - i);
    const t = d[i];
    d[i] = d[j];
    d[j] = t;
  }
  return d.slice(0, k);
}

// Deliberately naive evaluator: rank each of the 21 five-card subsets from first principles, take the max.
function naive5(cs) {
  const ranks = cs.map((c) => c.r).sort((a, b) => b - a);
  const flush = cs.every((c) => c.s === cs[0].s);
  const uniq = [...new Set(ranks)];
  let straight = 0;
  if (uniq.length === 5) {
    if (ranks[0] - ranks[4] === 4) straight = ranks[0];
    else if (ranks.join() === "14,5,4,3,2") straight = 5;
  }
  const count = {};
  for (const r of ranks) count[r] = (count[r] || 0) + 1;
  const groups = uniq.sort((a, b) => count[b] - count[a] || b - a);
  const shape = groups.map((r) => count[r]).join("");
  let cat;
  if (straight && flush) cat = 8;
  else if (shape === "41") cat = 7;
  else if (shape === "32") cat = 6;
  else if (flush) cat = 5;
  else if (straight) cat = 4;
  else if (shape === "311") cat = 3;
  else if (shape === "221") cat = 2;
  else if (shape === "2111") cat = 1;
  else cat = 0;
  const tiebreak = straight ? [straight] : groups;
  let v = cat;
  for (let i = 0; i < 5; i++) v = v * 15 + (tiebreak[i] || 0);
  return v;
}
export function naive7(cards) {
  let best = -1;
  const n = cards.length;
  for (let a = 0; a < n; a++)
    for (let b = a + 1; b < n; b++)
      for (let c = b + 1; c < n; c++)
        for (let d = c + 1; d < n; d++)
          for (let e = d + 1; e < n; e++) best = Math.max(best, naive5([cards[a], cards[b], cards[c], cards[d], cards[e]]));
  return best;
}
export const naiveCat = (v) => Math.floor(v / 15 ** 5);
