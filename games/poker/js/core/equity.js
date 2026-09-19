import { newDeck, cardKey } from "./cards.js";
import { eval7, evalCat, straightHigh } from "./eval.js";
import { handKey, handPct } from "./preflop.js";

// A range is { pct, filters }: the top pct% of starting hands, narrowed by one filter per postflop action
// the player has taken ({ len: board length at the time, mode: 'call' | 'aggr' | 'big' }).

// ===== Draw detection helpers =====
export function flushDrawSuit(hand, board) {
  const sc = [0, 0, 0, 0];
  for (const c of hand) sc[c.s]++;
  for (const c of board) sc[c.s]++;
  for (let s = 0; s < 4; s++) if (sc[s] === 4 && hand.some((c) => c.s === s)) return s;
  return -1;
}
export function straightOuts(hand, board) {
  // number of distinct ranks completing a straight (0 if already straight)
  let mask = 0;
  for (const c of hand) mask |= 1 << c.r;
  for (const c of board) mask |= 1 << c.r;
  if (straightHigh(mask)) return 0;
  let n = 0;
  for (let r = 2; r <= 14; r++) if (!(mask & (1 << r)) && straightHigh(mask | (1 << r))) n++;
  return n;
}

// Does a sampled opponent hand fit how they played postflop?
export function fitsBoard(hand, board, mode, pct) {
  const sc = eval7([...hand, ...board]),
    bsc = eval7(board),
    cat = evalCat(sc);
  if (mode !== "big" && (cat > evalCat(bsc) || (cat >= 4 && sc > bsc))) return true; // must beat what the board gives everyone
  if (mode === "big") {
    // an overbet: top pair or better, or a draw with a lot of outs
    if (bucketOf(hand, board, bsc) <= 3) return true;
    return board.length < 5 && flushDrawSuit(hand, board) >= 0 && straightOuts(hand, board) >= 1;
  }
  if (flushDrawSuit(hand, board) >= 0) return true;
  const so = straightOuts(hand, board);
  if (mode === "aggr") return so >= 2 || pct <= 8;
  return so >= 1 || pct <= 20;
}
export function handInRange(a, b, opp, board) {
  const pct = handPct(a, b);
  if (opp.pct < 100 && pct > opp.pct) return false;
  if (opp.filters)
    for (const f of opp.filters) {
      if (board.length >= f.len && !fitsBoard([a, b], board.slice(0, f.len), f.mode, pct)) return false;
    }
  return true;
}

export const BUCKETS = ["Straight or better", "Set or trips", "Two pair", "Top pair or overpair", "Weaker pair", "Draw", "Nothing"];
export function bucketOf(hand, board, bs) {
  const s = eval7([...hand, ...board]),
    cat = s >> 20,
    bcat = bs >> 20;
  const draw = () => (board.length < 5 && (flushDrawSuit(hand, board) >= 0 || straightOuts(hand, board) >= 2) ? 5 : 6);
  if (cat >= 4) return s === bs ? 6 : 0;
  if (cat === 3) return bcat === 3 ? draw() : 1;
  if (cat === 2) {
    if (bcat === 2) return s >> 12 === bs >> 12 ? draw() : 2;
    if (bcat === 1) {
      const bp = (bs >> 16) & 15,
        p1 = (s >> 16) & 15,
        p2 = (s >> 12) & 15;
      const own = p1 === bp ? p2 : p1;
      const others = board.map((c) => c.r).filter((r) => r !== bp);
      const top = others.length ? Math.max(...others) : 0;
      return own >= top ? 3 : 4;
    }
    return 2;
  }
  if (cat === 1) {
    if (bcat === 1) return draw();
    const pr = (s >> 16) & 15;
    return pr >= Math.max(...board.map((c) => c.r)) ? 3 : 4;
  }
  return draw();
}

// ===== Sampling from ranges =====
// ranges: one { pct, filters } per opponent. Each list holds every two-card holding still consistent with that range.
export function rangeLists(hero, board, ranges) {
  const dead = new Set([...hero, ...board].map(cardKey));
  const unseen = newDeck().filter((c) => !dead.has(cardKey(c)));
  const bs = board.length ? eval7(board) : 0;
  const build = (r) => {
    const list = [];
    for (let i = 0; i < unseen.length; i++)
      for (let j = i + 1; j < unseen.length; j++) {
        const a = unseen[i],
          b = unseen[j];
        if (!handInRange(a, b, r, board)) continue;
        const pct = handPct(a, b);
        let bk = -1;
        if (board.length) bk = bucketOf([a, b], board, bs);
        list.push({ a, b, ka: cardKey(a), kb: cardKey(b), key: handKey(a, b), bk, pct });
      }
    return list;
  };
  return ranges.map((r) => {
    let l = build(r);
    if (!l.length) l = build({ pct: r.pct, filters: [] });
    if (!l.length) l = build({ pct: 100, filters: [] });
    return l;
  });
}

export function pickRunout(deck, need, usedKeys, rng) {
  const out = [];
  let guard = 0;
  while (out.length < need && guard++ < 200) {
    const c = deck[(rng() * deck.length) | 0];
    const k = cardKey(c);
    if (usedKeys.includes(k)) continue;
    usedKeys.push(k);
    out.push(c);
  }
  return out;
}

// One joint sample: a hand for each opponent from their list (no shared cards), then a runout.
export function sampleTable(hero, board, lists, deck, dead, rng) {
  const used = [];
  const hands = [];
  for (const list of lists) {
    let h = null;
    for (let t = 0; t < 30; t++) {
      const c = list[(rng() * list.length) | 0];
      if (dead.has(c.ka) || dead.has(c.kb) || used.includes(c.ka) || used.includes(c.kb)) continue;
      h = c;
      break;
    }
    if (h) {
      used.push(h.ka, h.kb);
    }
    hands.push(h);
  }
  const full = board.concat(pickRunout(deck, 5 - board.length, used, rng));
  return { hands, full };
}

export function simLists(hero, board, lists, iters, rng) {
  const dead = new Set([...hero, ...board].map(cardKey));
  const deck = newDeck().filter((c) => !dead.has(cardKey(c)));
  let win = 0;
  for (let it = 0; it < iters; it++) {
    const t = sampleTable(hero, board, lists, deck, dead, rng);
    const hs = eval7([...hero, ...t.full]);
    let best = -1,
      nb = 0;
    for (const h of t.hands) {
      if (!h) continue;
      const s = eval7([h.a, h.b, ...t.full]);
      if (s > best) {
        best = s;
        nb = 1;
      } else if (s === best) nb++;
    }
    if (hs > best) win += 1;
    else if (hs === best) win += 1 / (nb + 1);
  }
  return win / iters;
}

// Monte Carlo equity for the hero against one range per opponent.
// Opponent hands are drawn from the enumerated range, so a tight range is sampled as faithfully as a wide one.
// (The old trainer tried 40 random hands and then gave up and dealt a random one, which for a 3% range happened
// four times in ten and overstated the hero's equity against 3-bets and 4-bets by several points.)
export function simulate(hero, board, ranges, iters, rng) {
  return simLists(hero, board, rangeLists(hero, board, ranges), iters, rng);
}
