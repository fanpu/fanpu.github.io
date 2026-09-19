import { newDeck, cardKey, cardStr, SUITS, RANK_NAMES } from "./cards.js";
import { eval7, evalCat, describeScore, straightHigh, CAT_NAMES } from "./eval.js";
import { handKey, handPct } from "./preflop.js";
import { flushDrawSuit, straightOuts, handInRange, BUCKETS, pickRunout, sampleTable, simLists } from "./equity.js";

export { rangeLists, simLists } from "./equity.js";

// Everything the coach can say about a spot: outs, what kind of hand you hold, what the board is like,
// what each opponent's range is made of, how each next card changes things, and what each bet size is worth.

// ===== Outs & draws (flop/turn) =====
export function outsAnalysis(hero, board) {
  if (board.length < 3 || board.length > 4) return null;
  const all = [...hero, ...board];
  const cur = eval7(all);
  const curCat = evalCat(cur);
  const usedKey = new Set(all.map(cardKey));
  const unseen = newDeck().filter((c) => !usedKey.has(cardKey(c)));
  let outs = 0,
    bigOuts = 0;
  const outCards = [];
  for (const c of unseen) {
    const cat = evalCat(eval7([...all, c]));
    // A card that pairs the board lifts everybody to "one pair": it is only an out if it leaves the hero
    // holding more than the board alone gives. (The old trainer counted those, so ace-high had "15 outs".)
    if (cat > curCat && cat > evalCat(eval7([...board, c]))) {
      outs++;
      const big = cat >= 4 && curCat < 4;
      if (big) bigOuts++;
      outCards.push({ r: c.r, s: c.s, big });
    }
  }
  const draws = [];
  const fs = flushDrawSuit(hero, board);
  if (fs >= 0) {
    const heroHigh = Math.max(...hero.filter((c) => c.s === fs).map((c) => c.r));
    draws.push((heroHigh === 14 ? "nut flush draw" : "flush draw") + " (9 cards)");
  }
  const so = straightOuts(hero, board);
  if (so >= 2) draws.push(so >= 3 ? "double-gutshot or open-ended straight draw (" + so * 4 + " cards)" : "open-ended straight draw (8 cards)");
  else if (so === 1) draws.push("gutshot straight draw (4 cards)");
  const boardMax = Math.max(...board.map((c) => c.r));
  if (curCat === 0 && hero[0].r > boardMax && hero[1].r > boardMax)
    draws.push("two overcards (6 cards make top pair; weak outs, a pair can still lose)");
  if (curCat === 1 && hero[0].r !== hero[1].r) draws.push("a pair: 5 cards make two pair or trips");
  return { current: cur, outs, bigOuts, outCards, draws, cardsToCome: 5 - board.length, hasRealDraw: fs >= 0 || so >= 1 };
}

// ===== Hand class on the board (the vocabulary of hand reading) =====
export function handClass(hero, board) {
  const all = [...hero, ...board];
  const score = eval7(all);
  const cat = evalCat(score);
  const bRanks = board.map((c) => c.r);
  const bMax = Math.max(...bRanks),
    bMin = Math.min(...bRanks);
  const sortedB = [...new Set(bRanks)].sort((a, b) => b - a);
  const bCnt = {};
  for (const r of bRanks) bCnt[r] = (bCnt[r] || 0) + 1;
  const boardPaired = Object.values(bCnt).some((v) => v >= 2);
  const h1 = hero[0].r,
    h2 = hero[1].r;
  const pocket = h1 === h2;
  const n = (r) => RANK_NAMES[r];
  let label = "",
    note = "";
  if (cat >= 4) {
    label = describeScore(score);
  } else if (cat === 3) {
    if (pocket && bCnt[h1]) {
      label = "a set of " + n(h1) + "s";
      note = "A set (pocket pair plus one on the board) is well hidden: opponents cannot see that the board helped you.";
    } else {
      label = "trips (" + n((score >> 16) & 15) + "s, one in your hand)";
      note = "Trips with a paired board are visible to everyone and easier to read than a set.";
    }
  } else if (cat === 2) {
    if (pocket && boardPaired && !bCnt[h1]) {
      label = "two pair, but the weak kind: your pocket " + n(h1) + "s plus the paired board";
      note = 'Everyone with a pair has "two pair" here. Any single card above your pair that pairs the board beats you.';
    } else if (bCnt[h1] && bCnt[h2]) {
      const top2 = [h1, h2].sort((a, b) => b - a);
      label = top2[0] === sortedB[0] && top2[1] === sortedB[1] ? "top two pair" : "two pair using both your cards";
      note = "Two pair with both cards is strong but vulnerable on wet boards: bet for value now rather than later.";
    } else if (boardPaired) {
      const pr = bCnt[h1] ? h1 : h2;
      label = "two pair: your pair of " + n(pr) + "s plus the board pair";
      note = "On a paired board, anyone with a higher pair also has two pair, and trips are possible.";
    } else label = "two pair";
  } else if (cat === 1) {
    const pr = (score >> 16) & 15;
    if (pocket) {
      if (h1 > bMax) {
        label = "an overpair (" + n(h1) + "s above every board card)";
        note = "An overpair beats every top-pair hand, but loses to two pair, sets and completed draws.";
      } else if (h1 < bMin) {
        label = "an underpair (pocket " + n(h1) + "s below the whole board)";
        note = "Only two outs to a set. Treat it as a bluff-catcher at best.";
      } else {
        const above = sortedB.filter((r) => r > h1).length;
        label = "a pocket pair with " + above + " overcard" + (above > 1 ? "s" : "") + " on the board";
        note = "Any opponent holding one of those higher cards has you beat.";
      }
    } else if (bCnt[pr]) {
      const kicker = h1 === pr ? h2 : h1;
      const idx = sortedB.indexOf(pr);
      const tier = ["top pair", "second pair", "third pair", "bottom pair"][Math.min(idx, 3)];
      const kickerGood = kicker >= 12 || kicker > bMax;
      label = tier + " (" + n(pr) + "s) with " + (kicker === 14 || kicker === 8 ? "an " : "a ") + n(kicker) + " kicker";
      if (tier === "top pair")
        note = kickerGood
          ? "Top pair with a strong kicker is a value hand on most boards, but not a hand to stack off 100 bb with."
          : "Top pair with a weak kicker: another top-pair hand with a better kicker is a real possibility. Bet for value, but slow down if raised.";
      else
        note =
          tier.charAt(0).toUpperCase() +
          tier.slice(1) +
          " is a medium-strength hand: good for one street of value or for catching bluffs, rarely for big pots.";
    } else {
      label = "no pair of your own (the board is paired)";
      note = "Your best hand is the board plus a kicker. You are behind anyone holding one of the paired cards.";
    }
  } else {
    const over = hero.filter((c) => c.r > bMax).length;
    label = over === 2 ? "nothing yet, two overcards" : over === 1 ? "nothing yet, one overcard" : "nothing, and the board is above your cards";
  }
  return { score, cat, label, note, boardPaired };
}

// ===== Board texture =====
export function boardTexture(board) {
  const ranks = board.map((c) => c.r);
  const suits = [0, 0, 0, 0];
  for (const c of board) suits[c.s]++;
  const maxSuit = Math.max(...suits);
  const uniq = [...new Set(ranks)].sort((a, b) => b - a);
  const paired = uniq.length < ranks.length;
  let connected = 0; // 2 = very, 1 = some, 0 = none
  const withAce = uniq.includes(14) ? [...uniq, 1] : uniq;
  for (let i = 0; i < withAce.length; i++)
    for (let j = i + 1; j < withAce.length; j++)
      for (let k = j + 1; k < withAce.length; k++) {
        if (withAce[i] - withAce[k] <= 4) connected = 2;
      }
  if (!connected)
    for (let i = 0; i < withAce.length; i++)
      for (let j = i + 1; j < withAce.length; j++) {
        if (withAce[i] - withAce[j] <= 3) connected = Math.max(connected, 1);
      }
  const hi = uniq[0];
  const parts = [];
  if (maxSuit >= 5) parts.push("five of a suit: a flush is on board");
  else if (maxSuit === 4) parts.push("four of a suit: any single " + SUITS[suits.indexOf(4)] + " makes a flush");
  else if (maxSuit === 3)
    parts.push(board.length === 3 ? "monotone (three of one suit): flush draws and made flushes are common" : "three of a suit: a flush is possible");
  else if (maxSuit === 2) parts.push("two-tone: a flush draw is possible");
  else parts.push("rainbow: no flush draw");
  if (paired) parts.push("paired: trips and full houses are possible, and two-pair hands are weaker");
  if (connected === 2) parts.push("connected: straights and straight draws are live");
  else if (connected === 1) parts.push("somewhat connected: a few straight draws");
  else parts.push("disconnected: few straight draws");
  let wet = (maxSuit >= 3 ? 2 : maxSuit === 2 ? 1 : 0) + connected;
  if (paired) wet = Math.max(0, wet - 1);
  const wetness = wet >= 3 ? "wet" : wet >= 2 ? "medium" : "dry";
  const who =
    hi >= 12
      ? "High cards on board favour the preflop raiser (big cards) more than the caller."
      : hi <= 9
        ? "A low board favours the caller (small pairs, connectors) more than a preflop raiser holding big cards."
        : "A middling board helps both ranges about equally.";
  const advice =
    wetness === "wet"
      ? "On wet boards, made hands are vulnerable: bet bigger to charge draws, and expect more semi-bluffs from opponents."
      : wetness === "dry"
        ? "On dry boards, hands change little from street to street: smaller bets work, and a bet is usually value or a pure bluff rather than a semi-bluff."
        : "A medium board: standard sizing (half to two-thirds pot) is fine.";
  return {
    wetness,
    text: board.map(cardStr).join(" ") + " is " + wetness + ": " + parts.join("; ") + ".",
    who,
    advice,
    hi,
    connected,
    maxSuit,
    paired,
  };
}

// Default flop plan for the player who raised before the flop, heads-up (continuation bet, or c-bet).
// Range advantage decides how often to bet. Nut advantage decides how big.
export function cbetPlan(board) {
  if (board.length !== 3) return null;
  const t = boardTexture(board);
  if (t.maxSuit >= 3) return null; // monotone flops play differently, leave them to the equity rule
  if (t.hi <= 9 || (t.connected === 2 && t.hi <= 10))
    return {
      kind: "check",
      frac: 0.7,
      why: "This low or tightly connected flop suits the caller's range (small pairs, suited connectors) better than yours, so you have no range advantage. Check most hands. Bet only strong hands and good draws, and bet big, because those hands need protection and want to build a pot.",
    };
  if (t.wetness === "dry" || (t.wetness === "medium" && t.hi >= 12 && t.connected < 2))
    return {
      kind: "small",
      frac: 0.33,
      why: "This dry, high-card flop favours the preflop raiser: you hold more of the big pairs and strong top pairs (range advantage) and more of the very best hands (nut advantage). The standard play is a small bet, about a third of the pot, with most of your range. It risks little, makes every weak hand pay or fold, and keeps your strong and weak hands looking the same.",
    };
  return {
    kind: "big",
    frac: 0.75,
    why: "This flop is wet and hits both ranges. Betting everything would be too loose, so split your range: bet big (about three-quarters of the pot) with strong hands and good draws, and check your medium hands.",
  };
}

// ===== "Right now" combinatorics: what beats you, and how each range fares =====
export function combosNow(hero, board, opps) {
  const usedKey = new Set([...hero, ...board].map(cardKey));
  const unseen = newDeck().filter((c) => !usedKey.has(cardKey(c)));
  const hs = eval7([...hero, ...board]);
  let best = -1,
    beatTotal = 0,
    total = 0;
  const beatBy = {};
  const perOpp = opps.map(() => ({ ahead: 0, tie: 0, behind: 0, total: 0 }));
  for (let i = 0; i < unseen.length; i++)
    for (let j = i + 1; j < unseen.length; j++) {
      const a = unseen[i],
        b = unseen[j];
      const s = eval7([a, b, ...board]);
      total++;
      if (s > best) best = s;
      if (s > hs) {
        beatTotal++;
        const nm = CAT_NAMES[evalCat(s)];
        beatBy[nm] = (beatBy[nm] || 0) + 1;
      }
      for (let o = 0; o < opps.length; o++) {
        if (!handInRange(a, b, opps[o], board)) continue;
        const po = perOpp[o];
        po.total++;
        if (s > hs) po.behind++;
        else if (s === hs) po.tie++;
        else po.ahead++;
      }
    }
  return { nuts: describeScore(Math.max(best, hs)), isNuts: hs >= best, beatTotal, total, beatBy, perOpp };
}

export function gridCounts(list) {
  const m = {};
  for (const h of list) m[h.key] = (m[h.key] || 0) + 1;
  return m;
}

export function composition(list, hero, board, rng) {
  const out = BUCKETS.map((name, k) => ({ name, k, n: 0, eq: null }));
  const by = BUCKETS.map(() => []);
  for (const h of list) by[h.bk].push(h);
  const dead = new Set([...hero, ...board].map(cardKey));
  const deck = newDeck().filter((c) => !dead.has(cardKey(c)));
  const need = 5 - board.length;
  const hs5 = need === 0 ? eval7([...hero, ...board]) : 0;
  for (let k = 0; k < by.length; k++) {
    const arr = by[k];
    out[k].n = arr.length;
    if (!arr.length) continue;
    let w = 0;
    if (need === 0) {
      for (const h of arr) {
        const s = eval7([h.a, h.b, ...board]);
        w += hs5 > s ? 1 : hs5 === s ? 0.5 : 0;
      }
      out[k].eq = w / arr.length;
    } else {
      const N = 260;
      for (let it = 0; it < N; it++) {
        const h = arr[(rng() * arr.length) | 0];
        const full = board.concat(pickRunout(deck, need, [h.ka, h.kb], rng));
        const a = eval7([...hero, ...full]),
          b = eval7([h.a, h.b, ...full]);
        w += a > b ? 1 : a === b ? 0.5 : 0;
      }
      out[k].eq = w / N;
    }
  }
  return { total: list.length, buckets: out };
}
export function nextCardMap(hero, board, lists, iters, rng) {
  if (board.length < 3 || board.length > 4) return null;
  const dead = new Set([...hero, ...board].map(cardKey));
  const cells = [];
  for (let s = 0; s < 4; s++)
    for (let r = 14; r >= 2; r--) {
      const c = { r, s };
      if (dead.has(cardKey(c))) {
        cells.push({ r, s, eq: null });
        continue;
      }
      cells.push({ r, s, eq: simLists(hero, [...board, c], lists, iters, rng) });
    }
  return cells;
}
// EV of betting or raising. Samples are drawn once; any size can then be priced instantly.
// Whether an opponent continues mirrors how these bots really decide when facing a bet:
// they weigh their equity against the range your betting implies (heroSeen) and compare it with the price.
// heroRange is how the hero's own line looks to the table: opponents weigh their hand against it when deciding to continue.
export function evSamples(hero, board, lists, iters, heroRange, rng) {
  const heroPct = heroRange.pct,
    heroFilters = heroRange.filters;
  const dead = new Set([...hero, ...board].map(cardKey));
  const deck = newDeck().filter((c) => !dead.has(cardKey(c)));
  const cache = new Map();
  const bdead = new Set(board.map(cardKey));
  const un = newDeck().filter((c) => !bdead.has(cardKey(c)));
  const seen = (mode) => {
    const r = { pct: heroPct, filters: heroFilters.concat([{ len: board.length, mode }]) };
    let hl = [];
    for (let i = 0; i < un.length; i++)
      for (let j = i + 1; j < un.length; j++) {
        if (handInRange(un[i], un[j], r, board)) hl.push({ a: un[i], b: un[j], ka: cardKey(un[i]), kb: cardKey(un[j]) });
      }
    if (!hl.length)
      for (let i = 0; i < un.length; i++)
        for (let j = i + 1; j < un.length; j++) hl.push({ a: un[i], b: un[j], ka: cardKey(un[i]), kb: cardKey(un[j]) });
    return hl;
  };
  const hlN = seen("aggr"),
    hlB = seen("big");
  const view = (h) => {
    const k = h.ka * 64 + h.kb;
    let v = cache.get(k);
    if (!v) {
      const oa = outsAnalysis([h.a, h.b], board);
      v = { eqR: simLists([h.a, h.b], board, [hlN], 60, rng), eqB: simLists([h.a, h.b], board, [hlB], 60, rng), drawy: !!(oa && oa.outs >= 8) };
      cache.set(k, v);
    }
    return v;
  };
  const out = [];
  for (let it = 0; it < iters; it++) {
    const t = sampleTable(hero, board, lists, deck, dead, rng);
    out.push({
      hs: eval7([...hero, ...t.full]),
      opp: t.hands.map((h) => {
        if (!h) return null;
        const v = view(h);
        return { eqR: v.eqR, eqB: v.eqB, drawy: v.drawy, s: eval7([h.a, h.b, ...t.full]) };
      }),
    });
  }
  return out;
}
export function evOfRaise(ctx, samples, raiseTo) {
  let sum = 0,
    foldAll = 0,
    calledN = 0,
    calledShare = 0;
  const potThen = ctx.P + (raiseTo - ctx.heroBet);
  const big = raiseTo - ctx.currentBet > 1.5 * (ctx.P + ctx.toCall);
  const price = ctx.opps.map((q) => {
    const c = Math.min(raiseTo, q.bet + q.stack) - q.bet;
    return c / (potThen + c);
  });
  for (const sm of samples) {
    let calls = 0,
      cap = 0,
      best = -1,
      nb = 0,
      any = false;
    for (let k = 0; k < ctx.opps.length; k++) {
      const q = ctx.opps[k],
        oh = sm.opp[k];
      if (!oh) continue;
      const e = big ? oh.eqB : oh.eqR;
      const stays = q.allIn || e > price[k] * 1.05 || (oh.drawy && e > price[k] * 0.8) || (q.station && e > price[k] * 0.85);
      if (!stays) continue;
      any = true;
      const qTo = q.allIn ? q.bet : Math.min(raiseTo, q.bet + q.stack);
      calls += qTo - q.bet;
      if (qTo > cap) cap = qTo;
      if (oh.s > best) {
        best = oh.s;
        nb = 1;
      } else if (oh.s === best) nb++;
    }
    if (!any) {
      sum += ctx.P;
      foldAll++;
      continue;
    }
    const heroTo = Math.max(Math.min(raiseTo, cap), Math.min(raiseTo, ctx.currentBet));
    const add = heroTo - ctx.heroBet;
    const share = sm.hs > best ? 1 : sm.hs === best ? 1 / (nb + 1) : 0;
    sum += share * (ctx.P + add + calls) - add;
    calledN++;
    calledShare += share;
  }
  const n = samples.length || 1;
  return { ev: sum / n, foldAll: foldAll / n, eqCalled: calledN ? calledShare / calledN : null };
}
