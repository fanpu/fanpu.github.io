import test from "node:test";
import assert from "node:assert/strict";
import { parseCards as P } from "../js/core/cards.js";
import { makeRng } from "../js/core/rng.js";
import { BUCKETS } from "../js/core/equity.js";
import {
  outsAnalysis,
  handClass,
  boardTexture,
  cbetPlan,
  combosNow,
  rangeLists,
  gridCounts,
  composition,
  simLists,
  nextCardMap,
  evSamples,
  evOfRaise,
} from "../js/core/analysis.js";

const ANY = { pct: 100, filters: [] };

test("outs", () => {
  const nfd = outsAnalysis(P("Ah 7h"), P("Kh 9h 2c"));
  assert.deepEqual([nfd.bigOuts, nfd.outs, nfd.cardsToCome, nfd.hasRealDraw], [9, 15, 2, true]);
  assert.match(nfd.draws[0], /nut flush draw/);
  const oesd = outsAnalysis(P("9s 8d"), P("7c 6h 2s"));
  assert.equal(oesd.bigOuts, 8);
  assert.match(oesd.draws.join(), /open-ended/);
  const gut = outsAnalysis(P("9s 8d"), P("6c 5h Ks"));
  assert.equal(gut.bigOuts, 4);
  assert.match(gut.draws.join(), /gutshot/);
  assert.equal(outsAnalysis(P("9h 8h"), P("7h 6h 2s")).bigOuts, 15, "flush draw plus open-ender share two cards");
  const overs = outsAnalysis(P("As Kd"), P("7c 5h 2s"));
  assert.deepEqual([overs.hasRealDraw, overs.bigOuts], [false, 0]);
  assert.match(overs.draws.join(), /overcards/);
  assert.equal(outsAnalysis(P("As Kd"), []), null);
  assert.equal(outsAnalysis(P("As Kd"), P("7c 5h 2s Jd 3c")), null);
});

test("hand class names the hand the way players do", () => {
  const label = (h, b) => handClass(P(h), P(b)).label;
  assert.match(label("7s 7d", "7c Kh 2s"), /^a set of 7s/);
  assert.match(label("Ks 7d", "7c 7h 2s"), /^trips/);
  assert.match(label("Qs Qd", "Jc 7h 2s"), /^an overpair/);
  assert.match(label("3s 3d", "Jc 7h 5s"), /^an underpair/);
  assert.match(label("9s 9d", "Jc 7h 5s"), /pocket pair with 1 overcard /);
  assert.equal(label("Ks 4d", "Kc 9h 2s"), "top pair (Kings) with a 4 kicker");
  assert.match(handClass(P("Ks 4d"), P("Kc 9h 2s")).note, /weak kicker/);
  assert.match(label("9s Ad", "Kc 9h 2s"), /^second pair \(9s\) with an Ace kicker/);
  assert.equal(label("Ks 9d", "Kc 9h 2s"), "top two pair");
  assert.match(label("5s 5d", "Kc Kh 9s"), /the weak kind/);
  assert.match(label("As Qd", "7c 5h 2s"), /two overcards/);
  assert.equal(label("As Ks", "Qs Js Ts"), "Royal flush");
});

test("board texture", () => {
  assert.equal(boardTexture(P("Ks 7d 2c")).wetness, "dry");
  assert.equal(boardTexture(P("9h 8h 7c")).wetness, "wet");
  const paired = boardTexture(P("Ks Kd 4c"));
  assert.deepEqual([paired.paired, paired.wetness], [true, "dry"]);
  assert.match(boardTexture(P("Ah Kh 2h")).text, /monotone/);
  assert.match(boardTexture(P("Ks 7d 2c")).who, /favour the preflop raiser/);
  assert.match(boardTexture(P("8s 6d 3c")).who, /favours the caller/);
});

test("c-bet plan follows the flop", () => {
  assert.equal(cbetPlan(P("As 8d 3c")).kind, "small");
  assert.equal(cbetPlan(P("7h 6d 5c")).kind, "check");
  assert.equal(cbetPlan(P("Jh Th 8c")).kind, "big");
  assert.equal(cbetPlan(P("Ah 9h 4h")), null);
  assert.equal(cbetPlan(P("As 8d 3c 2h")), null);
});

test("combosNow counts what beats you right now", () => {
  const topTwo = combosNow(P("As Ks"), P("Ah Kd 2c"), [ANY]);
  assert.equal(topTwo.isNuts, false);
  assert.equal(topTwo.beatBy["Three of a kind"], 1 + 1 + 3, "one combo each of aces and kings, three of deuces");
  assert.equal(topTwo.beatTotal, 5);
  assert.equal(topTwo.total, 1081);
  assert.equal(topTwo.perOpp[0].behind, 5);
  const topSet = combosNow(P("Ac Ad"), P("As Kd 2c"), [ANY]);
  assert.deepEqual([topSet.isNuts, topSet.beatTotal, topSet.nuts], [true, 0, "Three Aces"]);
});

test("range lists, grids and composition are consistent", () => {
  const hero = P("Ah 7h"),
    board = P("Kh 9h 2c");
  const tight = { pct: 15, filters: [{ len: 3, mode: "aggr" }] };
  const lists = rangeLists(hero, board, [ANY, tight]);
  assert.equal(lists[0].length, 1081);
  assert.ok(lists[1].length > 20 && lists[1].length < 250, "a bet from a 15% range leaves few combos: " + lists[1].length);
  assert.ok(lists[1].every((h) => h.pct <= 15 && h.bk >= 0 && h.bk < BUCKETS.length));
  const grid = gridCounts(lists[1]);
  assert.equal(
    Object.values(grid).reduce((a, b) => a + b, 0),
    lists[1].length
  );
  assert.ok(grid.KK >= 1 && !grid["72o"]);
  const comp = composition(lists[1], hero, board, makeRng(1));
  assert.equal(comp.total, lists[1].length);
  assert.equal(
    comp.buckets.reduce((s, b) => s + b.n, 0),
    comp.total
  );
  for (const b of comp.buckets) assert.ok(b.eq === null ? b.n === 0 : b.eq >= 0 && b.eq <= 1);
  const vsSets = comp.buckets[1].eq,
    vsNothing = comp.buckets[6].eq ?? 1;
  assert.ok(vsSets < 0.45 && vsSets < vsNothing, "a flush draw does worse against sets than against air");
  // An impossible range falls back rather than returning nothing.
  assert.ok(rangeLists(hero, board, [{ pct: 0.1, filters: [{ len: 3, mode: "big" }] }])[0].length > 0);
});

test("next-card map: hearts are good for a heart draw", () => {
  const hero = P("Ah 7h"),
    board = P("Kh 9h 2c");
  const lists = rangeLists(hero, board, [{ pct: 25, filters: [] }]);
  const cells = nextCardMap(hero, board, lists, 120, makeRng(2));
  assert.equal(cells.length, 52);
  assert.equal(cells.filter((c) => c.eq === null).length, 5);
  const avg = (f) => {
    const xs = cells.filter((c) => c.eq !== null && f(c));
    return xs.reduce((s, c) => s + c.eq, 0) / xs.length;
  };
  assert.ok(avg((c) => c.s === 1) > 0.85);
  assert.ok(avg((c) => c.s !== 1 && c.r !== 14 && c.r !== 7) < 0.45);
  assert.equal(nextCardMap(hero, [], lists, 10, makeRng(2)), null);
  const eq = simLists(hero, board, lists, 4000, makeRng(3));
  assert.ok(eq > 0.4 && eq < 0.65, "nut flush draw with an overcard against a 25% range: " + eq);
});

test("EV of a bet: fold equity and called equity", () => {
  const ctx = { P: 20, toCall: 0, heroBet: 0, currentBet: 0, opps: [{ bet: 0, stack: 100, allIn: false, station: false }] };
  const folds = [{ hs: 1, opp: [{ eqR: 0, eqB: 0, drawy: false, s: 5 }] }];
  assert.deepEqual(evOfRaise(ctx, folds, 10), { ev: 20, foldAll: 1, eqCalled: null });
  const callsAndWins = [{ hs: 1, opp: [{ eqR: 0.9, eqB: 0.9, drawy: false, s: 5 }] }];
  assert.deepEqual(evOfRaise(ctx, callsAndWins, 10), { ev: -10, foldAll: 0, eqCalled: 0 });
  const callsAndLoses = [{ hs: 9, opp: [{ eqR: 0.9, eqB: 0.9, drawy: false, s: 5 }] }];
  assert.deepEqual(evOfRaise(ctx, callsAndLoses, 10), { ev: 30, foldAll: 0, eqCalled: 1 });
  const short = { ...ctx, opps: [{ bet: 0, stack: 4, allIn: false, station: false }] };
  assert.equal(evOfRaise(short, callsAndLoses, 10).ev, 24, "the hero can only win what the short stack can call");

  const hero = P("Ks Kd"),
    board = P("Kh 9c 2d");
  const lists = rangeLists(hero, board, [{ pct: 30, filters: [] }]);
  const samples = evSamples(hero, board, lists, 300, { pct: 20, filters: [] }, makeRng(5));
  assert.equal(samples.length, 300);
  const live = { P: 14, toCall: 0, heroBet: 0, currentBet: 0, opps: [{ bet: 0, stack: 190, allIn: false, station: false }] };
  const small = evOfRaise(live, samples, 5),
    huge = evOfRaise(live, samples, 190);
  assert.ok(small.ev > 14 * 0.9, "top set bets for value: " + small.ev);
  assert.ok(huge.foldAll > small.foldAll, "a shove folds out more than a third-pot bet");
});
