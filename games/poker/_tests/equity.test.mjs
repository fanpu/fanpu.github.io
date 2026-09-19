import test from "node:test";
import assert from "node:assert/strict";
import { parseCards as P, newDeck, cardKey } from "../js/core/cards.js";
import { makeRng } from "../js/core/rng.js";
import { eval7 } from "../js/core/eval.js";
import { simulate, handInRange, flushDrawSuit, straightOuts } from "../js/core/equity.js";

const ANY = { pct: 100, filters: [] };
const near = (actual, expected, tol, msg) =>
  assert.ok(Math.abs(actual - expected) <= tol, `${msg}: got ${actual.toFixed(4)}, want ${expected} ± ${tol}`);

// Exact heads-up equity against a random hand, by brute force over every opponent holding and every runout.
function exact(hero, board) {
  const dead = new Set([...hero, ...board].map(cardKey));
  const deck = newDeck().filter((c) => !dead.has(cardKey(c)));
  let won = 0,
    total = 0;
  const runouts = [];
  const need = 5 - board.length;
  const pick = (start, acc) => {
    if (acc.length === need) return runouts.push(acc.slice());
    for (let i = start; i < deck.length; i++) {
      acc.push(deck[i]);
      pick(i + 1, acc);
      acc.pop();
    }
  };
  pick(0, []);
  for (const run of runouts) {
    const full = [...board, ...run];
    const used = new Set(run.map(cardKey));
    const hs = eval7([...hero, ...full]);
    const live = deck.filter((c) => !used.has(cardKey(c)));
    for (let i = 0; i < live.length; i++)
      for (let j = i + 1; j < live.length; j++) {
        const os = eval7([live[i], live[j], ...full]);
        won += hs > os ? 1 : hs === os ? 0.5 : 0;
        total++;
      }
  }
  return won / total;
}

test("preflop equity against random hands matches published values", () => {
  const rng = makeRng(1);
  near(simulate(P("As Ah"), [], [ANY], 40000, rng), 0.852, 0.012, "AA v 1");
  near(simulate(P("As Ks"), [], [ANY], 40000, rng), 0.67, 0.012, "AKs v 1");
  near(simulate(P("7s 2d"), [], [ANY], 40000, rng), 0.346, 0.012, "72o v 1");
  near(simulate(P("As Ah"), [], [ANY, ANY, ANY], 40000, rng), 0.639, 0.015, "AA v 3");
});

test("postflop equity matches exact enumeration", () => {
  const rng = makeRng(2);
  for (const [hero, board] of [
    ["Qh Jh", "Ah Kh 2c"], // big draw
    ["9s 9d", "Ac 9h 4d"], // set
    ["7c 6c", "Ks Qd 2h 3s"], // air on the turn
    ["Ad Kd", "Kc 8h 5s Td"], // top pair top kicker on the turn
  ]) {
    const want = exact(P(hero), P(board));
    near(simulate(P(hero), P(board), [ANY], 40000, rng), want, 0.012, hero + " on " + board);
  }
});

test("certainties", () => {
  const rng = makeRng(3);
  assert.equal(simulate(P("As Ks"), P("Qs Js Ts 2d 3c"), [ANY], 2000, rng), 1, "royal flush");
  near(simulate(P("2c 3d"), P("As Ks Qs Js Ts"), [ANY, ANY], 2000, rng), 1 / 3, 1e-9, "everyone plays the board");
  assert.equal(
    simulate(P("2c 3d"), P("Ah Kh Qh Jh 9s"), [{ pct: 100, filters: [{ len: 5, mode: "big" }] }], 2000, rng) < 0.02,
    true,
    "drawing dead against a range that overbet"
  );
});

test("ranges matter: kings against a tight 3-betting range", () => {
  const rng = makeRng(4);
  const vsAny = simulate(P("Ks Kh"), [], [ANY], 20000, rng);
  const vsTight = simulate(P("Ks Kh"), [], [{ pct: 3, filters: [] }], 20000, rng);
  // Top 3% is TT+ and AKs. With two kings gone: 6 combos each of AA, QQ, JJ, TT, 2 of AKs, 1 of KK.
  near(vsTight, (6 * 0.18 + 18 * 0.81 + 2 * 0.66 + 0.5) / 27, 0.02, "KK v top 3%");
  assert.ok(vsTight < vsAny - 0.1);
});

test("handInRange applies the preflop percentile and the postflop filters", () => {
  const board = P("Kc 8h 5s");
  assert.equal(handInRange(...P("As Ad"), { pct: 5, filters: [] }, []), true);
  assert.equal(handInRange(...P("7s 2d"), { pct: 5, filters: [] }, []), false);
  const called = { pct: 100, filters: [{ len: 3, mode: "call" }] };
  assert.equal(handInRange(...P("Kd Qd"), called, board), true, "top pair calls");
  assert.equal(handInRange(...P("7c 6c"), called, board), true, "an open-ended draw calls");
  assert.equal(handInRange(...P("Jc 3d"), called, board), false, "nothing does not");
});

test("draw helpers", () => {
  assert.equal(flushDrawSuit(P("Ah 7h"), P("Kh 9h 2c")), 1);
  assert.equal(flushDrawSuit(P("Ac 7d"), P("Kh 9h 2h")), -1, "three on board plus none in hand is not a draw");
  assert.equal(straightOuts(P("9s 8d"), P("7c 6h 2s")), 2);
  assert.equal(straightOuts(P("9s 8d"), P("6c 5h Ks")), 1);
  assert.equal(straightOuts(P("9s 8d"), P("7c 6h 5s")), 0, "already made");
});

test("simulate is reproducible from its seed", () => {
  const run = (seed) => simulate(P("Jc Td"), P("9h 8s 2c"), [{ pct: 30, filters: [] }, ANY], 3000, makeRng(seed));
  assert.equal(run(9), run(9));
  assert.notEqual(run(9), run(10));
});
