import test from "node:test";
import assert from "node:assert/strict";
import { parseCards as P } from "../js/core/cards.js";
import { PCT, TIER_END, RANGE_TIERS, expandRange, handKey, handPct } from "../js/core/preflop.js";
import { positionNames, seatPos, openPct, bbDefendPct, heroInPosition } from "../js/core/positions.js";

const combos = (k) => (k.length === 2 ? 6 : k[2] === "s" ? 4 : 12);

test("handKey", () => {
  assert.equal(handKey(...P("Kd As")), "AKo");
  assert.equal(handKey(...P("Kd Ad")), "AKs");
  assert.equal(handKey(...P("7d 7s")), "77");
});

test("PCT is a cumulative combo-weighted percentile over all 169 hands", () => {
  const keys = Object.keys(PCT);
  assert.equal(keys.length, 169);
  keys.sort((a, b) => PCT[a] - PCT[b]);
  assert.equal(keys[0], "AA");
  let cum = 0;
  for (const k of keys) {
    cum += combos(k);
    assert.ok(Math.abs(PCT[k] - (cum / 1326) * 100) < 1e-9, k);
  }
  assert.equal(cum, 1326);
  assert.equal(handPct(...P("As Ah")), (6 / 1326) * 100);
});

test("expandRange", () => {
  assert.deepEqual([...expandRange("TT+,A9s+,KQo,65s")].sort(), ["TT", "JJ", "QQ", "KK", "AA", "A9s", "ATs", "AJs", "AQs", "AKs", "KQo", "65s"].sort());
});

test("tiers nest: a hand first listed in tier k sits at or below TIER_END[k]", () => {
  assert.equal(TIER_END.length >= RANGE_TIERS.length, true);
  const seen = new Set();
  RANGE_TIERS.forEach((spec, t) => {
    for (const k of expandRange(spec)) {
      if (seen.has(k)) continue;
      seen.add(k);
      assert.ok(PCT[k] <= TIER_END[t] + 1e-9, k + " tier " + t);
      if (t > 0) assert.ok(PCT[k] > TIER_END[t - 1] - 1e-9, k + " above tier " + (t - 1));
    }
  });
  for (let t = 1; t < RANGE_TIERS.length; t++) assert.ok(TIER_END[t] > TIER_END[t - 1]);
});

test("position names", () => {
  assert.deepEqual(positionNames(2), ["BTN/SB", "BB"]);
  assert.deepEqual(positionNames(3), ["BTN", "SB", "BB"]);
  assert.deepEqual(positionNames(6), ["BTN", "SB", "BB", "UTG", "HJ", "CO"]);
  for (let n = 2; n <= 9; n++) {
    const names = positionNames(n);
    assert.equal(names.length, n);
    assert.equal(new Set(names).size, n);
    assert.ok(names.includes("BB"));
  }
  assert.equal(seatPos({ dealer: 4, config: { n: 6 } }, 4), "BTN");
  assert.equal(seatPos({ dealer: 4, config: { n: 6 } }, 0), "BB");
});

test("opening ranges widen toward the button", () => {
  for (const n of [4, 6, 9]) {
    const names = positionNames(n);
    const order = [...names.slice(3), "BTN"]; // preflop order of action, first to last
    for (let i = 1; i < order.length; i++) assert.ok(openPct(order[i], n) >= openPct(order[i - 1], n), n + ": " + order[i]);
    assert.ok(openPct(order[0], n) < openPct("BTN", n));
  }
  assert.ok(bbDefendPct("BTN", 6) > bbDefendPct("UTG", 6));
});

test("heroInPosition", () => {
  const st = (dealer, n) => ({ dealer, config: { n } });
  assert.equal(heroInPosition(st(0, 6), [1, 2, 3]), true); // hero on the button
  assert.equal(heroInPosition(st(5, 6), [2, 3]), false); // hero is the small blind
  assert.equal(heroInPosition(st(4, 6), [4]), false); // hero is the big blind
  assert.equal(heroInPosition(st(2, 6), [1]), false); // hero HJ (seat 0 is 4 after the button), opp CO acts after
  assert.equal(heroInPosition(st(2, 6), [3, 4]), true); // hero HJ against the blinds
  assert.equal(heroInPosition(st(2, 6), [5]), true); // hero HJ, opp UTG acts before
  assert.equal(heroInPosition(st(2, 6), [5, 2]), false); // ...but the button is also in
  assert.equal(heroInPosition(st(0, 2), [1]), true); // heads-up button
  assert.equal(heroInPosition(st(1, 2), [1]), false);
});
