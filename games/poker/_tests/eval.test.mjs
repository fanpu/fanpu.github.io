import test from "node:test";
import assert from "node:assert/strict";
import { parseCards as P, cardKey } from "../js/core/cards.js";
import { makeRng } from "../js/core/rng.js";
import { eval7, evalCat, describeScore, best5, CAT_NAMES } from "../js/core/eval.js";
import { randomCards, naive7, naiveCat } from "./helpers/hands.mjs";
import { loadOld } from "./helpers/old.mjs";

const CASES = [
  ["As Ks Qs Js Ts 2d 3c", 8, "Royal flush"],
  ["5h 4h 3h 2h Ah Kd Kc", 8, "Straight flush, 5 high"],
  ["9c 8c 7c 6c 5c Ac Ad", 8, "Straight flush, 9 high"],
  ["7s 7h 7d 7c Ks Qd 2c", 7, "Four 7s"],
  ["Ks Kh Kd Qs Qh Qd 2c", 6, "Full house, Kings full of Queens"],
  ["2s 2h 2d As Ah Kd Kc", 6, "Full house, 2s full of Aces"],
  ["As 9s 7s 4s 2s 8d 6c", 5, "Flush, Ace high"],
  ["9s 8s 7s 6d 5s 2s Kd", 5, "Flush, 9 high"], // the straight is there too; the flush outranks it
  ["Ah 2d 3c 4s 5h Kd Kc", 4, "Straight, 5 high"],
  ["Ah Kd Qc Js Th 2d 2c", 4, "Straight, Ace high"],
  ["8h 8d 8c As Kh 4d 2c", 3, "Three 8s"],
  ["As Ah Kd Kc Qs Qh 2c", 2, "Two pair, Aces and Kings"],
  ["Js Jh 9d 7c 5s 3h 2c", 1, "Pair of Jacks"],
  ["As Jh 9d 7c 5s 3h 2c", 0, "Ace high"],
];
test("categories and descriptions", () => {
  for (const [txt, cat, desc] of CASES) {
    const s = eval7(P(txt));
    assert.equal(evalCat(s), cat, txt);
    assert.equal(describeScore(s), desc, txt);
  }
  assert.equal(CAT_NAMES.length, 9);
});

test("ordering ladder, kickers and ties", () => {
  const ladder = [
    "As Ks Qs Js Ts 2d 3c", // royal
    "9c 8c 7c 6c 5c Ad Ah", // straight flush
    "7s 7h 7d 7c As Qd 2c", // quads, ace kicker
    "7s 7h 7d 7c Ks Qd 2c", // quads, king kicker
    "Ks Kh Kd Qs Qh 3d 2c", // kings full
    "Qs Qh Qd Ks Kh 3d 2c", // queens full
    "As 9s 7s 4s 2s 8d 6c", // flush
    "Ah Kd Qc Js Th 2d 2c", // broadway
    "Ah 2d 3c 4s 5h Kd Kc", // wheel
    "8h 8d 8c As Kh 4d 2c", // trips
    "As Ah Kd Kc Qs Qh 2c", // aces up, queen kicker (third pair counts as kicker)
    "As Ah Kd Kc Js 9h 2c", // aces up, jack kicker
    "Js Jh Ad 7c 5s 3h 2c", // pair, ace kicker
    "Js Jh Kd 7c 5s 3h 2c", // pair, king kicker
    "As Jh 9d 7c 5s 3h 2c", // ace high
  ].map((t) => eval7(P(t)));
  for (let i = 1; i < ladder.length; i++) assert.ok(ladder[i - 1] > ladder[i], "ladder step " + i);
  assert.equal(eval7(P("As Kd 9h 7c 5s 3h 2c")), eval7(P("Ah Kc 9d 7s 5h 3d 2s")), "suits do not matter without a flush");
  const board = "Ah Kd Qc Js Th";
  assert.equal(eval7(P("2s 3d " + board)), eval7(P("9c 9d " + board)), "both play the board");
  assert.equal(eval7(P("As Kd Qh Jc 9s")), eval7(P("As Kd Qh Jc 9s")), "five cards work");
});

test("agrees with a naive 21-subset evaluator on 100,000 hands", () => {
  const rng = makeRng(2026);
  let prev = null;
  for (let i = 0; i < 100000; i++) {
    const h = randomCards(rng, 7);
    const s = eval7(h),
      nv = naive7(h);
    assert.equal(evalCat(s), naiveCat(nv));
    if (prev) assert.equal(Math.sign(s - prev.s), Math.sign(nv - prev.nv));
    prev = { s, nv };
  }
});

test("agrees exactly with the old trainer's evaluator", () => {
  const old = loadOld(makeRng(1));
  const rng = makeRng(99);
  for (let i = 0; i < 20000; i++) {
    const h = randomCards(rng, 5 + (i % 3));
    assert.equal(eval7(h), old.eval7(h));
  }
});

test("best5 picks five of the given cards that make the hand", () => {
  const rng = makeRng(5);
  for (let i = 0; i < 5000; i++) {
    const h = randomCards(rng, 5 + (i % 3));
    const five = best5(h);
    assert.equal(five.length, 5);
    const keys = new Set(h.map(cardKey));
    assert.ok(five.every((c) => keys.has(cardKey(c))));
    assert.equal(new Set(five.map(cardKey)).size, 5);
    assert.equal(eval7(five), eval7(h));
  }
  const glow = best5(P("Ah Kd 2c 2d 7s Ac 9h")).map(cardKey).sort();
  assert.deepEqual(glow, P("Ah Ac 2c 2d Kd").map(cardKey).sort());
});
