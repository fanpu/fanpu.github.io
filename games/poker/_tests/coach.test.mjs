import test from "node:test";
import assert from "node:assert/strict";
import { makeRng } from "../js/core/rng.js";
import { legalActions } from "../js/core/engine.js";
import { createReads, observeAll } from "../js/core/reads.js";
import { analyze, grade, lessonFor, decisionCategory, CATEGORY_LESSON } from "../js/core/coach.js";
import { rigged, A } from "./helpers/table.mjs";

const LESSON_IDS = ["cards", "ranks", "flow", "outs", "odds", "equity", "start", "ranges", "texture", "betting", "defend", "cbet", "types"];

// Rig a hand, then play `script` (a list of [type, to?]) up to the hero's decision.
function spot(opts, script = []) {
  const { state, events } = rigged(opts);
  const reads = createReads(opts.n);
  observeAll(reads, events);
  for (const [type, to] of script) observeAll(reads, A(state, type, to));
  assert.equal(state.toAct, 0, "the script must end on the hero's turn");
  return { state, reads };
}
function advise(opts, script, options = { light: true, iters: 4000 }) {
  const { state, reads } = spot(opts, script);
  const a = analyze(state, reads, makeRng(8), options);
  const legal = legalActions(state);
  for (const act of [a.rec.action, ...a.rec.alts]) {
    assert.ok(["fold", "check", "call", "raise"].includes(act), "unknown action " + act);
    if (act === "check") assert.ok(legal.canCheck, "recommended a check facing a bet");
    if (act === "call") assert.ok(legal.toCall > 0, "recommended a call with nothing to call");
    if (act === "raise") assert.ok(legal.canRaise, "recommended a raise that is not allowed");
  }
  assert.ok(a.rec.reasons.length >= 1 && a.rec.reasons.every((r) => typeof r === "string" && r.length > 20));
  assert.ok(LESSON_IDS.includes(lessonFor(a).lessonId));
  return a;
}

test("preflop: tight early, wide late, never fold a free flop", () => {
  const utg = advise({ n: 9, dealer: 6, hands: ["7s 2d"] });
  assert.deepEqual([utg.posn, utg.category, utg.rec.action], ["UTG", "Open or fold", "fold"]);

  const btn = advise({ n: 6, dealer: 0, hands: ["As Ks"] }, [["fold"], ["fold"], ["fold"]]);
  assert.deepEqual([btn.posn, btn.rec.action], ["BTN", "raise"]);
  assert.equal(btn.rec.size, 5, "2.5 big blinds");

  const sameHandEarly = advise({ n: 9, dealer: 6, hands: ["9s 8s"] });
  const sameHandLate = advise({ n: 6, dealer: 0, hands: ["9s 8s"] }, [["fold"], ["fold"], ["fold"]]);
  assert.deepEqual([sameHandEarly.rec.action, sameHandLate.rec.action], ["fold", "raise"], "98s is a fold under the gun and an open on the button");

  const bb = advise({ n: 3, dealer: 1, hands: ["8s 3d"] }, [["call"], ["call"]]);
  assert.deepEqual([bb.posn, bb.category, bb.rec.action], ["BB", "Big blind, unopened", "check"]);
  assert.ok(!bb.rec.alts.includes("fold"));
});

test("preflop: facing raises", () => {
  // Seats with the button on 2: 3 SB, 4 BB, 5 UTG, 0 HJ (hero), 1 CO.
  const aces = advise({ n: 6, dealer: 2, hands: ["As Ah"] }, [["fold"], ["raise", 6], ["raise", 20], ["fold"], ["fold"], ["fold"]]);
  assert.deepEqual([aces.category, aces.rec.action], ["Facing a 3-bet", "raise"]);
  const weak = advise({ n: 6, dealer: 2, hands: ["Ks Jd"] }, [["fold"], ["raise", 6], ["raise", 20], ["fold"], ["fold"], ["fold"]]);
  assert.equal(weak.rec.action, "fold");
  assert.ok(weak.eq < aces.eq - 0.3, "KJo is crushed by a 3-betting range; aces are not");

  const vsOpen = advise({ n: 6, dealer: 0, hands: ["7d 2c"] }, [["raise", 6], ["fold"], ["fold"]]);
  assert.deepEqual([vsOpen.category, vsOpen.rec.action], ["Facing a raise", "fold"]);
});

test("river: raise the nuts, fold when dead", () => {
  const toRiver = [["call"], ["check"], ["check"], ["check"], ["check"], ["check"]];
  const nuts = advise({ n: 2, hands: ["As Ks", "7c 7d"], board: "Qs Js Ts 2d 3c" }, [...toRiver, ["raise", 4]]);
  assert.deepEqual([nuts.category, nuts.rec.action, nuts.combos.isNuts], ["Facing a bet", "raise", true]);
  assert.equal(grade(nuts.rec, "raise"), "correct");
  assert.equal(grade(nuts.rec, "call"), "acceptable");
  assert.equal(grade(nuts.rec, "fold"), "mistake");

  const dead = advise({ n: 2, hands: ["2c 3d", "Th 7d"], board: "Ah Kh Qh Jh 9s" }, [...toRiver, ["raise", 4]]);
  assert.equal(dead.rec.action, "fold");
  assert.ok(dead.eq < 0.1);
  assert.ok(Math.abs(dead.potOdds - 1 / 3) < 1e-9, "a pot-sized bet offers 2 to 1: call 4 to win 8");
  assert.ok(dead.evCall < 0);
});

test("flop: a nut flush draw at a good price continues", () => {
  const a = advise({ n: 2, hands: ["Ah 7h", "Kd Qc"], board: "Kh 9h 2c Td 4s" }, [["call"], ["check"], ["raise", 2]]);
  assert.notEqual(a.rec.action, "fold");
  assert.ok(a.potOdds < 0.34 && a.eq > a.potOdds, `equity ${a.eq.toFixed(2)} against a price of ${a.potOdds.toFixed(2)}`);
  assert.equal(a.oa.bigOuts, 9);
  assert.equal(lessonFor(a).lessonId, "outs");
  assert.match(a.hc.label, /nothing yet/);
  assert.equal(a.tex.wetness, "dry", "two-tone but disconnected");
});

test("flop as the preflop raiser: the plan comes from the board", () => {
  const script = [["raise", 5], ["call"], ["check"]];
  const dry = advise({ n: 2, hands: ["Qd Jd", null], board: "As 8d 3c 2h 2s" }, script);
  assert.deepEqual([dry.category, dry.rec.plan, dry.rec.action], ["Bet or check", "small", "raise"]);
  assert.equal(dry.ip, true);
  assert.equal(lessonFor(dry).lessonId, "cbet");
  const low = advise({ n: 2, hands: ["As Qd", null], board: "7h 6d 5c 2h 2s" }, script);
  assert.deepEqual([low.rec.plan, low.rec.action], ["check", "check"]);
});

test("the full analysis fills every panel", (t) => {
  const { state, reads } = spot({ n: 3, dealer: 0, hands: ["Ks Kd"], board: "Kh 9c 2d 5s 8h" }, [
    ["raise", 6],
    ["call"],
    ["call"],
    ["check"],
    ["check"],
  ]);
  const t0 = performance.now();
  const a = analyze(state, reads, makeRng(3));
  const ms = performance.now() - t0;
  assert.equal(a.opps.length, 2);
  for (const o of a.opps) {
    assert.ok(o.nCombos > 0 && o.comp.total === o.nCombos && Object.keys(o.grid).length > 0);
    assert.deepEqual(o.story.length, 2);
    assert.ok(o.pos && typeof o.read.vpip === "number");
  }
  assert.equal(a.nextCards.length, 52);
  assert.ok(a.ev.rows.length >= 3 && a.ev.rows.every((r) => Number.isFinite(r.ev)));
  assert.ok(a.ev.best === "raise" && a.rec.action === "raise", "top set bets");
  assert.ok(a.eq > 0.85 && a.spr > 5);
  assert.ok(ms < 3000, "full analysis took " + ms.toFixed(0) + " ms");
  t.diagnostic(`full analysis: ${ms.toFixed(0)} ms`);
  assert.doesNotThrow(() => structuredClone(a), "the analysis can cross to and from a worker");
});

test("light analysis is fast and refuses to run off-turn", (t) => {
  const { state, reads } = spot({ n: 6, dealer: 0, hands: ["Js Td"], board: "9h 8s 2c Kd 4s" }, [
    ["fold"],
    ["fold"],
    ["fold"],
    ["raise", 5],
    ["fold"],
    ["call"],
    ["check"],
  ]);
  analyze(state, reads, makeRng(1), { light: true, iters: 600 });
  const t0 = performance.now();
  for (let i = 0; i < 10; i++) analyze(state, reads, makeRng(i), { light: true, iters: 600 });
  const ms = (performance.now() - t0) / 10;
  t.diagnostic(`light analysis: ${ms.toFixed(1)} ms`);
  assert.ok(ms < 150, "light analysis took " + ms.toFixed(0) + " ms");
  A(state, "check");
  assert.throws(() => analyze(state, reads, makeRng(1)), /not to act/);
});

test("every decision category leads to a lesson", () => {
  for (const id of Object.values(CATEGORY_LESSON)) assert.ok(LESSON_IDS.includes(id));
  assert.equal(decisionCategory({ street: 0, raises: 0 }, { canCheck: false }), "Open or fold");
  assert.equal(decisionCategory({ street: 2, raises: 0 }, { canCheck: true }), "Bet or check");
  for (const c of ["Open or fold", "Big blind, unopened", "Facing a raise", "Facing a 3-bet", "Bet or check", "Facing a bet"])
    assert.ok(CATEGORY_LESSON[c], c);
});
