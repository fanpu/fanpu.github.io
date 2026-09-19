import test from "node:test";
import assert from "node:assert/strict";
import { makeRng } from "../js/core/rng.js";
import { createGame, startHand, legalActions, apply } from "../js/core/engine.js";
import { createReads, observe, observeAll, readOf, readLabel, rangeOf, READ_PRIOR } from "../js/core/reads.js";
import { rigged, A } from "./helpers/table.mjs";

function table(opts) {
  const { state, events } = rigged(opts);
  const reads = createReads(opts.n);
  observeAll(reads, events);
  return { state, reads, act: (type, to) => observeAll(reads, A(state, type, to)) };
}

test("with no history a read is the prior and carries no label", () => {
  const reads = createReads(6);
  const r = readOf(reads, 3);
  assert.deepEqual([r.n, Math.round(r.vpip), Math.round(r.pfr), r.label, r.sure], [0, READ_PRIOR.vpip, READ_PRIOR.pfr, "", false]);
  assert.equal(readLabel(reads, 3), "");
});

test("labels firm up with sample size", () => {
  const state = createGame({ n: 3 });
  const reads = createReads(3);
  const rng = makeRng(4);
  const hand = () => {
    observeAll(reads, startHand(state, rng));
    while (!state.handOver) {
      const l = legalActions(state);
      // seat 1 raises every time it can, seat 2 never puts a chip in voluntarily, the hero just calls
      const type = l.seat === 1 && l.canRaise && state.street === 0 ? "raise" : l.seat === 2 ? (l.canCheck ? "check" : "fold") : l.canCheck ? "check" : "call";
      observeAll(reads, apply(state, { seat: l.seat, type, ...(type === "raise" ? { to: l.minTo } : {}) }));
    }
  };
  for (let i = 0; i < 24; i++) hand();
  assert.equal(readLabel(reads, 1), "Maniac?");
  assert.equal(readLabel(reads, 2), "Nit?");
  for (let i = 0; i < 40; i++) hand();
  assert.equal(readLabel(reads, 1), "Maniac");
  assert.ok(readOf(reads, 2).vpip < 10);
  assert.ok(readOf(reads, 1).rawR > 90);
});

test("preflop actions narrow the assumed range", () => {
  const t = table({ n: 6, dealer: 0 }); // seats: 0 BTN, 1 SB, 2 BB, 3 UTG, 4 HJ, 5 CO
  const is = (seat, tag, pct) => {
    assert.equal(t.reads.players[seat].tag, tag);
    assert.ok(Math.abs(t.reads.players[seat].pct - pct) < 1e-9, `seat ${seat}: ${t.reads.players[seat].pct} vs ${pct}`);
  };
  assert.deepEqual([t.reads.players[1].tag, t.reads.players[2].tag], ["blind", "blind"]);
  t.act("call"); // UTG limps: assumed to hold their usual entering range (the prior VPIP)
  is(3, "limp", 28);
  t.act("raise", 8); // HJ raises: their raising range (the prior PFR)
  is(4, "raise", 18);
  assert.deepEqual(t.reads.players[4].story, ["raised from HJ"]);
  t.act("call"); // CO calls the raise: three-quarters of their entering range
  is(5, "call", 21);
  t.act("raise", 30); // the hero 3-bets from the button; no stats are kept on the hero, so fixed defaults
  is(0, "3bet", 7);
  t.act("fold");
  t.act("fold");
  t.act("fold");
  const pfrNow = readOf(t.reads, 4).pfr; // already above the prior: this hand's open-raise counts as evidence
  assert.ok(pfrNow > 18);
  t.act("raise", 80); // HJ 4-bets
  is(4, "4bet", pfrNow * 0.15);
  t.act("fold"); // CO
  t.act("call"); // hero calls the 4-bet
  is(0, "call3b", 7);
  assert.deepEqual(rangeOf(t.reads, 4).filters, []);
});

test("postflop actions add board filters", () => {
  const t = table({ n: 2 });
  t.act("raise", 6);
  t.act("call");
  t.act("raise", 6); // BB leads half pot
  assert.deepEqual(t.reads.players[1].filters, [{ len: 3, mode: "aggr" }]);
  t.act("call");
  assert.deepEqual(t.reads.players[0].filters, [{ len: 3, mode: "call" }]);
  t.act("check");
  t.act("raise", 60); // pot is 24: a 60 bet is more than 1.5x the pot
  assert.deepEqual(t.reads.players[0].filters.at(-1), { len: 4, mode: "big" });
  assert.deepEqual(t.reads.players[0].story, ["raised from BTN/SB", "called on the flop", "bet the turn"]);
});

test("a free check in the big blind is not a hand seen; stats survive into the next hand", () => {
  const t = table({ n: 3 });
  t.act("call");
  t.act("call");
  t.act("check"); // BB checks for free
  assert.deepEqual(t.reads.players[2].obs, { hands: 0, vpip: 0, pfr: 0 });
  assert.deepEqual(t.reads.players[1].obs, { hands: 1, vpip: 1, pfr: 0 }); // the small blind completed: that is voluntary
  observe(t.reads, { type: "handStart", handNo: 2, dealer: 1, sbSeat: 2, bbSeat: 0, rebuys: [] });
  assert.deepEqual([t.reads.players[1].tag, t.reads.players[1].pct, t.reads.players[1].filters, t.reads.players[1].story], ["", 100, [], []]);
  assert.equal(t.reads.players[1].obs.hands, 1);
});
