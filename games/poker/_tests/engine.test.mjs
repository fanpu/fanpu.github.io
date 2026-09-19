import test from "node:test";
import assert from "node:assert/strict";
import { makeRng } from "../js/core/rng.js";
import { createGame, startHand, legalActions, clampRaise, apply, pot, snapshot, replay, BB, BUYIN } from "../js/core/engine.js";
import { rigged, A, chips, ofType } from "./helpers/table.mjs";

test("blinds and order of action, three-handed", () => {
  const { state, events } = rigged({ n: 3 });
  assert.deepEqual(
    events.map((e) => e.type),
    ["handStart", "post", "post", "deal"]
  );
  assert.deepEqual([events[0].dealer, events[0].sbSeat, events[0].bbSeat], [0, 1, 2]);
  assert.equal(state.toAct, 0);
  assert.deepEqual(legalActions(state), { seat: 0, toCall: 2, canCheck: false, canRaise: true, minTo: 4, maxTo: 200, pot: 3 });
  A(state, "call");
  A(state, "call");
  assert.equal(state.toAct, 2);
  assert.equal(legalActions(state).canCheck, true);
  const ev = A(state, "check");
  const st = ofType(ev, "street")[0];
  assert.deepEqual([st.street, st.cards.length, st.pot], [1, 3, 6]);
  assert.equal(state.toAct, 1, "small blind acts first after the flop");
});

test("heads-up: the button posts the small blind, acts first preflop and last after", () => {
  const { state, events } = rigged({ n: 2 });
  assert.deepEqual([events[0].dealer, events[0].sbSeat, events[0].bbSeat], [0, 0, 1]);
  assert.equal(state.toAct, 0);
  A(state, "call");
  A(state, "check");
  assert.equal(state.street, 1);
  assert.equal(state.toAct, 1);
});

test("folding round to the big blind", () => {
  const { state } = rigged({ n: 3 });
  A(state, "fold");
  const ev = A(state, "fold");
  assert.deepEqual(
    ev.map((e) => e.type),
    ["action", "refund", "award", "handEnd"]
  );
  assert.deepEqual([ev[1].seat, ev[1].amount], [2, 1]);
  assert.deepEqual([ev[2].amount, ev[2].seats, ev[2].contested], [2, [2], false]);
  assert.deepEqual(ev[3].net, { 0: 0, 1: -1, 2: 1 });
  assert.equal(ev[3].showdown, false);
  assert.equal(state.handOver, true);
  assert.equal(state.toAct, -1);
  assert.equal(legalActions(state), null);
});

test("minimum raise follows the last full raise", () => {
  const { state } = rigged({ n: 3 });
  A(state, "raise", 6);
  assert.equal(legalActions(state).minTo, 10);
  A(state, "raise", 18);
  const legal = legalActions(state);
  assert.equal(legal.minTo, 30);
  assert.throws(() => A(state, "raise", 20), /raise/);
  assert.equal(clampRaise(legal, 20), 30);
  assert.equal(clampRaise(legal, 9999), 200);
  assert.equal(clampRaise(legal, 41.6), 42);
});

test("illegal actions throw and leave the state untouched", () => {
  const { state } = rigged({ n: 3 });
  const before = JSON.stringify(state);
  assert.throws(() => apply(state, { seat: 1, type: "call" }), /turn/);
  assert.throws(() => A(state, "check"), /check/);
  assert.throws(() => A(state, "raise", 201), /raise/);
  assert.throws(() => A(state, "raise", 3), /raise/);
  assert.throws(() => A(state, "dance"), /unknown/);
  assert.equal(JSON.stringify(state), before);
  A(state, "call");
  A(state, "call");
  assert.throws(() => A(state, "call"), /call/);
  A(state, "check");
  A(state, "check"); // flop: seat 1
  A(state, "check");
  A(state, "fold"); // folding when you could check is legal, if foolish
  assert.equal(state.players[0].folded, true);
  const { state: done } = rigged({ n: 2 });
  A(done, "fold");
  assert.throws(() => A(done, "check"), /over/);
});

test("actions are named bet or raise correctly", () => {
  const { state } = rigged({ n: 2 });
  assert.equal(A(state, "raise", 6)[0].kind, "raise");
  A(state, "call");
  const bet = A(state, "raise", 8)[0];
  assert.deepEqual([bet.kind, bet.added, bet.to, bet.street, bet.potBefore, bet.toCallBefore], ["bet", 8, 8, 1, 12, 0]);
  const re = A(state, "raise", 30)[0];
  assert.deepEqual([re.kind, re.added, re.to, re.toCallBefore, re.currentBetBefore], ["raise", 30, 30, 8, 8]);
  assert.equal(re.pos, "BTN/SB");
});

test("a short all-in does not reopen raising for players who already acted", () => {
  const { state } = rigged({ n: 3, stacks: [200, 27, 200] }); // seat 1 is the small blind with 27
  A(state, "raise", 20); // seat 0
  const shove = A(state, "raise", 27)[0]; // seat 1 all-in: 7 more than the bet, under the minimum raise of 18
  assert.deepEqual([shove.allIn, shove.full], [true, false]);
  assert.equal(legalActions(state).canRaise, true, "the big blind has not acted yet and may still raise");
  A(state, "call");
  const back = legalActions(state);
  assert.deepEqual([back.seat, back.toCall, back.canRaise], [0, 7, false]);
  A(state, "call");
  assert.equal(state.street, 1);
  assert.equal(legalActions(state).canRaise, true, "the lock ends with the betting round");
});

test("you cannot raise when nobody is left to respond", () => {
  const { state } = rigged({ n: 2, stacks: [200, 40] });
  A(state, "call");
  A(state, "raise", 40); // big blind shoves
  assert.deepEqual([legalActions(state).toCall, legalActions(state).canRaise], [38, false]);
});

test("three-way all-in builds a main pot, a side pot and a refund", () => {
  const { state } = rigged({
    n: 3,
    stacks: [50, 120, 200],
    hands: ["As Ah", "Ks Kh", "Qs Qh"],
    board: "2c 7d 9h Jc 3s",
  });
  const total = chips(state);
  A(state, "raise", 50);
  A(state, "raise", 120);
  const ev = A(state, "call");
  assert.deepEqual([ofType(ev, "refund").length, ofType(ev, "street").length, ofType(ev, "showdown").length], [0, 3, 1]);
  const awards = ofType(ev, "award");
  assert.deepEqual(
    awards.map((a) => [a.potIndex, a.amount, a.seats, a.contested]),
    [
      [0, 150, [0], true],
      [1, 140, [1], true],
    ]
  );
  assert.deepEqual(
    state.players.map((p) => p.stack),
    [150, 140, 80]
  );
  assert.equal(chips(state), total);
  const sd = ofType(ev, "showdown")[0];
  assert.equal(sd.hands.length, 3);
  assert.equal(sd.hands[0].desc, "Pair of Aces");
  assert.equal(sd.hands[0].best5.length, 5);
  assert.deepEqual(ofType(ev, "handEnd")[0].net, { 0: 100, 1: 20, 2: -120 });
});

test("an uncalled shove is refunded before the board runs out", () => {
  const { state } = rigged({ n: 2, stacks: [200, 40], hands: ["2c 3d", "As Ah"], board: "Kc 7d 9h Jc 4s" });
  A(state, "raise", 200);
  const ev = A(state, "call");
  assert.deepEqual(
    ev.map((e) => e.type),
    ["action", "refund", "street", "street", "street", "showdown", "award", "handEnd"]
  );
  assert.deepEqual([ev[1].seat, ev[1].amount], [0, 160]);
  assert.deepEqual(
    state.players.map((p) => p.stack),
    [160, 80]
  );
});

test("split pot: the odd chip goes to the first winner left of the button", () => {
  const again = rigged({ n: 3, dealer: 0, hands: ["As Kd", "Ah Kc", "7c 2d"], board: "Qs Jd Th 4c 5c" }).state;
  A(again, "raise", 5);
  A(again, "call");
  A(again, "fold"); // pot: 5 + 5 + 2 of dead money = 12
  for (let i = 0; i < 5; i++) A(again, "check");
  const final = A(again, "check");
  const award = ofType(final, "award")[0];
  assert.deepEqual(award.seats, [1, 0], "seat order starts left of the button");
  assert.equal(award.amount, 12);
  assert.deepEqual(award.shares, { 0: 6, 1: 6 });

  const odd = rigged({ n: 3, dealer: 0, stacks: [200, 200, 1], hands: ["As Kd", "Ah Kc", "7c 2d"], board: "Qs Jd Th 4c 5c" }).state;
  A(odd, "call"); // seat 2 is all-in for a 1-chip big blind; seat 0 calls 2
  A(odd, "call"); // seat 1 completes: invested 2, 2, 1 -> main pot 3 three ways, side pot 2
  for (let i = 0; i < 5; i++) A(odd, "check");
  const last = A(odd, "check");
  const aw = ofType(last, "award");
  assert.deepEqual(aw[0].shares, { 1: 2, 0: 1 }, "three chips split two ways: the extra one goes to seat 1");
  assert.deepEqual(aw[1].shares, { 0: 1, 1: 1 });
});

test("a busted player is rebought at the next hand", () => {
  const { state } = rigged({ n: 2, stacks: [200, 200], hands: ["As Ah", "Ks Kh"], board: "2c 7d 9h Jc 3s" });
  A(state, "raise", 200);
  A(state, "call");
  assert.deepEqual(
    state.players.map((p) => p.stack),
    [400, 0]
  );
  const ev = startHand(state, makeRng(2));
  assert.deepEqual(ev[0].rebuys, [1]);
  assert.equal(state.rebuys, 1);
  assert.equal(chips(state), 2 * BUYIN + BUYIN);
  assert.equal(state.dealer, 1);
});

test("replay reproduces the hand at every step", () => {
  const state = createGame({ n: 4 });
  startHand(state, makeRng(11));
  const snap = snapshot(state);
  const script = [["raise", 6], ["call"], ["fold"], ["call"], ["check"], ["raise", 10], ["call"], ["fold"], ["check"], ["check"], ["raise", 30], ["call"]];
  const live = [];
  const steps = [JSON.stringify(state)];
  for (const [type, to] of script) {
    live.push(...A(state, type, to));
    steps.push(JSON.stringify(state));
  }
  assert.equal(state.handOver, true);
  for (let k = 0; k <= script.length; k++) assert.equal(JSON.stringify(replay(snap, state.actions, k).state), steps[k], "step " + k);
  assert.deepEqual(replay(snap, state.actions).events, live);
  assert.equal(snap.actions.length, 0, "replay must not touch the snapshot");
  assert.equal(pot(replay(snap, state.actions, 1).state), 3 + 6);
  assert.equal(BB, 2);
});
