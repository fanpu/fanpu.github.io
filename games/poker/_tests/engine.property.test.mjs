import test from "node:test";
import assert from "node:assert/strict";
import { makeRng, randInt } from "../js/core/rng.js";
import { createGame, startHand, legalActions, apply, pot, snapshot, replay, BUYIN } from "../js/core/engine.js";

// Any legal action, chosen at random: the engine must hold its invariants whatever the players do.
function randomAction(state, rng) {
  const legal = legalActions(state);
  const x = rng();
  if (legal.canRaise && x < 0.35) {
    const big = rng() < 0.15; // shove now and then so side pots and short all-ins come up often
    return { seat: legal.seat, type: "raise", to: big ? legal.maxTo : legal.minTo + randInt(rng, Math.min(legal.maxTo - legal.minTo, 40) + 1) };
  }
  if (x > 0.85 && !legal.canCheck) return { seat: legal.seat, type: "fold" };
  return { seat: legal.seat, type: legal.canCheck ? "check" : "call" };
}

function playSession(n, seed, hands, check) {
  const rng = makeRng(seed);
  const state = createGame({ n });
  const stream = [];
  for (let h = 0; h < hands; h++) {
    const events = startHand(state, rng);
    const snap = snapshot(state);
    let steps = 0;
    check?.(state, events, "start");
    while (!state.handOver) {
      const ev = apply(state, randomAction(state, rng));
      events.push(...ev);
      check?.(state, ev, "step");
      assert.ok(++steps <= 40 * n, "hand did not terminate");
    }
    check?.(state, events, "end", snap);
    stream.push(events);
  }
  return stream;
}

for (let n = 2; n <= 9; n++) {
  test(`invariants hold over 400 random hands, ${n}-handed`, () => {
    playSession(n, 1000 + n, 400, (state, events, phase, snap) => {
      const total = state.players.reduce((s, p) => s + p.stack, 0) + pot(state);
      assert.equal(total, (n + state.rebuys) * BUYIN, "chips are conserved");
      for (const p of state.players) {
        assert.ok(p.stack >= 0 && p.bet >= 0 && p.invested >= 0 && p.bet <= p.invested || state.handOver, "no negative or inconsistent amounts");
        assert.ok(Number.isInteger(p.stack));
        assert.equal(p.allIn && !state.handOver ? p.stack : 0, 0, "all-in means no chips behind");
      }
      if (!state.handOver) {
        const legal = legalActions(state);
        const actor = state.players[state.toAct];
        assert.equal(legal.seat, state.toAct);
        assert.ok(!actor.folded && !actor.allIn && actor.stack > 0);
        assert.ok(legal.toCall >= 0 && legal.toCall <= actor.stack);
        if (legal.canRaise) assert.ok(legal.minTo > state.currentBet && legal.minTo <= legal.maxTo);
      }
      if (phase === "end") {
        const end = events.at(-1);
        assert.equal(end.type, "handEnd");
        assert.equal(Object.values(end.net).reduce((a, b) => a + b, 0), 0, "net sums to zero");
        for (const p of state.players) assert.equal(end.net[p.id], p.stack - p.startStack);
        // Every chip that went in comes out as an award or a refund.
        const put = events.filter((e) => e.type === "post" || e.type === "action").reduce((s, e) => s + (e.amount ?? e.added), 0);
        const out = events.filter((e) => e.type === "award" || e.type === "refund").reduce((s, e) => s + e.amount, 0);
        assert.equal(out, put);
        for (const a of events.filter((e) => e.type === "award")) assert.equal(Object.values(a.shares).reduce((x, y) => x + y, 0), a.amount);
        assert.equal(state.board.length <= 5, true);
        const sd = events.find((e) => e.type === "showdown");
        if (sd) assert.equal(state.board.length, 5, "a showdown always has a full board");
        // Replay from the hand-start snapshot lands on the same state and the same events.
        const again = replay(snap, state.actions);
        assert.deepEqual(again.state, state);
        assert.deepEqual(again.events, events.slice(events.findIndex((e) => e.type === "deal") + 1));
      }
    });
  });
}

test("the same seed gives the same session", () => {
  assert.deepEqual(playSession(6, 42, 150), playSession(6, 42, 150));
  assert.notDeepEqual(playSession(6, 42, 5), playSession(6, 43, 5));
});

test("no card is ever dealt twice", () => {
  for (const hand of playSession(9, 7, 200)) {
    const keys = [];
    for (const e of hand) {
      if (e.type === "deal") for (const cs of Object.values(e.hands)) keys.push(...cs.map((c) => c.r * 4 + c.s));
      if (e.type === "street") keys.push(...e.cards.map((c) => c.r * 4 + c.s));
    }
    assert.equal(new Set(keys).size, keys.length);
  }
});
