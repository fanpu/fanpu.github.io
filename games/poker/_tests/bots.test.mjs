import test from "node:test";
import assert from "node:assert/strict";
import { makeRng } from "../js/core/rng.js";
import { createGame, startHand, apply, legalActions } from "../js/core/engine.js";
import { createReads, observeAll } from "../js/core/reads.js";
import { STYLES, assignStyles, botDecide } from "../js/core/bots.js";
import { rigged, A } from "./helpers/table.mjs";

const byName = (name) => STYLES.find((s) => s.name === name);

// Bots in every seat. styles[i] is the style for seat i.
function session(styles, hands, seed) {
  const n = styles.length,
    rng = makeRng(seed);
  const state = createGame({ n }),
    reads = createReads(n);
  let decisions = 0;
  for (let h = 0; h < hands; h++) {
    observeAll(reads, startHand(state, rng));
    while (!state.handOver) {
      const action = botDecide(state, reads, state.toAct, styles[state.toAct], rng, 120);
      observeAll(reads, apply(state, action)); // apply throws if the action is illegal
      decisions++;
    }
  }
  return { reads, decisions };
}

test("every decision is legal, at every table size", () => {
  for (let n = 2; n <= 9; n++) {
    const styles = Array.from({ length: n }, (_, i) => STYLES[i % STYLES.length]);
    assert.ok(session(styles, 60, 300 + n).decisions >= 60); // heads-up, a hand can be a single fold
  }
});

test("styles play the way their labels say", () => {
  const order = ["TAG", "Nit", "TAG", "LAG", "Station", "Maniac"];
  const { reads } = session(order.map(byName), 1500, 77);
  const stat = (name, key) => {
    const o = reads.players[order.lastIndexOf(name)].obs;
    return (100 * o[key]) / o.hands;
  };
  const vpip = ["Nit", "TAG", "LAG", "Station", "Maniac"].map((s) => stat(s, "vpip"));
  for (let i = 1; i < vpip.length; i++) assert.ok(vpip[i] > vpip[i - 1] + 2, "VPIP order: " + vpip.map((v) => v.toFixed(1)));
  const pfr = ["TAG", "LAG", "Maniac"].map((s) => stat(s, "pfr"));
  assert.ok(pfr[0] < pfr[1] && pfr[1] < pfr[2], "PFR order: " + pfr.map((v) => v.toFixed(1)));
  assert.ok(Math.max(stat("Nit", "pfr"), stat("Station", "pfr")) < pfr[0]);
  assert.ok(stat("Station", "vpip") - stat("Station", "pfr") > 20, "a station enters a lot and rarely raises");
});

test("premiums are never folded before the flop; trash is", () => {
  for (const style of STYLES) {
    const rng = makeRng(5);
    // Seat 3 holds aces: unopened, then facing a raise, then facing a 3-bet and a 4-bet.
    const unopened = rigged({ n: 6, hands: [null, null, null, "As Ah"] });
    assert.equal(botDecide(unopened.state, createReads(6), 3, style, rng).type, "raise");

    const t = rigged({ n: 6, dealer: 2, hands: [null, null, null, "As Ah"] }); // seats: 3 SB, 4 BB, 5 UTG, 0 HJ, 1 CO, 2 BTN
    const reads = createReads(6);
    observeAll(reads, t.events);
    observeAll(reads, A(t.state, "raise", 6)); // UTG opens
    observeAll(reads, A(t.state, "raise", 20)); // HJ 3-bets
    observeAll(reads, A(t.state, "raise", 50)); // CO 4-bets
    observeAll(reads, A(t.state, "fold"));
    assert.equal(t.state.toAct, 3);
    assert.notEqual(botDecide(t.state, reads, 3, style, rng).type, "fold", style.name + " folded aces");
  }
  const nit = rigged({ n: 3, hands: [null, "7s 2d"] });
  const reads = createReads(3);
  observeAll(reads, nit.events);
  observeAll(reads, A(nit.state, "raise", 6));
  assert.deepEqual(botDecide(nit.state, reads, 1, byName("Nit"), makeRng(1)), { seat: 1, type: "fold" });
  assert.throws(() => botDecide(nit.state, reads, 2, byName("Nit"), makeRng(1)), /not to act/);
});

test("a free check is never turned into a fold or a call", () => {
  const t = rigged({ n: 3, hands: [null, null, "7s 2d"] });
  const reads = createReads(3);
  observeAll(reads, t.events);
  observeAll(reads, A(t.state, "call"));
  observeAll(reads, A(t.state, "call"));
  assert.equal(legalActions(t.state).canCheck, true);
  assert.deepEqual(botDecide(t.state, reads, 2, byName("Nit"), makeRng(1)), { seat: 2, type: "check" });
});

test("style seating is shuffled per session but reproducible", () => {
  const a = assignStyles(6, makeRng(1));
  assert.equal(a[0], null);
  assert.equal(a.length, 6);
  assert.ok(a.slice(1).every((i) => i >= 0 && i < STYLES.length));
  assert.ok(new Set(a.slice(1)).size >= 4, "a six-handed table has at least four different types");
  assert.deepEqual(a, assignStyles(6, makeRng(1)));
  assert.notDeepEqual(
    [1, 2, 3, 4, 5, 6].map((s) => assignStyles(9, makeRng(s)).join()),
    Array(6).fill(assignStyles(9, makeRng(1)).join())
  );
  assert.deepEqual(assignStyles(2, makeRng(3)).length, 2);
});
