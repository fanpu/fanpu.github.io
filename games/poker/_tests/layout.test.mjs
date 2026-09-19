import test from "node:test";
import assert from "node:assert/strict";
import { TABLE, CARD, seatLayout, boardSlots, POT, MUCK, DECK, DENOMS, chipBreakdown, insideFelt } from "../js/stage/layout.js";

const dist = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);

for (let n = 2; n <= 9; n++) {
  test(`seat layout, ${n}-handed`, () => {
    const seats = seatLayout(n);
    assert.equal(seats.length, n);
    seats.forEach((s, i) => assert.equal(s.seat, i));
    const hero = seats[0];
    assert.ok(Math.abs(hero.pos.x) < 1e-9 && seats.every((s) => s.pos.z <= hero.pos.z + 1e-9), "the hero sits bottom centre");
    if (n > 2) assert.ok(seats[1].pos.x < 0, "seat 1 is on the hero's left: play runs clockwise seen from above");
    for (let i = 1; i < n; i++) assert.ok(seats[i].t > seats[i - 1].t);

    for (const s of seats) {
      assert.equal(s.cards.length, 2);
      for (const p of [...s.cards, s.bet, s.stack, s.button]) assert.ok(insideFelt(p, 0.5), `seat ${s.seat}: everything on the felt`);
      assert.ok(Math.hypot(s.bet.x, s.bet.z) < Math.hypot(s.cards[0].x, s.cards[0].z) || Math.abs(s.bet.x) < Math.abs(s.cards[0].x) + 1e-9, "bets go toward the middle");
      assert.ok(dist(s.bet, s.cards[0]) > 1.4 && dist(s.bet, s.cards[1]) > 1.4 && dist(s.stack, s.cards[0]) > 1.2 && dist(s.stack, s.cards[1]) > 1.2);
      assert.ok(dist(s.stack, s.bet) > 1.2 && dist(s.button, s.bet) > 1.0 && dist(s.button, s.stack) > 1.0);
      for (const fixed of [...boardSlots(), POT, MUCK, DECK]) assert.ok(dist(s.bet, fixed) > 1.5 && dist(s.cards[0], fixed) > 1.5 && dist(s.cards[1], fixed) > 1.5, `seat ${s.seat} clears the middle`);
    }
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const gap = Math.min(...seats[i].cards.flatMap((a) => seats[j].cards.map((b) => dist(a, b))));
        assert.ok(gap > 2.2 * CARD.w, `seats ${i} and ${j} are ${gap.toFixed(2)} apart`);
        assert.ok(dist(seats[i].bet, seats[j].bet) > 1.6, `bets of ${i} and ${j}`);
        assert.ok(dist(seats[i].stack, seats[j].cards[0]) > 1.2 && dist(seats[i].stack, seats[j].cards[1]) > 1.2);
      }
  });
}

test("the middle of the table", () => {
  const b = boardSlots();
  assert.equal(b.length, 5);
  for (let i = 1; i < 5; i++) assert.ok(b[i].x - b[i - 1].x > CARD.w * 1.05, "board cards do not touch");
  assert.ok(Math.abs(b[2].x) < 1e-9);
  for (const p of [POT, MUCK, DECK]) {
    assert.ok(insideFelt(p, 1));
    for (const s of b) assert.ok(dist(p, s) > 1.6);
  }
  assert.ok(dist(POT, MUCK) > 2 && dist(POT, DECK) > 2);
  assert.ok(TABLE.a > TABLE.b);
});

test("chip breakdown is exact and compact", () => {
  assert.deepEqual(chipBreakdown(0), []);
  assert.deepEqual(chipBreakdown(131), [
    { value: 100, count: 1 },
    { value: 25, count: 1 },
    { value: 5, count: 1 },
    { value: 1, count: 1 },
  ]);
  const values = DENOMS.map((d) => d.value);
  assert.deepEqual(
    values,
    values.slice().sort((a, b) => b - a)
  );
  for (let amount = 1; amount <= 60000; amount += amount < 2000 ? 1 : 97) {
    const parts = chipBreakdown(amount);
    assert.equal(
      parts.reduce((s, p) => s + p.value * p.count, 0),
      amount
    );
    assert.ok(parts.reduce((s, p) => s + p.count, 0) <= 30, "too many chips for " + amount);
    assert.ok(parts.every((p) => p.count > 0 && values.includes(p.value)));
  }
});
