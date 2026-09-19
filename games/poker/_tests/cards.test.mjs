import test from "node:test";
import assert from "node:assert/strict";
import { makeRng, randInt } from "../js/core/rng.js";
import { newDeck, shuffle, cardKey, cardStr, parseCards } from "../js/core/cards.js";

test("rng is deterministic per seed and in [0,1)", () => {
  const a = makeRng(7),
    b = makeRng(7),
    c = makeRng(8);
  const xs = Array.from({ length: 1000 }, () => a());
  assert.deepEqual(
    xs.slice(0, 50),
    Array.from({ length: 50 }, () => b())
  );
  assert.notEqual(xs[0], c());
  assert.ok(xs.every((x) => x >= 0 && x < 1));
});
test("randInt covers its range uniformly enough", () => {
  const r = makeRng(1),
    n = [0, 0, 0, 0];
  for (let i = 0; i < 40000; i++) n[randInt(r, 4)]++;
  for (const k of n) assert.ok(Math.abs(k - 10000) < 400);
});
test("deck has 52 distinct cards; shuffle permutes deterministically", () => {
  const d = newDeck();
  assert.equal(new Set(d.map(cardKey)).size, 52);
  const s1 = shuffle(newDeck(), makeRng(3)).map(cardKey),
    s2 = shuffle(newDeck(), makeRng(3)).map(cardKey);
  assert.deepEqual(s1, s2);
  assert.notDeepEqual(s1, d.map(cardKey));
  assert.equal(new Set(s1).size, 52);
});
test("parseCards round-trips and rejects junk", () => {
  assert.deepEqual(parseCards("As Td 2c"), [
    { r: 14, s: 0 },
    { r: 10, s: 2 },
    { r: 2, s: 3 },
  ]);
  assert.equal(cardStr(parseCards("Kh")[0]), "K♥");
  assert.throws(() => parseCards("1x"));
});
