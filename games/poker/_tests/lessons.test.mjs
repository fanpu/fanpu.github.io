import test from "node:test";
import assert from "node:assert/strict";
import { DRILLS, LESSONS, seedLessons, takeStaged } from "../js/lessons/content.js";
import { makeRng, cardKey } from "../js/core/index.js";

const LESSON_IDS = ["cards", "ranks", "flow", "outs", "odds", "equity", "start", "ranges", "texture", "betting", "defend", "cbet", "types"];

test("thirteen lessons in order, each drill page pointing at a real drill", () => {
  assert.deepEqual(
    LESSONS.map((l) => l.id),
    LESSON_IDS
  );
  const used = new Set();
  for (const l of LESSONS) {
    assert.ok(l.title && l.blurb && l.pages.length >= 2, l.id);
    for (const p of l.pages) {
      assert.ok(p.t, l.id + " page title");
      if (p.drill) {
        assert.equal(typeof DRILLS[p.drill], "function", p.drill);
        assert.ok(p.need >= 3 && p.need <= 8);
        used.add(p.drill);
      } else assert.equal(typeof p.h, "function");
    }
  }
  assert.equal(Object.keys(DRILLS).length, 16);
  assert.deepEqual([...used].sort(), Object.keys(DRILLS).sort(), "every drill is used by some lesson");
});

test("every lesson page renders, with no stray placeholders", () => {
  seedLessons(makeRng(3));
  for (const l of LESSONS)
    for (const p of l.pages.filter((x) => x.h)) {
      const html = p.h();
      assert.ok(typeof html === "string" && html.length > 80, `${l.id} / ${p.t}`);
      assert.ok(!/undefined|NaN|\[object/.test(html.replace(/<[^>]*>/g, " ")), `${l.id} / ${p.t} contains a bad value`);
    }
  takeStaged();
});

for (const name of Object.keys(DRILLS)) {
  test(`drill ${name}: one right answer, distinct options, a real explanation`, () => {
    seedLessons(makeRng(100 + name.length));
    for (let i = 0; i < 150; i++) {
      takeStaged();
      const d = DRILLS[name]();
      assert.ok(d.q.length > 10);
      assert.ok(d.options.length >= 2 && d.options.length <= 4, "options: " + d.options.length);
      assert.equal(new Set(d.options).size, d.options.length, "options are distinct: " + d.options.join(" | "));
      assert.ok(Number.isInteger(d.answer) && d.answer >= 0 && d.answer < d.options.length, "answer index " + d.answer);
      assert.ok(d.explain.length > 30 && !/undefined|NaN/.test(d.explain.replace(/<[^>]*>/g, " ")));
      // Cards shown in one question never repeat.
      const rows = takeStaged();
      if (d.visual && rows.length) {
        const keys = rows[0].flatMap((g) => g.cards.map(cardKey));
        assert.equal(new Set(keys).size, keys.length, "a card appears twice in the question");
      }
    }
  });
}

test("drills are reproducible from a seed", () => {
  const run = (seed) => {
    seedLessons(makeRng(seed));
    return Object.keys(DRILLS).map((k) => {
      const d = DRILLS[k]();
      return d.q + "|" + d.options.join(",") + "|" + d.answer;
    });
  };
  assert.deepEqual(run(5), run(5));
  assert.notDeepEqual(run(5), run(6));
  takeStaged();
});
