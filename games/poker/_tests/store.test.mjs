import test from "node:test";
import assert from "node:assert/strict";
import { createStore, DEFAULT_SETTINGS } from "../js/ui/store.js";

function fakeStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return { getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => data.set(k, String(v)), removeItem: (k) => data.delete(k), data };
}
const decision = (category, grade) => ({ category, grade, action: "call", street: 1, handNo: 1, lessonId: "odds" });
const tick = () => new Promise((r) => setTimeout(r, 0));

test("defaults from empty storage", () => {
  const s = createStore(fakeStorage());
  assert.deepEqual(s.settings, DEFAULT_SETTINGS);
  assert.deepEqual(s.progress, { mode: "home", level: "guided", done: {} });
  assert.equal(s.stats.guided.hands, 0);
  assert.equal(s.stats.silent.recent.length, 0);
});

test("junk in storage never throws and falls back to defaults", () => {
  for (const junk of ["", "{", "null", "[]", '"x"', '{"speed":{}}', "42"]) {
    const s = createStore(fakeStorage({ "pokerTrainer.prefs": junk, "pokerTrainer.v2": junk, "pokerTrainer.stats1": junk }));
    assert.deepEqual(s.settings, DEFAULT_SETTINGS, junk);
    assert.equal(s.progress.mode, "home");
    assert.equal(s.stats.guided.decisions, 0);
  }
  const partial = createStore(fakeStorage({ "pokerTrainer.prefs": '{"players":99,"speed":"fast","fourColour":true}' }));
  assert.deepEqual(
    [partial.settings.players, partial.settings.speed, partial.settings.fourColour],
    [6, 1, true],
    "bad values dropped, good ones kept"
  );
});

test("settings round-trip", () => {
  const st = fakeStorage();
  const a = createStore(st);
  a.setSetting("players", 9);
  a.setSetting("pause", "always");
  a.setSetting("muted", false);
  const b = createStore(st);
  assert.deepEqual([b.settings.players, b.settings.pause, b.settings.muted], [9, "always", false]);
  assert.throws(() => a.setSetting("nonsense", 1));
  a.setSetting("players", 40);
  assert.equal(a.settings.players, 9, "an invalid value is refused");
});

test("the old trainer's prefs and progress migrate", () => {
  const old = fakeStorage({
    "pokerTrainer.prefs": JSON.stringify({ speed: "250", anim: false, pauseMode: "all", showPick: false, nPlayers: "4" }),
    "pokerTrainer.v2": JSON.stringify({ mode: "coach", lastMode: "coach", done: { cards: true, outs: true } }),
  });
  const s = createStore(old);
  assert.deepEqual([s.settings.players, s.settings.animations, s.settings.pause, s.settings.showPick], [4, false, "always", false]);
  assert.ok(s.settings.speed > 1, "a short bot delay becomes a faster table");
  assert.deepEqual(s.progress, { mode: "table", level: "silent", done: { cards: true, outs: true } });
  assert.equal(createStore(fakeStorage({ "pokerTrainer.v2": '{"mode":"trainer"}' })).progress.level, "guided");
  assert.equal(createStore(fakeStorage({ "pokerTrainer.v2": '{"mode":"tutorial"}' })).progress.mode, "learn");
});

test("decisions are scored per level, with streaks and a capped history", () => {
  const st = fakeStorage();
  const s = createStore(st);
  s.recordDecision("guided", decision("Facing a bet", "correct"));
  s.recordDecision("guided", decision("Facing a bet", "correct"));
  s.recordDecision("guided", decision("Facing a bet", "acceptable"));
  assert.equal(s.stats.guided.streak, 2, "acceptable neither extends nor breaks a streak");
  s.recordDecision("guided", decision("Open or fold", "mistake"));
  const g = s.stats.guided;
  assert.deepEqual([g.decisions, g.correct, g.acceptable, g.mistakes, g.streak], [4, 2, 1, 1, 0]);
  assert.deepEqual(g.byCat["Facing a bet"], { n: 3, score: 2.5, mistakes: 0 });
  assert.equal(s.stats.silent.decisions, 0);
  for (let i = 0; i < 60; i++) s.recordDecision("silent", decision("Bet or check", "correct"));
  assert.equal(s.stats.silent.recent.length, 40);
  assert.equal(s.stats.silent.recent[0].grade, "correct");
  s.recordHand("silent", 37);
  s.recordHand("silent", -12);
  assert.deepEqual([s.stats.silent.hands, s.stats.silent.net], [2, 25]);
  assert.equal(createStore(st).stats.silent.decisions, 60, "persisted");
});

test("weakest category needs enough evidence", () => {
  const s = createStore(fakeStorage());
  for (let i = 0; i < 10; i++) s.recordDecision("silent", decision("Facing a bet", i < 8 ? "correct" : "mistake"));
  for (let i = 0; i < 10; i++) s.recordDecision("silent", decision("Bet or check", i < 5 ? "correct" : "mistake"));
  for (let i = 0; i < 3; i++) s.recordDecision("silent", decision("Facing a 3-bet", "mistake"));
  const w = s.weakest("silent");
  assert.deepEqual([w.category, w.accuracy, w.n, w.lessonId], ["Bet or check", 0.5, 10, "betting"]);
  assert.equal(s.weakest("guided"), null);
});

test("storage that throws is survivable", () => {
  const broken = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {
      throw new Error("quota");
    },
    removeItem() {},
  };
  const s = createStore(broken);
  s.setSetting("players", 3);
  s.recordDecision("guided", decision("Open or fold", "correct"));
  s.completeLesson("cards");
  assert.equal(s.settings.players, 3);
  assert.equal(s.progress.done.cards, true);
  assert.doesNotThrow(() => createStore(undefined));
});

test("view updates are batched; unsubscribe works", async () => {
  const s = createStore(fakeStorage());
  let calls = 0;
  const off = s.subscribe(() => calls++);
  s.set({ phase: "hero" });
  s.set({ raiseTo: 12 });
  s.set({ raiseTo: 14 });
  assert.equal(calls, 0);
  await tick();
  assert.equal(calls, 1);
  assert.deepEqual([s.view.phase, s.view.raiseTo], ["hero", 14]);
  off();
  s.set({ phase: "bots" });
  await tick();
  assert.equal(calls, 1);
  s.reset();
  assert.equal(s.stats.guided.decisions, 0);
});
