import test from "node:test";
import assert from "node:assert/strict";
import * as core from "../js/core/index.js";
import { createStore } from "../js/ui/store.js";
import { createSession } from "../js/session.js";

const fakeStorage = () => {
  const m = new Map();
  return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) };
};
const stubDirector = () => ({
  plays: 0,
  async play() {
    this.plays++;
  },
  show() {},
  async wait() {},
  skip() {},
  setSpeed() {},
});
const inlineCoach = () => ({ analyze: async (s, r, o) => core.analyze(s, r, core.makeRng(o.seed), { light: true, iters: 250 }), cancelAll() {} });

// Play `hands` hands with a hero who follows `policy(view)`. Returns what happened.
async function play({ level, hands, seed = 5, pause = "never", policy, players = 4 }) {
  const store = createStore(fakeStorage());
  store.setSetting("players", players);
  store.setSetting("pause", pause);
  const session = createSession({ store, director: stubDirector(), coach: inlineCoach(), rng: core.makeRng(seed) });
  const phases = [],
    log = [];
  let leaked = false,
    feedbacks = 0,
    done;
  const finished = new Promise((r) => (done = r));
  let last = null;
  store.subscribe(() => {
    const v = store.view;
    if (v.phase !== last) phases.push((last = v.phase));
    if (level === "silent" && v.phase !== "handOver" && (v.analysis || v.lastGrade || v.decisions.length)) leaked = true;
    if (v.phase === "hero" && v.legal) {
      const [type, to] = policy(v);
      log.push(`${v.handNo}:${v.street}:${type}${to ? "@" + to : ""}`);
      session.act(type, to);
    } else if (v.phase === "feedback") {
      feedbacks++;
      session.resume();
    } else if (v.phase === "handOver") {
      const total = session.state.players.reduce((s, p) => s + p.stack, 0);
      assert.equal(total, (players + session.state.rebuys) * core.BUYIN, "chips conserved");
      log.push("net " + v.result.net);
      if (v.handNo >= hands) done();
      else session.next();
    }
  });
  session.start(level);
  await finished;
  await session.stop();
  return { store, session, phases, log, leaked, feedbacks };
}
const callStation = (v) => [v.legal.canCheck ? "check" : "call"];
const aggressive = (v) => (v.legal.canRaise && v.street < 2 ? ["raise", v.legal.minTo] : [v.legal.canCheck ? "check" : "call"]);

test("guided: hands play to completion and are scored", async () => {
  const r = await play({ level: "guided", hands: 30, policy: callStation });
  const st = r.store.stats.guided;
  assert.equal(st.hands, 30);
  assert.ok(st.decisions >= 30 && st.decisions === st.correct + st.acceptable + st.mistakes);
  assert.equal(r.store.stats.silent.decisions, 0, "levels are scored apart");
  assert.equal(r.store.view.net, st.net);
  assert.ok(st.recent[0].category && st.recent[0].lessonId && st.recent[0].recommended);
  const legalNext = {
    idle: ["dealing"],
    dealing: ["bots", "hero", "handOver"],
    bots: ["hero", "bots", "handOver", "feedback"],
    hero: ["bots"],
    feedback: ["bots"],
    handOver: ["dealing"],
  };
  for (let i = 1; i < r.phases.length; i++) assert.ok(legalNext[r.phases[i - 1]].includes(r.phases[i]), `${r.phases[i - 1]} -> ${r.phases[i]}`);
});

test("silent: nothing is revealed until the hand is over, then everything is", async () => {
  const r = await play({ level: "silent", hands: 25, policy: aggressive, pause: "always" });
  assert.equal(r.leaked, false);
  assert.equal(r.feedbacks, 0, "silent never pauses, whatever the setting");
  assert.equal(r.store.stats.silent.hands, 25);
  assert.ok(r.store.view.phase === "handOver" && r.store.view.decisions.every((d) => d.analysis && d.grade));
});

test("guided pauses for feedback when asked to", async () => {
  const always = await play({ level: "guided", hands: 8, policy: callStation, pause: "always" });
  assert.ok(always.feedbacks > 0);
  const onMistake = await play({ level: "guided", hands: 8, policy: callStation, pause: "mistake" });
  assert.ok(onMistake.feedbacks <= always.feedbacks);
  assert.ok(onMistake.feedbacks <= onMistake.store.stats.guided.mistakes + onMistake.store.stats.guided.acceptable);
});

test("the same seed gives the same session", async () => {
  const a = await play({ level: "guided", hands: 12, seed: 9, policy: aggressive });
  const b = await play({ level: "guided", hands: 12, seed: 9, policy: aggressive });
  assert.deepEqual(a.log, b.log);
  assert.notDeepEqual(a.log, (await play({ level: "guided", hands: 12, seed: 10, policy: aggressive })).log);
});

test("leaving mid-hand parks the hand; coming back resumes it", async () => {
  const store = createStore(fakeStorage());
  store.setSetting("players", 3);
  const session = createSession({ store, director: stubDirector(), coach: inlineCoach(), rng: core.makeRng(3) });
  const until = (phase) =>
    new Promise((r) => {
      const off = store.subscribe(() => store.view.phase === phase && (off(), r()));
    });
  session.start("guided");
  await until("hero");
  const handNo = store.view.handNo,
    actions = session.state.actions.length;
  await session.stop();
  assert.equal(session.act("fold"), false, "a parked table ignores input");
  assert.equal(session.state.actions.length, actions);
  store.set({ phase: "idle" });
  session.start("silent");
  await until("hero");
  assert.deepEqual([store.view.handNo, store.view.level], [handNo, "silent"]);
  assert.equal(session.act("fold"), true); // folding guarantees the bots play the hand out without us
  await until("handOver");
  assert.equal(store.view.handNo, handNo);
  assert.equal(store.stats.silent.hands, 1, "the hand is scored at the level it finished at");
  await session.stop();
});
