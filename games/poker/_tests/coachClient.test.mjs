import test from "node:test";
import assert from "node:assert/strict";
import { createCoach } from "../js/worker/coachClient.js";
import { analyze, makeRng, createReads, observeAll } from "../js/core/index.js";
import { rigged, A } from "./helpers/table.mjs";

// A stand-in for Worker. `script(message, reply)` decides what the "worker" does with each message.
function fakeWorker(script) {
  const w = {
    posted: [],
    terminated: false,
    postMessage(m) {
      w.posted.push(m);
      setTimeout(() => !w.terminated && script(m, (data) => w.onmessage?.({ data }), w), 0);
    },
    terminate() {
      w.terminated = true;
    },
  };
  return w;
}
const polite = (m, reply) =>
  m.type === "ping" ? reply({ type: "pong" }) : reply({ type: "result", id: m.id, analysis: { echo: m.seed, light: m.opts.light } });

function heroSpot() {
  const { state, events } = rigged({ n: 3, dealer: 0, hands: ["As Kd"] });
  const reads = createReads(3);
  observeAll(reads, events);
  return { state, reads };
}

test("requests go to the worker and answers come back to the right caller", async () => {
  const { state, reads } = heroSpot();
  let made;
  const coach = createCoach({ workerFactory: () => (made = fakeWorker(polite)) });
  const [a, b] = await Promise.all([coach.analyze(state, reads, { seed: 11, light: true }), coach.analyze(state, reads, { seed: 22, light: false })]);
  assert.deepEqual(
    [a, b],
    [
      { echo: 11, light: true },
      { echo: 22, light: false },
    ]
  );
  assert.equal(coach.mode, "worker");
  assert.equal(made.posted.filter((m) => m.type === "analyze").length, 2);
  assert.notEqual(made.posted[1].state, state, "the state is cloned, never shared");
  coach.dispose();
});

test("cancelAll rejects what is pending and ignores late answers", async () => {
  const { state, reads } = heroSpot();
  const held = [];
  const coach = createCoach({
    workerFactory: () =>
      fakeWorker((m, reply) =>
        m.type === "ping" ? reply({ type: "pong" }) : held.push(() => reply({ type: "result", id: m.id, analysis: "late" }))
      ),
  });
  const p = coach.analyze(state, reads, { seed: 1 });
  await new Promise((r) => setTimeout(r, 5));
  coach.cancelAll();
  await assert.rejects(p, (e) => e.cancelled === true);
  held.forEach((f) => f()); // the worker answers anyway: nothing should happen
  const q = coach.analyze(state, reads, { seed: 2 });
  await new Promise((r) => setTimeout(r, 5));
  held.at(-1)();
  assert.equal(await q, "late");
  coach.dispose();
});

test("falls back to the main thread when the worker cannot start, errors, or never answers", async () => {
  const { state, reads } = heroSpot();
  const want = analyze(state, reads, makeRng(7), { light: true, iters: 800 });
  const factories = {
    "throws on construction": () => {
      throw new Error("no workers here");
    },
    "reports an error": () =>
      fakeWorker((m, reply) => (m.type === "ping" ? reply({ type: "pong" }) : reply({ type: "error", id: m.id, message: "boom" }))),
    "never answers the ping": () => fakeWorker(() => {}),
  };
  for (const [name, workerFactory] of Object.entries(factories)) {
    const coach = createCoach({ workerFactory, pingMs: 20 });
    const got = await coach.analyze(state, reads, { seed: 7, light: true, iters: 800 });
    assert.equal(coach.mode, "main", name);
    assert.deepEqual(got.rec, want.rec, name);
    assert.equal(got.eq, want.eq, name + ": same seed, same numbers");
    coach.dispose();
  }
});

test("a worker that is only slow to start is used once it answers", async () => {
  const { state, reads } = heroSpot();
  // A worker that is still loading answers nothing; once up, it works through its queue in order.
  let up = false;
  const queue = [];
  const answer = (m, reply) => (m.type === "ping" ? reply({ type: "pong" }) : reply({ type: "result", id: m.id, analysis: "from the worker" }));
  const release = () => ((up = true), queue.splice(0).forEach(([m, reply]) => answer(m, reply)));
  const coach = createCoach({ pingMs: 15, workerFactory: () => fakeWorker((m, reply) => (up ? answer(m, reply) : queue.push([m, reply]))) });
  const early = await coach.analyze(state, reads, { seed: 3, light: true, iters: 300 });
  assert.equal(coach.mode, "main");
  assert.ok(early.rec, "answered on the main thread while the worker was still loading");
  release();
  await new Promise((r) => setTimeout(r, 5));
  assert.deepEqual([coach.mode, coach.fallbackReason], ["worker", ""]);
  assert.equal(await coach.analyze(state, reads, { seed: 4 }), "from the worker");
  coach.dispose();
});

test("errors from the coach reach the caller", async () => {
  const { state, reads } = heroSpot();
  A(state, "raise", 6); // seat 0 opened; move on so that the hero is no longer to act
  const coach = createCoach({
    workerFactory: () => {
      throw new Error("x");
    },
  });
  await assert.rejects(coach.analyze(state, reads, { seed: 1, iters: 99999 }), /not to act/, "errors from the coach reach the caller");
  coach.dispose();
});
