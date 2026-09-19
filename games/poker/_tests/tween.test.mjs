import test from "node:test";
import assert from "node:assert/strict";
import { Animator, ease } from "../js/stage/tween.js";

test("a tween runs from 0 to exactly 1 and then resolves", async () => {
  const anim = new Animator();
  const seen = [];
  let done = false;
  const p = anim.tween(1, (k) => seen.push(k), ease.linear).then(() => (done = true));
  for (let i = 0; i < 3; i++) anim.step(250);
  await Promise.resolve();
  assert.equal(done, false);
  assert.equal(anim.busy, true);
  anim.step(250);
  await p;
  assert.deepEqual(seen, [0.25, 0.5, 0.75, 1]);
  assert.equal(anim.busy, false);
});

test("speed, skip and reduced motion all land on the final value", async () => {
  const fast = new Animator();
  fast.speed = 2;
  let k1 = 0;
  const p1 = fast.tween(1, (k) => (k1 = k), ease.linear);
  fast.step(500);
  await p1;
  assert.equal(k1, 1);

  const anim = new Animator();
  const ks = [0, 0];
  const both = Promise.all([anim.tween(5, (k) => (ks[0] = k)), anim.tween(9, (k) => (ks[1] = k))]);
  anim.step(16);
  anim.skip();
  await both;
  assert.deepEqual(ks, [1, 1]);

  const calm = new Animator();
  calm.reduced = true;
  let k3 = 0;
  const p3 = calm.tween(3, (k) => (k3 = k));
  calm.step(16);
  await p3;
  assert.equal(k3, 1);

  let k4 = -1;
  await new Animator().tween(0, (k) => (k4 = k));
  assert.equal(k4, 1, "a zero-length tween completes without a step");
});

test("wait resolves, and tweens started by a finishing tween are not lost", async () => {
  const anim = new Animator();
  let chained = 0;
  const p = anim.wait(0.1).then(() => anim.tween(0.1, (k) => (chained = k)));
  anim.step(100);
  await Promise.resolve();
  await Promise.resolve();
  anim.step(100);
  await p;
  assert.equal(chained, 1);
});

test("easings are anchored and monotonic", () => {
  for (const [name, f] of Object.entries(ease)) {
    assert.ok(Math.abs(f(0)) < 1e-9 && Math.abs(f(1) - 1) < 1e-9, name);
    if (name !== "outBack") for (let i = 1; i <= 100; i++) assert.ok(f(i / 100) >= f((i - 1) / 100) - 1e-12, name + " is monotonic");
  }
  assert.ok(Math.max(...Array.from({ length: 100 }, (_, i) => ease.outBack(i / 100))) > 1, "outBack overshoots");
});
