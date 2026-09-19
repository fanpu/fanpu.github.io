// Time for the stage. Every motion is a tween on one Animator, stepped by the render loop (or by a test),
// and every tween is a promise, so choreography reads as plain async code: await deal; await flip.
export const ease = {
  linear: (t) => t,
  in: (t) => t * t * t,
  out: (t) => 1 - (1 - t) ** 3,
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2),
  outBack: (t) => 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2, // overshoots a little, then settles
};

export class Animator {
  constructor() {
    this.tweens = new Set();
    this.speed = 1;
    this.reduced = false; // prefers-reduced-motion, or animations switched off: everything lands at once
  }
  get busy() {
    return this.tweens.size > 0;
  }
  // update(k) is called with eased progress, and always finally with exactly 1.
  tween(seconds, update, easing = ease.inOut) {
    return new Promise((resolve) => {
      const tw = { elapsed: 0, ms: seconds * 1000, update, easing, resolve };
      if (seconds <= 0) return this.#finish(tw);
      this.tweens.add(tw);
    });
  }
  wait(seconds) {
    return this.tween(seconds, () => {}, ease.linear);
  }
  step(dtMs) {
    for (const tw of [...this.tweens]) {
      tw.elapsed += dtMs * this.speed;
      if (this.reduced || tw.elapsed >= tw.ms) this.#finish(tw);
      else tw.update(tw.easing(tw.elapsed / tw.ms));
    }
  }
  // Jump every running tween to its end. Anything the finished tweens go on to start runs normally.
  skip() {
    for (const tw of [...this.tweens]) this.#finish(tw);
  }
  #finish(tw) {
    this.tweens.delete(tw);
    tw.update(1);
    tw.resolve();
  }
}
