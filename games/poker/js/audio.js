// Table sounds, synthesized: no audio files. Muted by default; the AudioContext is only created once sound is
// switched on and the player has touched the page (browsers require that).
export function createAudio({ muted = true } = {}) {
  let ctx = null,
    noise = null,
    off = muted;

  function ready() {
    if (off) return null;
    if (!ctx) {
      const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
      noise = ctx.createBuffer(1, ctx.sampleRate * 0.5, ctx.sampleRate);
      const d = noise.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }
  // A burst of filtered noise: paper, felt, air.
  function hiss(at, { dur = 0.08, freq = 3000, type = "bandpass", q = 0.8, gain = 0.25 }) {
    const src = ctx.createBufferSource(),
      f = ctx.createBiquadFilter(),
      g = ctx.createGain();
    src.buffer = noise;
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    g.gain.setValueAtTime(gain, at);
    g.gain.exponentialRampToValueAtTime(0.0008, at + dur);
    src.connect(f).connect(g).connect(ctx.destination);
    src.start(at, Math.random() * 0.3, dur + 0.02);
  }
  // A short pitched tick: clay on clay, knuckles on wood, a bell.
  function tick(at, { freq = 1800, dur = 0.05, type = "triangle", gain = 0.2, drop = 0.6 }) {
    const o = ctx.createOscillator(),
      g = ctx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, at);
    o.frequency.exponentialRampToValueAtTime(freq * drop, at + dur);
    g.gain.setValueAtTime(gain, at);
    g.gain.exponentialRampToValueAtTime(0.0008, at + dur);
    o.connect(g).connect(ctx.destination);
    o.start(at);
    o.stop(at + dur + 0.02);
  }
  const chip = (at, gain = 0.16) => (
    tick(at, { freq: 2300 + Math.random() * 500, dur: 0.035, gain }), hiss(at, { dur: 0.03, freq: 5200, gain: gain * 0.6 })
  );

  const SOUNDS = {
    deal: (t) => [0, 0.07, 0.14].forEach((d) => hiss(t + d, { dur: 0.07, freq: 3400, gain: 0.2 })),
    flip: (t) => (hiss(t, { dur: 0.09, freq: 2400, gain: 0.22 }), tick(t + 0.07, { freq: 320, dur: 0.05, type: "sine", gain: 0.12 })),
    chip: (t) => [0, 0.045, 0.1].forEach((d) => chip(t + d)),
    chips: (t) => Array.from({ length: 7 }, (_, i) => chip(t + i * 0.04 + Math.random() * 0.02, 0.12)),
    allin: (t) => Array.from({ length: 12 }, (_, i) => chip(t + i * 0.035 + Math.random() * 0.02, 0.15)),
    check: (t) => [0, 0.11].forEach((d) => tick(t + d, { freq: 170, dur: 0.07, type: "sine", gain: 0.3, drop: 0.5 })),
    fold: (t) => hiss(t, { dur: 0.22, freq: 900, type: "lowpass", q: 0.4, gain: 0.16 }),
    win: (t) => [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tick(t + i * 0.085, { freq: f, dur: 0.5, type: "sine", gain: 0.13, drop: 1 })),
    right: (t) => [659.25, 987.77].forEach((f, i) => tick(t + i * 0.09, { freq: f, dur: 0.3, type: "sine", gain: 0.12, drop: 1 })),
    wrong: (t) => tick(t, { freq: 196, dur: 0.28, type: "triangle", gain: 0.14, drop: 0.8 }),
  };

  return {
    play(name) {
      const c = ready();
      if (c && SOUNDS[name]) SOUNDS[name](c.currentTime + 0.005);
    },
    setMuted(v) {
      off = v;
    },
    get muted() {
      return off;
    },
  };
}
