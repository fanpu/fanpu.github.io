/* FlyJack sound effects: all synthesized with WebAudio (no audio files). Muted by default; the choice is remembered. */
(function () {
  'use strict';
  const KEY = 'flyjack.sound';
  let ctx = null;
  let noise = null;
  let enabled = false;
  try { enabled = localStorage.getItem(KEY) === 'on'; } catch (e) { enabled = false; }

  function context() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
      noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
      const d = noise.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }

  function envelope(node, t0, attack, hold, release, peak) {
    node.gain.setValueAtTime(0.0001, t0);
    node.gain.exponentialRampToValueAtTime(peak, t0 + attack);
    node.gain.setValueAtTime(peak, t0 + attack + hold);
    node.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + hold + release);
  }

  function noiseBurst({ duration, filter, freq, q = 1, peak = 0.3, delay = 0 }) {
    const c = context();
    if (!c || !enabled) return;
    const t0 = c.currentTime + delay;
    const src = c.createBufferSource();
    src.buffer = noise;
    const f = c.createBiquadFilter();
    f.type = filter;
    f.frequency.value = freq;
    f.Q.value = q;
    const g = c.createGain();
    envelope(g, t0, 0.004, duration * 0.3, duration * 0.7, peak);
    src.connect(f).connect(g).connect(c.destination);
    src.start(t0, Math.random() * 0.5);
    src.stop(t0 + duration + 0.05);
  }

  function tone({ freq, to = null, duration, type = 'sine', peak = 0.15, delay = 0 }) {
    const c = context();
    if (!c || !enabled) return;
    const t0 = c.currentTime + delay;
    const o = c.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(freq, t0);
    if (to) o.frequency.exponentialRampToValueAtTime(to, t0 + duration);
    const g = c.createGain();
    envelope(g, t0, 0.01, duration * 0.2, duration * 0.8, peak);
    o.connect(g).connect(c.destination);
    o.start(t0);
    o.stop(t0 + duration + 0.05);
  }

  window.FlyAudio = {
    get enabled() { return enabled; },
    setEnabled(on) {
      enabled = !!on;
      try { localStorage.setItem(KEY, enabled ? 'on' : 'off'); } catch (e) { /* storage unavailable */ }
      if (enabled) context();
    },
    cardSlide() { noiseBurst({ duration: 0.16, filter: 'bandpass', freq: 2400, q: 0.7, peak: 0.22 }); },
    cardFlip() { noiseBurst({ duration: 0.04, filter: 'highpass', freq: 3500, peak: 0.35 }); },
    chip(n = 1) {
      for (let i = 0; i < Math.min(n, 5); i++) {
        tone({ freq: 3200 + 400 * Math.random(), duration: 0.05, type: 'triangle', peak: 0.08, delay: i * 0.06 });
        noiseBurst({ duration: 0.03, filter: 'highpass', freq: 5000, peak: 0.12, delay: i * 0.06 });
      }
    },
    win() { [523.25, 659.25, 783.99].forEach((f, i) => tone({ freq: f, duration: 0.22, peak: 0.12, delay: i * 0.11 })); },
    lose() { tone({ freq: 220, to: 150, duration: 0.5, type: 'triangle', peak: 0.12 }); },
    push() { tone({ freq: 440, duration: 0.18, peak: 0.08 }); },
  };
})();
