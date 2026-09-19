import { CATEGORY_LESSON } from "../core/coach.js";

// One place for everything the interface remembers: settings, progress through the lessons, the scorecards,
// and `view`, the live picture of the table that the session publishes and every panel draws from.
// Nothing here touches the DOM, so it is tested in node with a fake storage.
const KEYS = { prefs: "pokerTrainer.prefs", progress: "pokerTrainer.v2", stats: "pokerTrainer.stats1" };

export const DEFAULT_SETTINGS = { players: 6, speed: 1, animations: true, pause: "mistake", showPick: true, fourColour: false, muted: true };
const VALID = {
  players: (v) => Number.isInteger(v) && v >= 2 && v <= 9,
  speed: (v) => typeof v === "number" && v >= 0.5 && v <= 3,
  animations: (v) => typeof v === "boolean",
  pause: (v) => ["never", "mistake", "always"].includes(v),
  showPick: (v) => typeof v === "boolean",
  fourColour: (v) => typeof v === "boolean",
  muted: (v) => typeof v === "boolean",
};
const emptyStats = () => ({ hands: 0, net: 0, decisions: 0, correct: 0, acceptable: 0, mistakes: 0, streak: 0, byCat: {}, recent: [] });
const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

export function createStore(storage) {
  if (storage === undefined) {
    try {
      storage = globalThis.localStorage;
    } catch {
      storage = null; // some browsers throw on access when storage is blocked
    }
  }
  const memory = {};
  const read = (key) => {
    try {
      const v = JSON.parse(storage.getItem(key));
      return isObject(v) ? v : {};
    } catch {
      return isObject(memory[key]) ? memory[key] : {};
    }
  };
  const write = (key, value) => {
    memory[key] = value;
    try {
      storage.setItem(key, JSON.stringify(value));
    } catch {
      /* private mode or a full quota: carry on in memory */
    }
  };

  // Settings. The old single-file trainer stored the same ideas under different names; read those too.
  const raw = read(KEYS.prefs);
  const legacy = {};
  if ("nPlayers" in raw) legacy.players = parseInt(raw.nPlayers, 10);
  if ("anim" in raw) legacy.animations = raw.anim;
  if ("pauseMode" in raw) legacy.pause = { all: "always", mistake: "mistake", none: "never", off: "never" }[raw.pauseMode];
  if (typeof raw.speed === "string") {
    const delay = parseInt(raw.speed, 10); // it was the bots' thinking time in ms
    if (Number.isFinite(delay)) legacy.speed = delay >= 1000 ? 0.75 : delay >= 500 ? 1 : delay >= 200 ? 1.6 : 2.5;
  }
  const settings = { ...DEFAULT_SETTINGS };
  for (const [k, ok] of Object.entries(VALID)) for (const src of [legacy, raw]) if (ok(src[k])) settings[k] = src[k];

  const p = read(KEYS.progress);
  const progress = { mode: "home", level: "guided", done: {} };
  if (p.mode === "trainer") Object.assign(progress, { mode: "table", level: "guided" });
  else if (p.mode === "coach") Object.assign(progress, { mode: "table", level: "silent" });
  else if (p.mode === "tutorial" || p.mode === "learn") progress.mode = "learn";
  else if (p.mode === "table") progress.mode = "table";
  if (p.level === "guided" || p.level === "silent") progress.level = p.level;
  if (isObject(p.done)) for (const [id, v] of Object.entries(p.done)) if (v === true) progress.done[id] = true;

  const s = read(KEYS.stats);
  const stats = { guided: emptyStats(), silent: emptyStats() };
  for (const level of ["guided", "silent"]) {
    const src = isObject(s[level]) ? s[level] : {};
    for (const k of ["hands", "net", "decisions", "correct", "acceptable", "mistakes", "streak"]) if (Number.isFinite(src[k])) stats[level][k] = src[k];
    if (isObject(src.byCat)) for (const [cat, b] of Object.entries(src.byCat)) if (isObject(b) && Number.isFinite(b.n)) stats[level].byCat[cat] = { n: b.n, score: +b.score || 0, mistakes: +b.mistakes || 0 };
    if (Array.isArray(src.recent)) stats[level].recent = src.recent.filter(isObject).slice(0, 40);
  }

  const subscribers = new Set();
  let scheduled = false;
  const notify = () => {
    if (scheduled) return;
    scheduled = true;
    queueMicrotask(() => {
      scheduled = false;
      for (const fn of [...subscribers]) fn(store);
    });
  };

  const store = {
    settings,
    progress,
    stats,
    view: { phase: "idle" },
    subscribe(fn) {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
    // Shallow-merge into the view. Several calls in one tick reach subscribers as a single update.
    set(patch) {
      Object.assign(store.view, patch);
      notify();
    },
    setSetting(key, value) {
      if (!VALID[key]) throw new Error("unknown setting " + key);
      if (!VALID[key](value)) return;
      settings[key] = value;
      write(KEYS.prefs, settings);
      notify();
    },
    setProgress(patch) {
      Object.assign(progress, patch);
      write(KEYS.progress, progress);
      notify();
    },
    completeLesson(id) {
      progress.done[id] = true;
      write(KEYS.progress, progress);
      notify();
    },
    // d: { category, grade: 'correct' | 'acceptable' | 'mistake', ...whatever the review wants to show }
    recordDecision(level, d) {
      const st = stats[level];
      st.decisions++;
      st[d.grade === "correct" ? "correct" : d.grade === "acceptable" ? "acceptable" : "mistakes"]++;
      if (d.grade === "correct") st.streak++;
      else if (d.grade === "mistake") st.streak = 0;
      const b = (st.byCat[d.category] ??= { n: 0, score: 0, mistakes: 0 });
      b.n++;
      b.score += d.grade === "correct" ? 1 : d.grade === "acceptable" ? 0.5 : 0;
      if (d.grade === "mistake") b.mistakes++;
      st.recent.unshift(d);
      st.recent.length = Math.min(st.recent.length, 40);
      write(KEYS.stats, stats);
      notify();
    },
    recordHand(level, netChips) {
      stats[level].hands++;
      stats[level].net += netChips;
      write(KEYS.stats, stats);
      notify();
    },
    // Where is this player leaking most? Only categories with enough decisions to mean something.
    weakest(level, minN = 8) {
      let worst = null;
      for (const [category, b] of Object.entries(stats[level].byCat)) {
        if (b.n < minN) continue;
        const accuracy = b.score / b.n;
        if (!worst || accuracy < worst.accuracy) worst = { category, accuracy, n: b.n, lessonId: CATEGORY_LESSON[category] || "equity" };
      }
      return worst;
    },
    reset() {
      stats.guided = emptyStats();
      stats.silent = emptyStats();
      write(KEYS.stats, stats);
      notify();
    },
  };
  return store;
}
