# Poker trainer M3: the playable table — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sit down and play: a turn loop with bots, an action dock, a HUD, the coach running in a Web Worker, the five-tab info panel, Guided and Silent coach levels, persisted settings and stats, on desktop and phone.

**Architecture:** `session.js` owns the hand loop (engine + reads + bots + coach + director) and publishes one plain `view` object through `store.js`; every panel in `ui/` is a function of that view and sends intents back (`act`, `next`, `continue`). Panels never touch the stage or the engine. The coach runs in a module worker behind `coachClient.js`, with a main-thread fallback. Panels claim screen space through the `--stage-*` variables and the camera refits.

**Tech Stack:** as M1/M2. No new dependencies.

**Spec:** `games/poker/_docs/2026-09-19-poker-trainer-3d-design.md` (sections "One table, a coach dial", "Coach computes off the main thread", "Smaller improvements", "The table is a region", "Settings and persistence"). Plan 3 of 6.

## Global Constraints

- `ui/` imports `core/`, `ui/store.js` and `stage/textures.js` (card painter) only. No `three`, no `director`, no `session`.
- All game randomness flows from one seeded rng (`?seed=` or time); the coach worker gets its own seed per request so results do not depend on timing.
- Keys: F fold, C check/call, R bet/raise, 1-4 size presets, Space/Enter next hand / continue / skip animation. Ignored while typing in an input and with ctrl/meta.
- localStorage keys: `pokerTrainer.prefs` (extended), `pokerTrainer.v2` (read and migrated), `pokerTrainer.stats1` (new). All reads tolerate junk. In-memory fallback when storage throws.
- Phone (max-width 720px): table on top, dock beneath with a DOM strip of the hero's hand and the board; info opens as a bottom sheet over the table. Touch targets at least 44px.
- Everything the hero needs to act is reachable without the 3D scene (DOM strip, labels, buttons), so assistive tech is not locked out.
- Formatting and commits as before; prefix `poker table:`.

---

### Task 1: Store, settings, persistence

**Files:** Create `js/ui/store.js`; Test `_tests/store.test.mjs`

**Interfaces — Produces:**
- `DEFAULT_SETTINGS = { players: 6, speed: 1, animations: true, pause: 'mistake', showPick: true, fourColour: false, muted: true }`
- `createStore(storage = globalThis.localStorage) -> { settings, progress, stats, view, set(patch), setSetting(key, value), subscribe(fn) -> unsubscribe, recordDecision(level, d), recordHand(level, netChips), completeLesson(id), reset() }`
  - `progress = { mode: 'home'|'learn'|'table', level: 'guided'|'silent', done: { [lessonId]: true } }`, migrated from v2 (`trainer` -> table/guided, `coach` -> table/silent, `tutorial` -> learn).
  - `stats = { guided: S, silent: S }`, `S = { hands, net, decisions, correct, acceptable, mistakes, streak, byCat: { [category]: { n, score, mistakes } }, recent: [ up to 40 decision summaries ] }`.
  - `weakest(level, minN = 8) -> { category, accuracy, n, lessonId } | null`.
  - `view` is the session's published state; `set` shallow-merges and notifies subscribers once per microtask.

- [x] **Step 1: Failing tests:** defaults with empty storage; junk JSON in every key falls back to defaults without throwing; settings round-trip; old prefs `{speed:'600',anim:true,pauseMode:'mistake',showPick:true,nPlayers:'6'}` migrate; each v2 mode migrates; `recordDecision` updates totals, byCat, streak (reset on mistake, unchanged on acceptable) and caps `recent` at 40; guided and silent are kept apart; `weakest` ignores categories under `minN` and returns the lowest accuracy with its lesson id; storage that throws on `setItem` does not throw out of the store; subscribers are called once for several `set` calls in a tick; unsubscribe works.
- [x] **Step 2:** Fail. **Step 3:** Implement. **Step 4:** Pass. Commit.

### Task 2: Coach worker

**Files:** Create `js/worker/coachWorker.js`, `js/worker/coachClient.js`; Test `_tests/coachClient.test.mjs`

**Interfaces — Produces:** `createCoach({ workerFactory? }) -> { analyze(state, reads, { seed, light, iters }) -> Promise<analysis>, cancelAll(), dispose(), mode: 'worker' | 'main' }`. Requests carry an id; `cancelAll` rejects pending promises with `{ cancelled: true }` and ignores late replies. If the worker cannot be constructed, errors, or does not answer a ping within 1.5 s, the client switches to main-thread `analyze` (with `iters` capped at 3000) and replays outstanding requests. `workerFactory` is injectable for tests.

- [x] **Step 1: Failing tests** with a fake worker: resolves with the worker's reply; ids keep concurrent requests apart; `cancelAll` rejects and late replies are dropped; a worker that throws on construction, or posts `error`, falls back to main and still resolves; the main-thread result for a seed equals `core.analyze` with `makeRng(seed)`.
- [x] **Step 2:** Fail. **Step 3:** Implement. **Step 4:** Pass. Commit.

### Task 3: Session (the hand loop)

**Files:** Create `js/session.js`; Test `_tests/session.test.mjs` (with a stub director and an inline coach)

**Interfaces — Consumes** core, `createCoach`, a director (`play`, `applyScene`, `skip`, `setSpeed`), store. **Produces:** `createSession({ store, director, coach, rng, wait }) -> { start(level), stop(), act(type, to?), next(), resume(), setLevel(level) }` and publishes through `store.set`:

```
view = { phase: 'idle'|'dealing'|'bots'|'hero'|'feedback'|'handOver', level, handNo, net,
         legal, hero: { cards, pos, stack }, board, pot, analysis | null, analysing: bool,
         raiseTo, lastGrade | null, result | null, decisions: [...this hand], opponents: [{ seat, name, pos, read, label }] }
```

Rules: analysis is requested the moment the hero is to act; in Guided the dock enables at once and the panels fill in when it lands; acting before it lands waits for it (grading needs it). Guided pauses per `settings.pause`; Silent never pauses and never publishes `analysis` or `lastGrade` until `handOver`. `stop()` parks the loop at the next await (leaving the table mid-hand pauses it; `start` resumes the same hand). Grading follows OLD `heroAct` (check/call equivalence, EV of the chosen action from `evOfRaise`).

- [x] **Step 1: Failing tests:** a scripted hero plays 30 hands to completion at both levels with chips conserved; phases occur in a legal order; Silent never exposes analysis mid-hand; pause-on-mistake enters `feedback` and `resume()` continues; `stop()` then `start()` resumes the same hand number; stats are recorded per level; identical seeds give identical decision logs.
- [x] **Step 2:** Fail. **Step 3:** Implement. **Step 4:** Pass. Commit.

### Task 4: Panels

**Files:** Create `js/ui/hud.js`, `js/ui/dock.js`, `js/ui/info.js`, `js/ui/rangeGrid.js`, `js/ui/settings.js`, `js/ui/home.js`, `js/ui/html.js` (tiny escaping/template helper), `css/panels.css`, `css/dock.css`; Modify `index.html`, `js/main.js`, `css/base.css`; Delete `css/mock.css` and the mock code in `main.js`

- `hud.js`: header (home, level switch Guided/Silent, hand number, session net, settings) and, in Guided, the stat row ported from OLD `hudHTML`.
- `dock.js`: hand-and-board strip (DOM cards from the shared painter, hand description), Fold / Check-Call / Bet-Raise with EV under each in Guided and the coach's pick marked, presets (Coach, Min, ½, ⅔, Pot, All-in), slider with live "all fold x%, win y% if called", feedback card, hand-over summary with Next hand, key hints on desktop.
- `info.js`: tabs Ranges / EV / Next card / Hand / Why, content ported from OLD (`infoRangesHTML`, `evTableHTML`, `nextCardsHTML`, `infoHandHTML`, `verdictHTML`), one renderer each (no compact/full duplicates). Desktop: right drawer. Phone: bottom sheet opened by the tab bar, closed by swipe-down, tap outside or Escape. Shows a computing state while `analysing`.
- `settings.js`: popover for players, speed, animations, pause, show coach's pick, four-colour deck, sound.
- `home.js`: the three steps (Learn / Train / Prove it) with progress and the weakest-category prompt. Learn is present but marked "arrives with the lessons" until M5.
- Panels set `--stage-top/right/bottom` from their measured sizes (ResizeObserver), so the camera always fits what is really there.

- [x] **Step 1:** Implement. **Step 2:** `shoot.mjs` gains `until=<phase>` (waits for `document.body.dataset.phase`) and click/keys scripting; capture hero-to-act (each tab), feedback, hand over, settings, home, at both sizes. **Step 3:** Review by eye, fix, repeat. Commit.

### Task 5: Play-test and tune

- [x] Drive 40 hands per level in headless Chrome with a scripted hero (`?auto=coach`), asserting no console errors and that `phase` never stalls more than 20 s.
- [x] Measure how often the hero faces a 3-bet after opening (M1 open item). If above roughly 25% six-handed, temper the LAG/Maniac 3-bet bluff frequency and re-run the bot style tests.
- [x] Check worker timing in the browser; raise `iters` while the median analysis stays under 400 ms.
- [x] Full test suite, format, commit.
