# Poker trainer M4: review and progress — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a hand ends, replay it: a timeline of every action with the hero's decisions graded, the 3D table rewinding to whichever moment is chosen, the coach's analysis for that moment, and the hero's equity street by street. On the home screen, a scorecard by decision category.

**Architecture:** The engine is event-sourced, so a moment in a hand is `replay(hand.snap, actions, k)`; the director already knows how to show any state at once. The session owns rewinding (`rewind(k)`), publishes a `timeline` and an `equityTrail` at hand-over, and the coach panel draws them. Equity per street is asked of the coach worker through a second request kind, so it costs the table nothing.

**Spec:** `2026-09-19-poker-trainer-3d-design.md`, "Hand replay in the review" and "Progress that persists and points somewhere". Plan 4 of 6.

## Global Constraints

As M3. Rewinding never mutates the live engine state or the hand's record. Opponents' hole cards are shown in a rewound view only if they were shown down in the real hand.

### Task 1: Equity trail through the worker

**Files:** Modify `js/worker/coachWorker.js`, `js/worker/coachClient.js`, `js/session.js`; Test `_tests/coachClient.test.mjs`, `_tests/session.test.mjs`

- `coach.equity(heroCards, board, ranges, { seed, iters = 4000 }) -> Promise<number>`, same id/cancel/fallback machinery as `analyze`.
- The session asks for it at the deal and at each new street while the hero is still in, and publishes `equityTrail: [{ street, eq }]` with the hand-over view (never earlier in Silent).
- [x] Failing tests (client: worker and main-thread paths agree for a seed; session: a trail entry per street the hero saw, in order, none after folding, absent from the view before hand-over in Silent). Implement. Pass. Commit.

### Task 2: Rewind

**Files:** Modify `js/session.js`; Test `_tests/session.test.mjs`

- At hand-over the view gains `timeline: [{ k, seat, name, pos, street, kind, to, added, hero, grade? }]`, one entry per action, `k` being the number of actions applied before it.
- `session.rewind(k | null)`: only at hand-over. `k` shows the table as it stood before action `k` (`replay(snap, actions, k)`), `null` returns to the finished hand with its showdown. Publishes `view.at = k | null`. Calls `director.show(state, { reveal })` where `reveal` is the hero plus any seat shown down.
- [x] Failing tests (timeline matches the engine's action events; `rewind(k)` hands the director a state with exactly `k` actions applied and the right player to act; live state untouched; `rewind` ignored mid-hand; `next()` after a rewind deals the next hand normally). Implement. Pass. Commit.

### Task 3: The replayer in the coach panel

**Files:** Modify `js/ui/info.js`, `css/panels.css`, `js/ui/dock.js`

- At hand-over the panel is headed by the equity trail (small bar per street) and the timeline, grouped by street; the hero's decisions carry their grade colour; the chosen moment is marked. Choosing a hero decision also shows its analysis in the tabs below; choosing anyone else's action shows the table reads. The first mistake (else the first decision) is preselected and the table rewinds to it when Review is opened.
- Left/Right arrows step through the timeline; Escape or "Back to the result" returns to the finished hand.
- [x] Implement. Capture desktop and phone with `shoot.mjs`. Review by eye. Commit.

### Task 4: Scorecard on the home screen

**Files:** Create `js/ui/scorecard.js`; Modify `js/ui/home.js`, `css/panels.css`

- Per level: decisions, agreement with the coach, net, and a row per category (spots, accuracy bar, mistakes), worst first; the last 40 decisions as a compact list (hand, street, category, what you did, what the coach preferred).
- [x] Implement with a node test for the pure row-building function (`scorecardRows(stats)`), capture, commit.
