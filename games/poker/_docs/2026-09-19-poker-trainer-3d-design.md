# Poker trainer: 3D rebuild — design

Date: 2026-09-19
Status: awaiting review (rev 2: parity requirement dropped)

## Vision

Take someone from "what beats what" to making sound no-limit hold'em decisions,
in three steps that share one table:

1. **Learn**: short lessons, each immediately drilled until it sticks.
2. **Train**: play real hands against varied opponents with a coach that shows
   its working: ranges, equity, EV per action, and why.
3. **Prove it**: play with the coach silent, then get an honest review.

The rebuild stays faithful to that vision, not to the current implementation.
The current single-file version is reference material: its lesson prose, drill
ideas, bot styles and coach heuristics are good and are carried over where they
serve the vision, rewritten where they do not.

## Goal

Rebuild `games/poker/` from a single 227 KB `index.html` into a multi-file
project with a three.js table in the same family as `games/flyjack/`, and use
the rebuild to make the whole experience better, not only prettier.

Success means:

1. `/games/poker/` loads at the same URL with no build step.
2. The table is a lit 3D scene with animated cards and chips, and lessons,
   training and review all happen on it.
3. The poker logic is demonstrably correct (tests below), and the render loop
   stays smooth while the coach is computing.
4. Existing users keep their lesson progress and table preferences.

## Decisions already made

| Question     | Decision                                                      |
| ------------ | ------------------------------------------------------------- |
| Rendering    | Full 3D, three.js. DOM glass panels layered over a canvas     |
| Scope        | Faithful to the vision; free to change behaviour and features |
| Identity     | Sibling of flyjack with its own felt and accent               |
| Code shipped | Native ES modules, no bundler, three.js vendored locally      |

## Non-goals

- No backend, accounts, multiplayer or solver data.
- No build step, TypeScript or framework.
- No 2D fallback renderer for the table.
- No new game variants (tournaments, PLO, antes-only formats).

## Experience changes

These are the sweeping changes, in priority order.

### 1. Lessons happen on the table

Today lessons are a DOM page with pictures of cards. In the rebuild the lesson
text lives in a glass side panel and the 3D table is the illustration, driven
by the lesson:

- "Hand ranks", "who wins": hands are dealt on the felt; the winning five
  rise and glow using the same showdown treatment as real play.
- "Position": seats light up in order of action; the button slides.
- "Outs": the unseen cards that complete the draw fan out above the board.
- "Pot odds", "bet sizing", "MDF": real chip stacks form the pot and the bet.
- "Board texture", "c-bet": flops are dealt and the relevant cards pulse.
- Drills pose their question on the table and answer buttons sit in the dock
  where action buttons sit in play, so the learner's hands are already in the
  right place when they graduate to Train.

All 13 lesson topics and 16 drill types are kept. Prose is carried over and
edited for the new staging. Visuals that are genuinely tabular (the 13x13
range grid, EV tables) stay DOM, inside the panel.

### 2. One table, a coach dial

Trainer and Coaching are the same table with hints on or off, implemented
today as two modes plus a body class. They become one **Table** with a coach
level, keeping the Train -> Prove it journey:

- **Guided**: live HUD numbers, info drawer, coach's pick highlighted, EV per
  bet size, pause-after-decision grading.
- **Silent**: no live help; decisions are graded quietly and revealed in the
  post-hand review. This is the score the home screen calls the honest one.

Home still presents three steps (Learn / Train / Prove it); the last two open
the table at the corresponding coach level. Scores are tracked separately per
level so Silent stays honest.

### 3. Hand replay in the review

Because the engine is event-sourced (below), the post-hand review becomes a
replayer: a timeline of the hand with the hero's decisions marked and graded.
Clicking a decision rewinds the 3D table to that moment and shows the coach's
analysis for it: ranges, equity, EV, why. The equity-by-street trail stays,
drawn above the timeline. The first mistake is preselected.

### 4. Progress that persists and points somewhere

Scorecard and decision history persist. Home shows accuracy by category and
surfaces the weakest one with a direct link to its lesson and drill ("You are
leaking most on c-bets: 58% over 40 spots. Practise"). No new content is
needed; it reuses the existing category -> lesson mapping.

### 5. Coach computes off the main thread

Monte Carlo equity and EV sampling move to a module Web Worker so the scene
holds frame rate while the coach thinks. The info drawer shows a brief
computing state, then fills in. With more time budget available, sample counts
go up, so numbers are steadier than today.

### 6. Smaller improvements

- Keyboard play: F fold, C check/call, R raise, 1-4 bet-size presets, Space
  next hand / skip animation, arrow keys step the replay.
- `?seed=N` reproduces a session exactly.
- Optional four-colour deck.
- Hover or tap a seat for that opponent's observed stats and inferred style.
- Sound, muted by default.

## Architecture

```
games/poker/
  index.html          thin shell: canvas mount, empty panel slots, analytics,
                      <script type="module" src="js/main.js">
  css/
    base.css          design tokens, glass panel primitives, buttons
    hud.css           header, seat labels, decision dock
    panels.css        info drawer, review, stats, settings, home
    lessons.css       lesson panel, drills, DOM cards, range grid
  vendor/
    three.module.min.js   three 0.186.0, vendored from npm, no CDN
  js/
    core/             PURE: no DOM, no three.js; runs in node and in a worker
      rng.js          seeded mulberry32; the only entropy source
      cards.js        card encoding, deck, shuffle
      eval.js         7-card evaluator, hand description, best-five extraction
      preflop.js      range tiers, hand percentiles
      positions.js    position names, opening and defending percentages
      equity.js       Monte Carlo equity vs ranges
      analysis.js     outs, hand class, board texture, combos, range
                      composition, next-card map, EV sampling
      engine.js       event-sourced hold'em engine (below)
      reads.js        what an observer infers from engine events: VPIP/PFR,
                      assumed ranges, the story of the hand
      bots.js         five styles, decisions
      coach.js        analyze, recommend, grade, category -> lesson
    worker/
      coachWorker.js  runs equity/analysis/coach off-thread
      coachClient.js  promise API over the worker, with cancellation
    stage/            three.js ONLY: knows nothing about poker rules
      stage.js        renderer, scene, lights, fog, RAF loop, resize,
                      adaptive quality, context-loss recovery
      tween.js        promise-based Animator, easings, skip(), speed scale
      textures.js     procedural painters: felt, card faces and back, chips
      table.js        table and rail meshes, seat layout for 2-9 players,
                      dealer button
      cards3d.js      pooled card meshes: deal, flip, muck, reveal, highlight,
                      fan
      chips3d.js      instanced chip stacks: bet, sweep to pot, award
      shots.js        named camera shots and orbit/zoom controls
      fx.js           win burst particles, glow, seat spotlight
      labels.js       DOM labels positioned by projecting world points
    ui/               DOM glass panels; read from store, never touch stage
      store.js        single app state, settings, subscribe(), persistence
      home.js  hud.js  dock.js  info.js  review.js  stats.js  settings.js
      concepts.js  rangeGrid.js
    lessons/
      content.js      the 13 lessons: prose plus staging cues
      drills.js       the 16 drill generators (pure, node-testable); each
                      returns a question, options, answer and a table scene
      lessons.js      lesson/drill view and navigation
    audio.js          synthesized WebAudio, muted by default
    director.js       the one bridge: turns engine events and lesson scenes
                      into sequenced stage animations, sounds and labels
    session.js        the turn loop: engine + bots + coach + director
    main.js           boot, mode routing, WebGL detection
  _tests/             node --test suites
  _docs/              this spec and the implementation plan
```

Directories prefixed with `_` are skipped by Jekyll, so tests and docs sit next
to the code without being published. No `_config.yml` change is needed.

### Layer rules

- `core/` imports nothing outside `core/` and never touches `Math.random`,
  `window` or `document`.
- `stage/` imports three.js and other `stage/` files only. Its API is in terms
  of seats, cards and chip amounts, never ranges or what a street means.
- `ui/` and `lessons/` import `core/` and `store.js`. They may import
  `stage/textures.js` for the shared card painter, nothing else from `stage/`.
- `director.js` is the only file that knows both engine events and `stage/`.
- `session.js` and `main.js` wire everything.

### Engine: event-sourced

The engine is rewritten rather than ported. `engine.js` exposes:

- `createGame(config)` -> state; `startHand(state, rng)` -> events
- `legalActions(state)` -> what the player to act may do, with min/max raise
- `apply(state, action)` -> events (state is updated in place)
- `snapshot(state)` at hand start, and `replay(snapshot, actions, upto)` ->
  `{ state, events }` at any point in that hand

Events: `handStart`, `post`, `deal`, `action`, `refund`, `street`, `showdown`,
`award`, `handEnd`. Every state change is described by an event, so the director can
animate it, the review can rewind to it, and tests can assert on it. The
engine never waits on animation; `session.js` awaits the director before asking
the next bot or enabling hero controls.

A **scene** is the lesson-side equivalent: a plain description of what should
be on the table (seats, cards face up or down, chips, highlights). The
director can apply a scene with or without animation. Review rewind and
context-loss recovery use the same path: build a scene from engine state and
apply it instantly.

### Settings and persistence

All settings live on `store.settings`; nothing in game logic reads the DOM.

| Key                   | Contents                                     | Status    |
| --------------------- | -------------------------------------------- | --------- |
| `pokerTrainer.prefs`  | table settings; gains fourColour, muted      | extended  |
| `pokerTrainer.v2`     | mode, lastMode, completed lessons            | read, migrated |
| `pokerTrainer.stats1` | per-coach-level scorecard, last 40 decisions | new       |

Old `mode` values `trainer` and `coach` map to Table at Guided and Silent;
`tutorial` maps to Learn.
Lesson ids are kept so completion carries over. Reads tolerate missing or
malformed values. If localStorage is unavailable, an in-memory store is used.

## Visual design

### Scene

Near-black room (`#050607`), one warm spot lamp above the table, a hemisphere
fill and a cool rim light so card edges and chip sides read. PCF soft shadows,
ACES filmic tone mapping, sRGB output, light fog. Deep green felt (`#0f3d2e`),
brass accent (`#d8b36a`), glass panels and system font stack shared with
flyjack.

### Table

Racetrack oval with a padded rail, a painted betting line and faint arc
lettering on the felt. Seats for 2-9 players are distributed around the oval
with the hero fixed at the bottom nearest the camera. The dealer button is a
puck that slides between seats.

### Cards

Thin boxes with rounded-corner faces painted into canvases; oversized indices
so rank and suit are legible from the default camera. Optional four-colour
deck. The same painter produces the DOM cards used in panels.

### Chips

Clay-style cylinders with striped edge textures, coloured by denomination,
drawn with instancing. Amounts are broken into denominations for display only.

### Choreography

All motion runs through `tween.js`, scales with the speed setting, and can be
skipped by click or Space.

| Event    | Motion                                                         |
| -------- | -------------------------------------------------------------- |
| deal     | cards arc from the dealer seat; hero cards flip up with a lift |
| post/bet | stack slides from the seat across the betting line             |
| fold     | cards slide to the muck and dim                                |
| street   | bets sweep to the pot; burn; flop fans, turn/river land singly |
| showdown | hands flip; winning five rise and glow; losers desaturate      |
| award    | pot slides to the winner; brass particle burst                 |

### Camera

Default three-quarter view from behind the hero. Tweened shots for deal, board
and showdown, plus lesson shots (top-down for position, close on the board for
texture). Pointer drag orbits, wheel zooms, double-click resets. Under 720px
wide the default shot pulls back and up so every seat fits in portrait.

### Labels and panels

Names, stacks, bet amounts and action bubbles are DOM elements projected from
world points each frame. Panels are blurred glass: header; bottom dock
(actions, raise slider with presets, EV per size, coach's pick; or drill
answers in lessons); right drawer (Ranges / EV / Next card / Hand / Why in
play, lesson text in Learn); review sheet with timeline; stats; settings
popover. Under 720px the drawer becomes a bottom sheet.

### Audio

Synthesized card snap, chip click, check knock, fold swish and win chime.
Muted by default; the choice persists.

## Performance and accessibility

- Pixel ratio capped at 2. An FPS meter steps down shadow map size, then pixel
  ratio, after three consecutive slow seconds.
- Coach work runs in a worker and is cancelled when the hand moves on.
- `prefers-reduced-motion` and the animations-off setting make tweens resolve
  instantly; state still ends correct because the director always applies
  final positions.
- Action and answer buttons are real `<button>`s with keyboard shortcuts.
  Everything the 3D scene conveys (cards, bets, pot, whose turn) is also
  present in DOM labels or panels, so screen readers and the no-WebGL path
  are not locked out of information.
- Without WebGL, lessons still work using DOM cards in the panel; the table
  modes show a notice.

## Error handling

- WebGL context loss: pause the turn loop, rebuild GPU resources on restore,
  re-apply the current scene without animation.
- A rejected or skipped animation never blocks the turn loop; the director
  snaps the stage to the engine's state and continues.
- Worker failure or absence: fall back to running the coach on the main
  thread with today's smaller sample counts.

## Testing

Tests assert that the poker is right, not that it matches the old code.

### Core correctness (node --test)

- **Evaluator**: every hand category on hand-written cases including wheel
  straights, steel wheel, board plays, kickers and ties; best-five extraction.
  Differential test of 100,000 seeded 7-card hands against a deliberately
  naive brute-force evaluator (rank all 21 five-card subsets) written in the
  test file. The old `eval7` is used as a second oracle, extracted from
  `git show 5a64bf9c:games/poker/index.html` and run in `node:vm`.
- **Equity**: known matchups within Monte Carlo tolerance (AA v KK ~82%, AKs v
  QQ ~46%, flush draw + overcards on the flop, drawing dead = 0, locked = 1).
- **Engine invariants**, property-tested over thousands of seeded random-play
  hands at 2-9 players: chips are conserved; no negative stacks; only legal
  actions are accepted; min-raise rules hold; betting rounds terminate;
  all-in side pots award correctly (hand-written 3- and 4-way cases); button
  and blinds rotate, including heads-up blind order; `replay` reproduces
  `apply` exactly; same seed gives the same hand.
- **Analysis**: outs, hand class, board texture and combo counts on
  hand-written cases.
- **Bots**: always return a legal action; style ordering holds in aggregate
  (maniac VPIP > LAG > TAG > nit; station calls most).
- **Coach**: never recommends an illegal action; folds when drawing dead
  facing a bet; does not fold the nuts; grade is `correct` when the hero
  takes the recommended action.
- **Drills**: for every generator over many seeds, exactly one option is
  correct, options are distinct, and the stated answer agrees with `core/`.
- **Store**: persistence round-trips, migration from the old keys, malformed
  values.
- **Layout**: seat positions for every player count do not overlap and keep
  the hero at the bottom; chip denomination breakdown sums to the amount.

### Visual

A script drives headless Chrome to capture: idle table, mid-deal, flop with
bets out, showdown, each info tab, review with replay, several lessons and a
drill, at 1440x900 and 390x844. Screenshots are reviewed by eye at each
milestone.

### Manual smoke

Play full hands at both coach levels at 2, 6 and 9 players; complete a lesson
end to end; open a review and scrub it; reload and confirm progress, prefs and
stats survive; load with a localStorage from the old version and confirm
migration.

## Milestones

Each ends in something runnable and is checked before the next begins.

1. **Core**: `core/` with its full test suite green in node.
2. **Stage**: lit table, cards and chips, scene application and the
   choreography table above, driven by a scripted demo hand. **Look check:
   screenshots go to Fan Pu before building on top.**
3. **Table**: turn loop, bots, dock, HUD, labels, coach worker, info drawer,
   both coach levels. Playable end to end.
4. **Review and progress**: replayer, stats persistence, home with weakest
   category.
5. **Learn**: lesson panel, all 13 lessons staged on the table, all 16 drills.
6. **Polish**: audio, mobile layout, reduced motion, context loss, no-WebGL
   path, keyboard, migration, final screenshot pass.

## Rollout

Work happens on branch `poker-trainer-3d`, which is not deployed until merged.
The new shell replaces `index.html` from milestone 2 onward; the old version
remains in git history at `5a64bf9c` for reference. `.prettierignore` excludes `games/**`,
so CI does not check these files; they are still formatted with the repo's
`.prettierrc` (print width 150) for consistency.
