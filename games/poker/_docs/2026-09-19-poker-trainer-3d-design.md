# Poker trainer: 3D rebuild — design

Date: 2026-09-19
Status: awaiting review

## Goal

Rebuild `games/poker/` from a single 227 KB `index.html` into a multi-file
project with a three.js table in the same family as `games/flyjack/`, while
keeping every existing feature behaving exactly as it does today.

Success means:

1. `/games/poker/` still loads at the same URL with no build step.
2. Every mode, lesson, drill, bot style, coach recommendation and grade is
   behaviourally identical to the current version, proven by tests.
3. The table is a lit 3D scene with animated cards and chips, verified by
   screenshots at desktop and phone sizes.
4. Existing users keep their lesson progress and table preferences.

## Decisions already made

| Question     | Decision                                                  |
| ------------ | --------------------------------------------------------- |
| Rendering    | Full 3D, three.js. DOM glass panels layered over a canvas |
| Scope        | Feature parity plus the small fixes listed below          |
| Identity     | Sibling of flyjack with its own felt and accent           |
| Code shipped | Native ES modules, no bundler, three.js vendored locally  |

## Non-goals

- No changes to lesson content, drill design, bot strategy, coach logic or
  grading thresholds.
- No backend, accounts, multiplayer or solver data.
- No build step, TypeScript or framework.
- No 2D fallback renderer for the table.

## Architecture

```
games/poker/
  index.html          thin shell: canvas mount, empty panel slots, analytics,
                      <script type="module" src="js/main.js">
  css/
    base.css          design tokens, glass panel primitives, buttons
    hud.css           header, seat labels, decision dock
    panels.css        info drawer, coach review, stats, settings, home
    tutorial.css      lesson cards, drills, DOM cards
  vendor/
    three.module.min.js   three 0.186.0, vendored from npm, no CDN
  js/
    core/             PURE: no DOM, no three.js, importable from node
      rng.js          seeded mulberry32; one injectable random source
      cards.js        deck, card encoding, shuffling
      eval.js         eval7, straightHigh, describeScore
      preflop.js      RANGE_TIERS, PCT, TIER_END, chenScore
      equity.js       Monte Carlo simulate
      analysis.js     outs, handClass, boardTexture, cbetPlan, combosNow,
                      range lists, composition, next-card map, EV sampling
      engine.js       createGame, startHand, act, advance, showdown (side
                      pots), endHand; appends to an event log
      bots.js         STYLES, READ_PRIOR, botDecide
      coach.js        analyze, recommend, grade, lessonFor
    stage/            three.js ONLY: knows nothing about poker rules
      stage.js        renderer, scene, lights, fog, RAF loop, resize,
                      adaptive quality
      tween.js        promise-based Animator, easings, skip(), speed scale
      textures.js     procedural painters: felt, card faces and back, chips
      table.js        table and rail meshes, seat layout for 2-9 players,
                      dealer button
      cards3d.js      pooled card meshes: deal, flip, muck, reveal, highlight
      chips3d.js      instanced chip stacks: bet, sweep to pot, award
      shots.js        named camera shots and orbit/zoom controls
      fx.js           win burst particles, winning-card glow
      labels.js       DOM labels positioned by projecting world points
    ui/               DOM glass panels; read from store, never touch stage
      store.js        single app state, settings object, subscribe(),
                      persistence
      home.js  hud.js  controls.js  info.js  coachReview.js
      stats.js  settings.js  concepts.js
    tutorial/
      lessons.js      the 13 lessons' content
      drills.js       the 16 drill generators (pure, node-testable)
      tutorial.js     lesson/drill view and navigation
    audio.js          synthesized WebAudio, muted by default
    director.js       the one bridge: engine events -> sequenced stage
                      animations, sounds and label updates
    main.js           boot, mode routing, WebGL detection
  _tests/             node --test suites
  _docs/              this spec and the implementation plan
```

Directories prefixed with `_` are skipped by Jekyll, so tests and docs sit next
to the code without being published. No `_config.yml` change is needed.

### Layer rules

- `core/` imports nothing outside `core/`.
- `stage/` imports three.js and other `stage/` files only. Its API is in terms
  of seats, cards and chip amounts, never hands, ranges or streets' meaning.
- `ui/` and `tutorial/` import `core/` (for analysis helpers) and `store.js`.
  They may import `stage/textures.js` for the shared card painter, nothing
  else from `stage/`.
- `director.js` is the only file that imports both `core/engine.js` and
  `stage/`.
- `main.js` wires everything.

### Engine event log

Today the engine mutates a game object `g` and the page re-renders wholesale.
Animation needs to know what just happened, so `engine.js` appends plain
objects to `g.events` as it mutates state. Decision logic is not altered.

Event types: `handStart`, `post` (blind/ante), `deal` (hole cards), `action`
(seat, kind, amount, toAmount), `street` (flop/turn/river with cards),
`showdown` (revealed hands), `award` (seat, amount, pot index), `handEnd`.

`director.js` drains the log after each engine call and plays the events in
order, awaiting each animation. The engine never waits on the director; the
turn loop in `main.js` awaits the director before asking the next bot or
enabling hero controls.

### Settings

Game logic currently reads `$('speed')`, `$('anim')`, `$('showPick')`,
`$('pauseMode')` and `$('nPlayers')` straight from the DOM. These become
fields on `store.settings`, passed in where needed. A new `fourColour` deck
setting and a `muted` audio setting are added.

### Persistence

| Key                   | Contents                                  | Status    |
| --------------------- | ----------------------------------------- | --------- |
| `pokerTrainer.prefs`  | table settings                            | unchanged |
| `pokerTrainer.v2`     | mode, lastMode, completed lessons         | unchanged |
| `pokerTrainer.stats1` | scorecard by category, last 40 decisions  | new       |

Reads tolerate missing or malformed values and fall back to defaults.

### Fixes included

- Stats and decision history persist across reloads.
- `?seed=N` makes a session reproducible.
- The `stalled` flag is replaced by an explicit paused state in the turn loop:
  leaving the table mid-hand pauses it, returning resumes it.
- The duplicated compact and full panel renderers collapse to one component
  per panel with a `compact` option.
- Dead `#stats` and `#log` nodes and their CSS are removed.
- Seat layout derives from table geometry instead of `h=238`.
- Tutorial option buttons are queried within the tutorial root.
- `/assets/js/game-analytics.js` is loaded, matching the other games.

## Visual design

### Scene

Near-black room (`#050607`), one warm spot lamp above the table, a hemisphere
fill and a cool rim light so card edges and chip sides read. PCF soft shadows,
ACES filmic tone mapping, sRGB output, light fog. Deep green felt (`#0f3d2e`), brass
accent (`#d8b36a`), glass panels and system font stack shared with
flyjack.

### Table

Racetrack oval with a padded rail, a painted betting line and faint arc
lettering on the felt. Seats for 2-9 players are distributed around the oval
with the hero fixed at the bottom nearest the camera. The dealer button is a
puck that slides between seats.

### Cards

Thin boxes with rounded-corner faces painted into canvases; oversized indices
so rank and suit are legible from the default camera. Optional four-colour
deck. The same painter produces the DOM cards used in lessons and panels.

### Chips

Clay-style cylinders with striped edge textures, coloured by denomination,
drawn with instancing. Amounts are broken into denominations for display only.

### Choreography

All motion runs through `tween.js`, scales with the bot-speed setting, and can
be skipped by click or key.

| Event    | Motion                                                           |
| -------- | ---------------------------------------------------------------- |
| deal     | cards arc from the dealer seat; hero cards flip up with a lift   |
| post/bet | stack slides from the seat across the betting line               |
| fold     | cards slide to the muck and dim                                  |
| street   | bets sweep to the pot; burn; flop fans, turn/river land singly   |
| showdown | hands flip; winning five rise and glow; losers desaturate        |
| award    | pot slides to the winner; brass particle burst                   |

### Camera

Default three-quarter view from behind the hero. Tweened shots for deal, board
and showdown. Pointer drag orbits, wheel zooms, double-click resets. On
viewports under 720px wide the default shot pulls back and up so every seat
fits in portrait.

### Labels and panels

Names, stacks, bet amounts and action bubbles are DOM elements projected from
world points each frame. Panels are blurred glass: header; bottom decision
dock (actions, raise slider, EV per size, coach's pick highlight); right
drawer with Ranges / EV / Next card / Hand / Why tabs; centred coach review
sheet with the equity sparkline; stats; settings popover. Coaching mode hides
the hint panels as it does today. Under 720px the drawer becomes a bottom
sheet.

### Tutorial

Stays DOM: restyled glass lesson cards using the shared card painter. The 3D
table idles dimmed with a slow orbit behind it.

### Audio

Synthesized card snap, chip click, check knock, fold swish and win chime.
Muted by default; the choice persists.

## Performance and accessibility

- Pixel ratio capped at 2. An FPS meter steps down shadow map size, then pixel
  ratio, after three consecutive slow seconds.
- `prefers-reduced-motion` and the animations-off setting make tweens resolve
  instantly; state still ends up correct because the director always applies
  final positions.
- Action buttons are real `<button>`s with keyboard shortcuts preserved.
  Information conveyed by the 3D scene (cards, bets, pot) is always also
  present in DOM labels or panels.
- Without WebGL, Tutorial works fully; Trainer and Coaching show a notice
  instead of the table.

## Error handling

- WebGL context loss: pause the turn loop, rebuild GPU resources on restore,
  re-apply the current game state without animation.
- A rejected or skipped animation never blocks the turn loop; the director
  snaps the stage to the engine's state and continues.
- localStorage unavailable (private mode): fall back to the in-memory store
  the current code already uses.

## Testing

### Parity (written before any code moves)

A characterization harness extracts the script blocks from the committed
single-file version (`git show 5a64bf9c:games/poker/index.html`), runs the
DOM-free section in a `node:vm` context with a seeded `Math.random` (the old
code's only entropy source, 17 call sites), and records golden output:

- 100 seeded hands at each of 2, 6 and 9 players (300 total) with a scripted
  hero policy:
  full game state after every action, every bot decision, every coach
  `analyze`/`recommend`/`grade` result.
- `eval7` on a fixed set of 10,000 seeded 7-card hands.
- `simulate`, outs, handClass, boardTexture, combos and range composition on
  fixed seeded inputs.
- 50 seeded instances of each of the 16 drill generators.

The new modules, driven by the same seeds, must reproduce the goldens exactly.
The event log is excluded from the state comparison and tested on its own.

### Unit

Event log ordering and completeness per hand; side-pot awards; store
persistence round-trips and malformed-value handling; chip denomination
breakdown; seat layout for every player count (no overlaps, hero at bottom).

### Visual

A script drives headless Chrome to capture: idle table, mid-deal, flop with
bets out, showdown, each info tab, coach review, a lesson page and a drill, at
1440x900 and 390x844. Screenshots are reviewed by eye at each milestone. The
first milestone, a lit table with cards and chips before any panels exist, is
sent to Fan Pu for a look check before work continues.

### Manual smoke

Play full hands in Trainer and Coaching at 2, 6 and 9 players; complete one
lesson end to end; reload and confirm progress, prefs and stats survive.

## Rollout

Work happens on branch `poker-trainer-3d`. The old `index.html` is replaced in
the same branch once parity and visual checks pass; it remains in git history
and is the source for the parity harness. `.prettierignore` excludes
`games/**`, so CI does not check these files; they are still formatted with
the repo's `.prettierrc` (print width 150) for consistency.
