# Poker trainer M2: stage — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A lit three.js poker table that can show any table state instantly and animate every engine event, demonstrated by bots playing real hands, and captured in screenshots for Fan Pu's look check.

**Architecture:** `js/stage/` renders and animates but knows no poker rules: its vocabulary is seats, cards, chip amounts and highlights. Pure geometry (`layout.js`) and timing (`tween.js`) have no three.js dependency and are node-tested. `js/director.js` is the only bridge: it turns engine events into awaited stage calls, and can also snap the stage to a **scene** (a plain description of what is on the table) with no animation, which is what screenshots, review rewind and context-loss recovery use.

**Tech Stack:** three.js 0.186.0 vendored as ES modules, Canvas 2D for procedural textures, headless Chrome via puppeteer-core for screenshots (dev-only, under `_tools/`, not published).

**Spec:** `games/poker/_docs/2026-09-19-poker-trainer-3d-design.md` (sections "Visual design", "Performance and accessibility", "Error handling"). Plan 2 of 6.

## Global Constraints

- `stage/` imports three.js and other `stage/` files only. No import from `core/`, `ui/` or `director.js`.
- Units: 1 world unit = 10 cm. Table long axis is x, hero sits at +z (nearest the camera), y is up.
- Palette: room `#050607`, felt `#0f3d2e`, brass accent `#d8b36a`, card stock `#fbfaf5`. System font stack only; no remote fonts or images.
- All motion goes through `tween.js`; nothing else calls `requestAnimationFrame` except the single loop in `stage.js`.
- Every animated method returns a promise and leaves the object at its exact final transform even when skipped or when motion is reduced.
- Pixel ratio capped at 2. Shadow map 2048 by default, stepping down to 1024 then off, then pixel ratio to 1, after 3 consecutive seconds under 40 fps.
- Cards are `{ r, s }` as in core; the stage keys card textures by `r * 4 + s`.
- Formatting: `npx prettier@3.1.1 --no-config --print-width 150 --trailing-comma es5`.
- Commit after every task, message prefix `poker stage:`.

## Deviation from the spec's file list (spec updated to match)

- `stage/layout.js` added: all table geometry as pure functions (seat positions, card slots, bet spots, chip breakdown), so it can be tested in node. `stage/table.js` builds meshes from it.

---

### Task 1: Vendor three.js, tween, layout

**Files:** Create `vendor/three.module.min.js`, `vendor/three.core.min.js`, `vendor/THREE-LICENSE`, `js/stage/tween.js`, `js/stage/layout.js`; Test `_tests/tween.test.mjs`, `_tests/layout.test.mjs`

**Interfaces — Produces:**
- `tween.js`: `ease = { linear, inOut, out, in, outBack }`; `class Animator { speed; reduced; tween(seconds, update(k), easing) -> Promise; wait(seconds) -> Promise; step(dtMs); skip(); get busy }`. `reduced = true` makes every tween finish on its first step.
- `layout.js`:
  - `TABLE = { a: 17, b: 9.5, rail: 1.6, y: 0 }` (semi-axes of the felt racetrack, rail width)
  - `CARD = { w: 1.26, h: 1.76, t: 0.02 }`, `CHIP = { r: 0.39, t: 0.066 }`
  - `seatLayout(n) -> [{ seat, angle, pos:{x,z}, cards:[{x,z,rot}], bet:{x,z}, stack:{x,z}, label:{x,z}, button:{x,z} }]`; seat 0 at angle +90 degrees (bottom), others clockwise as seen from above, spaced evenly around the racetrack perimeter
  - `boardSlots() -> [{x,z}] x5`, `POT = {x,z}`, `MUCK = {x,z}`, `DECK = {x,z}`
  - `DENOMS = [{ value, color, edge }]` for 500, 100, 25, 5, 1; `chipBreakdown(amount, maxChips = 24) -> [{ value, count }]` whose values sum to `amount` exactly, greedy from the top, merging upward when over `maxChips`

- [x] **Step 1:** `cd /tmp && npm pack three@0.186.0`, extract `build/three.module.min.js`, `build/three.core.min.js` and `LICENSE` into `vendor/`. Verify `three.module.min.js` imports `./three.core.min.js` by relative path.
- [x] **Step 2: Failing tests.** Tween: a 1 s tween stepped by 250 ms calls `update` with increasing `k` ending at exactly 1 and then resolves; `speed = 2` halves the steps needed; `skip()` jumps every active tween to `k = 1` and resolves; `reduced` resolves on the first step; `wait` resolves; easings map 0 to 0 and 1 to 1 and `inOut` is monotonic. Layout: for every n in 2..9, n seats, seat 0 has the largest z and x near 0, angles strictly ordered, minimum distance between any two seats' card centres > `2.2 * CARD.w`, every `bet` spot is closer to the centre than its `pos`, every position lies inside the felt, no bet spot overlaps a board slot or the pot (distance > 1.5); `chipBreakdown` sums exactly for amounts 1..2000 and never exceeds `maxChips` chips for amounts up to 5000.
- [x] **Step 3:** Run, watch fail. **Step 4:** Implement. **Step 5:** Pass. Commit.

### Task 2: Textures

**Files:** Create `js/stage/textures.js`

**Interfaces — Produces:** painters that draw into a given 2D context, so the DOM can reuse them, plus texture factories:
- `paintCardFace(g, w, h, card, { fourColour })`, `paintCardBack(g, w, h)`, `paintFelt(g, size)`, `paintChipTop(g, size, denom)`, `paintChipEdge(g, w, h, denom)`, `paintButton(g, size)`
- `cardCanvas(card, opts) -> HTMLCanvasElement` (for DOM use), `makeTextures(renderer, { fourColour }) -> { face(card), back, felt, chipTop(i), chipEdge(i), button, dispose() }` with faces created lazily and cached.
- Card face: large corner index (rank 38% of card width, suit beneath), one big centre pip or court letter, 512x716 px. Suit colours: two-colour `#161616` / `#b3261e`; four-colour adds `#1d5fbf` diamonds and `#1d7a3a` clubs.
- Felt: radial gradient from `#14503c` at the lamp's pool to `#0a2a20`, fine grain noise from a fixed-seed generator (so screenshots are stable), brass betting line as a racetrack inset 62% of the way out, faint arc lettering "NO LIMIT HOLD'EM" top and "TRAINER" bottom.

- [x] **Step 1:** Implement. **Step 2:** Verified visually in Task 6 (`?pose=cards` shows all 52 faces plus a back). Commit.

### Task 3: Stage, table, camera

**Files:** Create `js/stage/stage.js`, `js/stage/table.js`, `js/stage/shots.js`

**Interfaces — Produces:**
- `new Stage(container, { reducedMotion, fourColour })` with `.scene .camera .renderer .anim (Animator) .textures`, `.onFrame(fn(dtMs))`, `.start()`, `.stop()`, `.resize()`, `.setQuality(level)`, `.dispose()`; handles `webglcontextlost` / `webglcontextrestored` by emitting `stage.onContextRestored`.
- Lights: warm `SpotLight(0xffd9a8)` above centre casting shadows, `HemisphereLight(0x6f7a8c, 0x0b0805)`, cool rim `DirectionalLight(0x8fb4ff)` from behind the far rail. `ACESFilmicToneMapping`, `SRGBColorSpace`, `PCFSoftShadowMap`, `Fog(0x050607)`.
- `buildTable(stage) -> { group, setSeats(n), moveButton(seat, animate) -> Promise }`: racetrack felt (ExtrudeGeometry from a rounded-rect Shape, felt texture mapped by world xz), padded rail (tube-like extrusion, dark leather `#1a1210` with a specular sheen), apron beneath, a floor disc that catches the lamp's falloff.
- `shots.js`: `SHOTS = { table, deal, board, showdown, top, lesson }` as `{ pos, target, fov }`; `createCameraRig(stage) -> { to(name, seconds) -> Promise, snap(name), enableOrbit(dom), setPortrait(bool) }`. Portrait variants pull back and up so nine seats fit 390x844.

- [x] **Step 1:** Implement. **Step 2:** `index.html` minimal shell + `js/main.js` that mounts the stage and table. **Step 3:** Screenshot via Task 6 tooling; iterate on lighting until the felt pool, rail sheen and shadows read well. Commit.

### Task 4: Cards and chips

**Files:** Create `js/stage/cards3d.js`, `js/stage/chips3d.js`

**Interfaces — Produces:**
- `createCards(stage) -> { place(id, card, slot, { faceUp }) , deal(id, card, from, slot, { faceUp, delay }) -> Promise, flip(id, faceUp) -> Promise, moveTo(id, slot) -> Promise, muck(id) -> Promise, lift(ids, on) -> Promise, dim(ids, on), fan(idPrefix, cards, centre) -> Promise, remove(id), clear() }`. `id` is a string (`"h0a"`, `"h0b"`, `"b2"`, `"out7"`); `slot` is `{ x, z, rot, y? }`. Meshes are pooled. A card is a thin `BoxGeometry` with face/back materials and an ivory edge; lift raises and adds an emissive brass rim via a second slightly larger mesh.
- `createChips(stage) -> { setStack(key, amount, at) , slide(key, toKey, at) -> Promise, merge(fromKeys, toKey, at) -> Promise, split(fromKey, shares:[{ key, amount, at }]) -> Promise, clear() }`. One `InstancedMesh` per denomination; a stack is drawn as columns of at most 12 chips; instance matrices are rewritten on change.

- [x] **Step 1:** Implement. **Step 2:** Screenshot `?pose=flop` and `?pose=showdown`. Commit.

### Task 5: Labels, fx, director

**Files:** Create `js/stage/labels.js`, `js/stage/fx.js`, `js/director.js`, `css/base.css`, `css/hud.css`

**Interfaces — Produces:**
- `createLabels(stage, container) -> { set(key, { x, y, z }, html, className), remove(key), clear() }`; positions are projected each frame; off-screen or behind-camera labels are hidden.
- `createFx(stage) -> { burst(at), spotlightSeat(seat | null), glowCards(ids, on) }`.
- `director.js`: `createDirector(stage, parts) -> { applyScene(scene, { animate = false }) -> Promise, play(events, ctx) -> Promise, skip(), setSpeed(x) }`.
  - **Scene shape** (plain JSON): `{ n, dealer, heroSeat: 0, seats: [{ seat, name, stack, bet, cards: [card|null, card|null] | null, faceUp, folded, allIn, acting, tag }], board: [card], pot, highlight: { cards: [cardKey], seats: [seat] }, fan: [card] }`.
  - `sceneFromState(state, { reveal: [seats] })` lives in `director.js` and is the only place engine state is translated for the stage.
  - `play` handles `handStart, post, deal, action, refund, street, showdown, award, handEnd` per the spec's choreography table; after the last event it re-applies `sceneFromState` without animation so the stage can never drift from the engine.
- Seat labels show name, stack in bb, position tag, and the last action as a bubble that fades.

- [x] **Step 1:** Implement. **Step 2:** `main.js` autoplay: bots in every seat (hero seat plays the coach's light advice), `?seed=`, `?n=`, `?speed=`. **Step 3:** Watch several hands via screenshots at intervals; fix drift, overlap and timing. Commit.

### Task 6: Screenshot tooling and the look check

**Files:** Create `_tools/package.json`, `_tools/shoot.mjs`, `_tools/.gitignore` (`node_modules`, `out`)

- `shoot.mjs` serves `games/poker/` on a local port, launches `/usr/bin/google-chrome` headless through puppeteer-core with WebGL enabled, and for each of the poses `table, cards, deal, flop, bets, showdown, nine, heads-up` captures 1440x900 and 390x844 into `_tools/out/`. Poses are reached with `?pose=<name>&seed=1`, which builds a fixed scene and calls `applyScene` with no animation, then sets `document.body.dataset.ready = "1"`; the script waits for that attribute.
- [x] **Step 1:** Implement and run. **Step 2:** Review every image by eye; fix what looks wrong; repeat. **Step 3:** Send the desktop and phone captures of `flop`, `showdown` and `nine` to Fan Pu. **STOP for the look check before starting M3.**
