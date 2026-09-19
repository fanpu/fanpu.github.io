import * as core from "./core/index.js";
import { Stage } from "./stage/stage.js";
import { buildTable } from "./stage/table.js";
import { createCameraRig } from "./stage/shots.js";
import { createCards } from "./stage/cards3d.js";
import { createChips } from "./stage/chips3d.js";
import { createLabels } from "./stage/labels.js";
import { createFx } from "./stage/fx.js";
import { createDirector, sceneFromState } from "./director.js";
import { cardCanvas } from "./stage/textures.js";

// Milestone 2 shell: the stage on its own. With ?pose=<name> it shows one fixed scene (for screenshots);
// otherwise bots play real hands through the director, with the hero's seat following the coach's advice.
const params = new URLSearchParams(location.search);
const num = (key, fallback) => (params.has(key) ? +params.get(key) : fallback);
const { parseCards: P } = core;

function boot() {
  const mount = document.getElementById("stage");
  if (params.get("ui") === "mock") mockPanels();
  if (!Stage.supported()) {
    document.body.insertAdjacentHTML(
      "beforeend",
      `<div class="notice">This table is drawn with WebGL, which this browser is not offering.<br />The lessons will still work once they are built.</div>`
    );
    return;
  }
  const stage = new Stage(mount, { reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches, fourColour: params.has("four") });
  const parts = {
    stage,
    table: buildTable(stage),
    rig: null,
    cards: createCards(stage),
    chips: createChips(stage),
    labels: createLabels(stage, mount),
    fx: createFx(stage),
  };
  parts.table.setSeats(6);
  parts.rig = createCameraRig(stage, parts.table);
  const director = createDirector(parts);
  parts.rig.enableOrbit(stage.renderer.domElement);
  window.__poker = { core, stage, director, ...parts };

  const pose = params.get("pose");
  if (pose === "back") return showArtwork(stage);
  if (pose) showPose(pose, parts, director);
  else autoplay(parts, director);
  stage.start();
  document.body.dataset.ready = "1";
}

// ---------- placeholder panels (?ui=mock): the screen-space budget the real panels of milestone 3 will fill ----------
// The panels claim their space through the --stage-* variables; the camera reframes the table into what is left.
function mockPanels() {
  const link = Object.assign(document.createElement("link"), { rel: "stylesheet", href: "css/mock.css" });
  document.head.appendChild(link);
  document.body.classList.add("mock");
  document.getElementById("hud").innerHTML = `
    <header class="glass m-head"><b>Poker Trainer</b><span>Train &middot; guided</span><i>hand 12 &nbsp; +14.5 bb</i></header>
    <aside class="glass m-drawer">
      <nav><b>Ranges</b><span>EV</span><span>Next card</span><span>Hand</span><span>Why</span></nav>
      <div class="m-grid"></div><p>13 &times; 13 range grid, composition bars, EV table</p>
    </aside>
    <footer class="glass m-dock">
      <div class="m-strip"><span class="m-cards" id="mHand"></span><em>top pair, nut flush draw</em><span class="m-cards" id="mBoard"></span></div>
      <div class="m-actions"><button>Fold</button><button>Call 6 bb</button><button class="pick">Raise to 18 bb</button></div>
      <div class="m-sizes"><span>&frac12; pot</span><span>&frac23; pot</span><span>pot</span><span>all-in</span><input type="range" aria-label="raise size" /></div>
      <nav class="m-tabs"><b>Ranges</b><span>EV</span><span>Next card</span><span>Hand</span><span>Why</span></nav>
    </footer>`;
  const grid = document.querySelector(".m-grid");
  for (let i = 0; i < 169; i++)
    grid.appendChild(
      Object.assign(document.createElement("i"), {
        style: `opacity:${(0.15 + 0.85 * Math.max(0, 1 - ((i % 13) + Math.floor(i / 13)) / 14)).toFixed(2)}`,
      })
    );
  for (const [id, txt] of [
    ["mHand", "Ah Qh"],
    ["mBoard", "Kh 9h 2c"],
  ])
    for (const c of P(txt)) document.getElementById(id).appendChild(cardCanvas(c, {}, 120));
}

// ---------- the painted artwork at full size, to check it without the table in the way ----------
function showArtwork(stage) {
  stage.stop();
  const wall = document.createElement("div");
  wall.style.cssText =
    "position:fixed;inset:0;z-index:5;display:flex;gap:28px;align-items:center;justify-content:center;background:#123a2c;flex-wrap:wrap;overflow:auto";
  for (const card of [null, { r: 14, s: 0 }, { r: 12, s: 1 }, { r: 10, s: 2 }, { r: 7, s: 3 }]) {
    const c = cardCanvas(card, { fourColour: params.has("four") }, 512);
    c.style.cssText = "height:min(78vh,560px);width:auto;border-radius:4.5%;box-shadow:0 12px 40px rgba(0,0,0,.5)";
    wall.appendChild(c);
  }
  document.body.appendChild(wall);
  document.body.dataset.ready = "1";
}

// ---------- fixed scenes, for screenshots and for checking the look ----------
function showPose(name, { stage, rig, cards }, director) {
  const n = name === "nine" ? 9 : name === "headsup" ? 2 : 6;
  const state = core.createGame({ n });
  const rng = core.makeRng(num("seed", 1));
  state.dealer = n - 1; // startHand moves the button on by one: the hero is the dealer
  core.startHand(state, rng);
  const rig6 = (hands, board) => {
    hands.forEach((h, i) => h && (state.players[i].cards = P(h)));
    state.deck.push(...P(board).reverse());
  };
  const A = (type, to) => core.apply(state, { seat: state.toAct, type, ...(to ? { to } : {}) });
  if (name === "table") {
    state.handOver = true;
    for (const p of state.players) Object.assign(p, { stack: p.startStack, bet: 0, invested: 0, cards: [] });
    director.applyScene(sceneFromState(state));
  } else if (name === "cards") {
    director.applyScene(
      sceneFromState(Object.assign(state, { handOver: true, players: state.players.map((p) => ({ ...p, cards: [], bet: 0, invested: 0 })) }))
    );
    const deck = core.newDeck();
    deck.forEach((c, i) =>
      cards.place("x" + i, c, { x: ((i % 13) - 6) * 1.42, z: (Math.floor(i / 13) - 1.5) * 1.95 - 0.2, rot: 0, y: 0.2 }, { faceUp: true })
    );
    cards.place("xb", null, { x: 0, z: 5.6, rot: 0, y: 0.2 }, { faceUp: false });
    rig.snap("top");
  } else if (name === "preflop" || name === "headsup") {
    rig6(["As Kh"], "");
    if (n > 2) A("raise", 6);
    director.applyScene(sceneFromState(state));
  } else if (name === "flop" || name === "nine") {
    rig6(["Ah Qh"], "Kh 9h 2c");
    const script =
      n === 9
        ? [["fold"], ["raise", 6], ["fold"], ["call"], ["fold"], ["call"], ["call"], ["fold"], ["call"]]
        : [["raise", 6], ["fold"], ["call"], ["call"], ["fold"], ["call"]];
    for (const [t, to] of script) if (!state.handOver && state.street === 0) A(t, to);
    while (!state.handOver && state.street === 1 && state.toAct !== 0)
      A(state.currentBet ? "call" : state.toAct % 2 ? "raise" : "check", state.currentBet ? undefined : 12);
    director.applyScene(sceneFromState(state));
  } else if (name === "showdown") {
    rig6(["Ah Qh", null, null, "Kd Ks"], "Kh 9h 2c 5h 9s");
    const sd = [];
    const script = [["raise", 6], ["fold"], ["fold"], ["call"], ["fold"], ["fold"]]; // seat 3 opens, the hero calls on the button
    for (const [t, to] of script) if (!state.handOver && state.street === 0) A(t, to);
    while (!state.handOver) sd.push(...A(state.currentBet ? "call" : state.street === 3 && state.toAct === 0 ? "raise" : "check", 20));
    const show = sd.find((e) => e.type === "showdown");
    const award = sd.find((e) => e.type === "award");
    const winner = show.hands.find((h) => award.seats.includes(h.seat));
    director.applyScene(sceneFromState(state, { reveal: show.hands.map((h) => h.seat), highlight: { cards: winner.best5 } }));
    rig.snap("showdown");
  }
  for (let i = 0; i < 3; i++) stage.frame(400); // settle lifts and the spotlight
}

// ---------- bots play; the hero's seat follows the coach ----------
async function autoplay({ stage }, director) {
  const n = num("n", 6);
  const rng = core.makeRng(num("seed", Date.now() % 100000));
  const state = core.createGame({ n }),
    reads = core.createReads(n),
    styles = core.assignStyles(n, rng);
  director.setSpeed(num("speed", 1));
  director.applyScene(sceneFromState(state));
  addEventListener("keydown", (e) => e.code === "Space" && director.skip());
  for (;;) {
    let events = core.observeAll(reads, core.startHand(state, rng));
    await director.play(events, state);
    while (!state.handOver) {
      await stage.anim.wait(0.5);
      const seat = state.toAct;
      let action;
      if (seat === 0) {
        const legal = core.legalActions(state);
        const a = core.analyze(state, reads, rng, { light: true, iters: 500 });
        action =
          a.rec.action === "raise" ? { seat, type: "raise", to: core.clampRaise(legal, a.rec.size || legal.minTo) } : { seat, type: a.rec.action };
      } else action = core.botDecide(state, reads, seat, core.STYLES[styles[seat]], rng);
      events = core.observeAll(reads, core.apply(state, action));
      await director.play(events, state);
    }
    await stage.anim.wait(2.2);
  }
}

boot();
