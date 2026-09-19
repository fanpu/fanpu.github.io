import * as core from "./core/index.js";
import { sceneFromState } from "./director.js";
import { cardCanvas } from "./stage/textures.js";

// Development views, reached by URL and used by _tools/shoot.mjs:
//   ?pose=<name>   one fixed scene, no animation (table, cards, preflop, flop, showdown, nine, headsup, back)
//   ?demo          bots play every seat, the hero's following the coach's advice
const { parseCards: P } = core;
const params = new URLSearchParams(location.search);
const num = (key, fallback) => (params.has(key) ? +params.get(key) : fallback);

// ---------- the painted artwork at full size, to check it without the table in the way ----------
export function showArtwork(stage) {
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
export function showPose(name, { stage, rig, cards }, director) {
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
export async function autoplay({ stage }, director) {
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
