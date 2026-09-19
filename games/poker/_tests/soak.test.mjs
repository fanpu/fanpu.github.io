import test from "node:test";
import assert from "node:assert/strict";
import * as core from "../js/core/index.js";

const {
  makeRng,
  createGame,
  startHand,
  apply,
  legalActions,
  clampRaise,
  pot,
  createReads,
  observeAll,
  STYLES,
  assignStyles,
  botDecide,
  analyze,
  grade,
  BUYIN,
  BB,
} = core;

// A whole session as the app will run it: bots in every seat but the hero's, the hero playing the coach's advice.
function session(n, seed, hands) {
  const rng = makeRng(seed);
  const state = createGame({ n }),
    reads = createReads(n),
    styles = assignStyles(n, rng);
  const log = [];
  let decisions = 0;
  for (let h = 0; h < hands; h++) {
    const events = observeAll(reads, startHand(state, rng));
    while (!state.handOver) {
      const seat = state.toAct;
      let action;
      if (seat === 0) {
        const legal = legalActions(state);
        const a = analyze(state, reads, rng, { light: true, iters: 300 });
        const type = a.rec.action;
        assert.ok(
          type === "fold" || (type === "check" && legal.canCheck) || (type === "call" && legal.toCall > 0) || (type === "raise" && legal.canRaise),
          `illegal advice ${type} in ${a.category}`
        );
        assert.equal(grade(a.rec, type), "correct");
        assert.ok(a.eq >= 0 && a.eq <= 1);
        action = type === "raise" ? { seat, type, to: clampRaise(legal, a.rec.size || legal.minTo) } : { seat, type };
        log.push(a.category + ":" + type + (action.to ? "@" + action.to : ""));
        decisions++;
      } else action = botDecide(state, reads, seat, STYLES[styles[seat]], rng, 100);
      events.push(...observeAll(reads, apply(state, action)));
    }
    log.push(JSON.stringify(events.at(-1).net));
    assert.equal(state.players.reduce((s, p) => s + p.stack, 0) + pot(state), (n + state.rebuys) * BUYIN);
  }
  // Net is summed from hand results, so rebuys do not distort it.
  return { log, decisions, hands, heroNet: log.filter((l) => l[0] === "{").reduce((s, l) => s + JSON.parse(l)[0], 0) };
}

for (const n of [2, 6, 9]) {
  test(`${n}-handed: 1,000 hands of bots against the coach's own advice`, (t) => {
    const s = session(n, 500 + n, 1000);
    assert.ok(s.decisions > 300);
    // Not asserted: 1,000 hands is far too few to pin a win rate down. Over 9,600 six-handed hands the coach's line measured +116 ± 66 bb/100.
    t.diagnostic(`${n}-handed: ${s.decisions} hero decisions, coach line ${((s.heroNet / BB / s.hands) * 100).toFixed(0)} bb/100 in this sample`);
  });
}

test("a session is reproducible from its seed", () => {
  assert.deepEqual(session(6, 42, 120).log, session(6, 42, 120).log);
  assert.notDeepEqual(session(6, 42, 20).log, session(6, 43, 20).log);
});

test("the core's public surface", () => {
  for (const name of [
    "makeRng",
    "parseCards",
    "eval7",
    "best5",
    "handPct",
    "seatPos",
    "createGame",
    "startHand",
    "apply",
    "legalActions",
    "snapshot",
    "replay",
    "fmt",
    "observe",
    "readOf",
    "simulate",
    "outsAnalysis",
    "rangeLists",
    "botDecide",
    "analyze",
    "recommend",
    "grade",
    "lessonFor",
  ])
    assert.equal(typeof core[name], "function", name);
});
