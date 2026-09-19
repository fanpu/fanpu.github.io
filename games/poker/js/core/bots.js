import { handPct } from "./preflop.js";
import { simulate } from "./equity.js";
import { outsAnalysis } from "./analysis.js";
import { legalActions, clampRaise, alive, BB } from "./engine.js";
import { rangeOf } from "./reads.js";
import { shuffle } from "./cards.js";

// Five opponents to learn to tell apart. vpip and pfr are the share of hands (top x%) a style enters and
// raises with; aggr is how readily it bets after the flop; bluff is how often it fires with nothing.
export const STYLES = [
  { name: "Nit", vpip: 14, pfr: 11, aggr: 0.35, bluff: 0.05, blurb: "very tight; a raise means a real hand" },
  { name: "TAG", vpip: 22, pfr: 18, aggr: 0.65, bluff: 0.12, blurb: "tight and aggressive; solid" },
  { name: "LAG", vpip: 34, pfr: 27, aggr: 0.8, bluff: 0.22, blurb: "loose and aggressive; bluffs more" },
  { name: "Station", vpip: 45, pfr: 9, aggr: 0.2, bluff: 0.05, blurb: "calls far too much; never bluff them" },
  { name: "Maniac", vpip: 55, pfr: 45, aggr: 0.95, bluff: 0.35, blurb: "raises with anything; let them hang themselves" },
];

// Who sits where is shuffled per session, so the table has to be read rather than remembered.
// Returns a style index per seat; seat 0 (the hero) is null.
export function assignStyles(n, rng) {
  const pool = [1, 3, 2, 0, 4, 1, 2, 3].slice(0, n - 1); // small tables still get a mix of types
  return [null, ...shuffle(pool, rng)];
}

const ANY = { pct: 100, filters: [] };

// Decide for the player to act. Always returns an action the engine will accept.
export function botDecide(state, reads, seat, style, rng, iters = 250) {
  const o = legalActions(state);
  if (!o || o.seat !== seat) throw new Error("botDecide: seat " + seat + " is not to act");
  const [type, to] = choose(state, reads, seat, style, o, rng, iters);
  if (type === "raise") {
    if (o.canRaise) return { seat, type, to: clampRaise(o, to) };
    return { seat, type: o.canCheck ? "check" : "call" };
  }
  if (o.canCheck) return { seat, type: "check" }; // never fold, or "call", for free
  return { seat, type };
}

function choose(state, reads, seat, st, o, rng, iters) {
  const p = state.players[seat],
    potNow = o.pot;
  const rnd = rng();
  if (state.street === 0) {
    const pct = handPct(p.cards[0], p.cards[1]);
    const limpers = reads.players.filter((q) => q.tag === "limp").length;
    if (state.raises === 0) {
      if (pct <= st.pfr) return ["raise", BB * (2.5 + limpers)];
      if (pct <= st.vpip || (rnd < 0.5 && st.name === "Station" && pct <= 60)) return ["call"];
      return ["fold"];
    }
    if (state.raises === 1) {
      // 3-bet the top third of the raising range for value, and now and then as a bluff. (Tuned down from 0.4 and
      // 0.5: at those values a six-handed table 3-bet a third of the hero's opens, half of them from the maniac alone.)
      if (pct <= st.pfr * 0.33 || (rnd < st.bluff * 0.3 && pct <= st.vpip)) return ["raise", state.currentBet * 3.2];
      if (pct <= st.vpip * 0.75) return ["call"];
      return ["fold"];
    }
    if (pct <= Math.max(2.5, st.pfr * 0.15)) return ["raise", o.maxTo];
    if (pct <= Math.max(6, st.vpip * 0.35)) return ["call"];
    return ["fold"];
  }
  const others = alive(state).filter((q) => q !== p);
  const eq = simulate(
    p.cards,
    state.board,
    others.map(() => ANY),
    iters,
    rng
  );
  const oa = outsAnalysis(p.cards, state.board);
  const drawy = !!oa && oa.outs >= 8;
  if (o.canCheck) {
    if (eq > 0.62 && rnd < 0.85) return ["raise", potNow * 0.65];
    if (drawy && rnd < st.aggr) return ["raise", potNow * 0.5];
    if (eq > 0.45 && rnd < st.aggr * 0.5) return ["raise", potNow * 0.5];
    if (rnd < st.bluff) return ["raise", potNow * 0.45];
    return ["check"];
  }
  const req = o.toCall / (potNow + o.toCall);
  // Facing a bet, bots judge their hand against the ranges the action implies, not against random cards.
  const eqR = simulate(
    p.cards,
    state.board,
    others.map((q) => rangeOf(reads, q.id)),
    iters,
    rng
  );
  if (eq > 0.74 && eqR > 0.6 && rnd < st.aggr) return ["raise", state.currentBet * 2.6];
  if (eqR > req * 1.05) return ["call"];
  if (drawy && eqR > req * 0.8) return ["call"];
  if (rnd < st.bluff * 0.35) return ["raise", state.currentBet * 2.6];
  if (st.name === "Station" && rnd < 0.5 && eqR > req * 0.7) return ["call"];
  return ["fold"];
}
