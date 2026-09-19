import { newDeck, shuffle } from "./cards.js";
import { eval7, describeScore, best5 } from "./eval.js";
import { seatPos } from "./positions.js";

// No-limit hold'em, event-sourced. The engine knows the rules and nothing else: no ranges, no coach,
// no animation. Every change to the table is reported as an event, so the stage can animate it, the
// review can rewind to it and tests can assert on it. State is plain JSON and is updated in place.
export const BB = 2,
  SB = 1,
  BUYIN = 200; // 100 big blinds
export const STREET_NAMES = ["preflop", "flop", "turn", "river"];
const DEFAULT_NAMES = ["You", "Ava", "Ben", "Chen", "Dara", "Eli", "Fumi", "Gus", "Hana"];

export function createGame({ n, names = DEFAULT_NAMES }) {
  if (!(n >= 2 && n <= 9)) throw new Error("createGame: n must be 2..9");
  const players = [];
  for (let id = 0; id < n; id++)
    players.push({
      id,
      name: names[id],
      stack: BUYIN,
      startStack: BUYIN,
      cards: [],
      bet: 0,
      invested: 0,
      folded: false,
      allIn: false,
      needAct: false,
      raiseLocked: false,
    });
  return {
    config: { n, sb: SB, bb: BB, buyin: BUYIN },
    handNo: 0,
    rebuys: 0,
    dealer: -1,
    street: 0,
    board: [],
    deck: [],
    toAct: -1,
    cursor: -1,
    currentBet: 0,
    minRaise: BB,
    raises: 0,
    lastAggressor: -1,
    handOver: true,
    result: null,
    actions: [],
    players,
  };
}

export const pot = (state) => state.players.reduce((s, p) => s + p.invested, 0);
export const alive = (state) => state.players.filter((p) => !p.folded);
export const actors = (state) => state.players.filter((p) => !p.folded && !p.allIn);

function put(p, amount) {
  const a = Math.min(amount, p.stack);
  p.stack -= a;
  p.bet += a;
  p.invested += a;
  if (p.stack === 0) p.allIn = true;
  return a;
}

export function startHand(state, rng) {
  if (!state.handOver) throw new Error("startHand: the current hand is not over");
  const n = state.config.n;
  state.handNo++;
  state.dealer = (state.dealer + 1) % n;
  Object.assign(state, { street: 0, board: [], handOver: false, result: null, actions: [], raises: 0, currentBet: BB, minRaise: BB });
  state.deck = shuffle(newDeck(), rng);
  const rebuys = [];
  for (const p of state.players) {
    if (p.stack <= 0) {
      p.stack = BUYIN;
      state.rebuys++;
      rebuys.push(p.id);
    }
    Object.assign(p, {
      startStack: p.stack,
      cards: [state.deck.pop(), state.deck.pop()],
      bet: 0,
      invested: 0,
      folded: false,
      allIn: false,
      needAct: true,
      raiseLocked: false,
    });
  }
  const sbSeat = n === 2 ? state.dealer : (state.dealer + 1) % n,
    bbSeat = (sbSeat + 1) % n;
  const events = [{ type: "handStart", handNo: state.handNo, dealer: state.dealer, sbSeat, bbSeat, rebuys }];
  events.push({ type: "post", seat: sbSeat, kind: "sb", amount: put(state.players[sbSeat], SB) });
  events.push({ type: "post", seat: bbSeat, kind: "bb", amount: put(state.players[bbSeat], BB) });
  const hands = {};
  for (const p of state.players) hands[p.id] = p.cards;
  events.push({ type: "deal", hands });
  state.lastAggressor = bbSeat;
  state.cursor = bbSeat;
  advance(state, events);
  return events;
}

// What the player to act may do, or null when the hand is over.
export function legalActions(state) {
  if (state.handOver || state.toAct < 0) return null;
  const p = state.players[state.toAct];
  const toCall = Math.min(state.currentBet - p.bet, p.stack);
  const maxTo = p.bet + p.stack;
  const someoneCanRespond = state.players.some((q) => q !== p && !q.folded && !q.allIn);
  return {
    seat: p.id,
    toCall,
    canCheck: toCall === 0,
    canRaise: p.stack > toCall && !p.raiseLocked && someoneCanRespond,
    minTo: Math.min(maxTo, state.currentBet + state.minRaise),
    maxTo,
    pot: pot(state),
  };
}
export const clampRaise = (legal, to) => Math.max(legal.minTo, Math.min(legal.maxTo, Math.round(to)));

// action = { seat, type: 'fold' | 'check' | 'call' | 'raise', to? }. "raise" covers betting too; `to` is the
// total the player will have in front of them this street. Throws on anything illegal, before touching state.
export function apply(state, action) {
  const legal = legalActions(state);
  if (!legal) throw new Error("apply: the hand is over");
  if (action.seat !== legal.seat) throw new Error("apply: not seat " + action.seat + "'s turn (seat " + legal.seat + " is to act)");
  const { type, to } = action;
  if (type === "check" && !legal.canCheck) throw new Error("apply: cannot check facing a bet");
  if (type === "call" && legal.toCall === 0) throw new Error("apply: nothing to call");
  if (type === "raise" && !(legal.canRaise && Number.isInteger(to) && to >= legal.minTo && to <= legal.maxTo))
    throw new Error("apply: illegal raise to " + to + " (allowed " + (legal.canRaise ? legal.minTo + ".." + legal.maxTo : "none") + ")");
  if (!["fold", "check", "call", "raise"].includes(type)) throw new Error("apply: unknown action " + type);

  const p = state.players[legal.seat];
  const ev = {
    type: "action",
    seat: p.id,
    kind: type,
    added: 0,
    to: p.bet,
    allIn: false,
    full: false,
    street: state.street,
    toCallBefore: legal.toCall,
    potBefore: legal.pot,
    currentBetBefore: state.currentBet,
    raisesBefore: state.raises,
    boardLen: state.board.length,
    pos: seatPos(state, p.id),
  };
  state.actions.push(type === "raise" ? { seat: p.id, type, to } : { seat: p.id, type });

  if (type === "fold") p.folded = true;
  else if (type === "call") ev.added = put(p, legal.toCall);
  else if (type === "raise") {
    if (state.currentBet === 0) ev.kind = "bet";
    const increment = to - state.currentBet;
    ev.full = increment >= state.minRaise;
    ev.added = put(p, to - p.bet);
    state.currentBet = to;
    state.raises++;
    state.lastAggressor = p.id;
    if (ev.full) state.minRaise = increment;
    for (const q of state.players) {
      if (q === p || q.folded || q.allIn) continue;
      // A full raise reopens the betting for everyone. A short all-in only asks those who had already
      // acted to call the difference or fold; it does not let them raise again.
      if (ev.full) q.raiseLocked = false;
      else if (!q.needAct) q.raiseLocked = true;
      q.needAct = true;
    }
  }
  p.needAct = false;
  ev.to = p.bet;
  ev.allIn = p.allIn;
  state.cursor = p.id;
  const events = [ev];
  advance(state, events);
  return events;
}

// Move to the next decision: the next player to act, or the next street, or the end of the hand.
function advance(state, events) {
  const n = state.config.n;
  state.toAct = -1;
  if (alive(state).length === 1) return finish(state, events, false);
  for (let k = 1; k <= n; k++) {
    const p = state.players[(state.cursor + k) % n];
    if (!p.folded && !p.allIn && p.needAct) {
      state.toAct = p.id;
      return;
    }
  }
  refundUncalled(state, events);
  if (state.street === 3) return finish(state, events, true);
  const runOut = actors(state).length <= 1; // nobody left to bet against: deal the rest face up
  do {
    state.street++;
    for (const p of state.players) Object.assign(p, { bet: 0, raiseLocked: false, needAct: !p.folded && !p.allIn && !runOut });
    Object.assign(state, { currentBet: 0, minRaise: BB, raises: 0 });
    const cards = state.street === 1 ? [state.deck.pop(), state.deck.pop(), state.deck.pop()] : [state.deck.pop()];
    state.board.push(...cards);
    events.push({ type: "street", street: state.street, cards, pot: pot(state) });
  } while (runOut && state.street < 3);
  if (runOut) return finish(state, events, true);
  state.cursor = state.dealer;
  advance(state, events);
}

// The part of the biggest bet that nobody matched goes back before anything else happens.
function refundUncalled(state, events) {
  const byBet = state.players.slice().sort((a, b) => b.bet - a.bet);
  const amount = byBet[0].bet - byBet[1].bet;
  if (amount <= 0) return;
  const p = byBet[0];
  p.bet -= amount;
  p.invested -= amount;
  p.stack += amount;
  p.allIn = false;
  events.push({ type: "refund", seat: p.id, amount });
}

function finish(state, events, showdown) {
  const n = state.config.n;
  if (!showdown) refundUncalled(state, events);
  const contenders = alive(state);
  const scores = {};
  if (showdown) {
    const hands = contenders.map((p) => {
      const all = [...p.cards, ...state.board];
      scores[p.id] = eval7(all);
      return { seat: p.id, cards: p.cards, score: scores[p.id], desc: describeScore(scores[p.id]), best5: best5(all) };
    });
    events.push({ type: "showdown", hands });
  } else scores[contenders[0].id] = 1;

  // Side pots, one per distinct amount invested. Adjacent pots with the same contenders are merged.
  const fromButton = (p) => (p.id - state.dealer - 1 + n) % n; // 0 = first seat left of the button
  const levels = [...new Set(state.players.map((p) => p.invested).filter((v) => v > 0))].sort((a, b) => a - b);
  const pots = [];
  let prev = 0,
    carry = 0;
  for (const level of levels) {
    let amount = carry;
    for (const p of state.players) amount += Math.max(0, Math.min(p.invested, level) - prev);
    prev = level;
    const eligible = contenders.filter((p) => p.invested >= level);
    if (!eligible.length) {
      carry = amount; // everyone who put this much in has folded: it rolls into the next pot down
      continue;
    }
    carry = 0;
    const key = eligible.map((p) => p.id).join(",");
    const last = pots[pots.length - 1];
    if (last && last.key === key) last.amount += amount;
    else pots.push({ key, amount, eligible });
  }
  if (carry && pots.length) pots[pots.length - 1].amount += carry;

  const awards = [];
  pots.forEach((po, potIndex) => {
    const best = Math.max(...po.eligible.map((p) => scores[p.id]));
    const winners = po.eligible.filter((p) => scores[p.id] === best).sort((a, b) => fromButton(a) - fromButton(b));
    const share = Math.floor(po.amount / winners.length);
    let odd = po.amount - share * winners.length;
    const shares = {};
    for (const w of winners) {
      shares[w.id] = share + (odd > 0 ? 1 : 0);
      if (odd > 0) odd--;
      w.stack += shares[w.id];
    }
    const award = { type: "award", potIndex, amount: po.amount, seats: winners.map((w) => w.id), shares, contested: po.eligible.length > 1 };
    awards.push(award);
    events.push(award);
  });

  const net = {};
  for (const p of state.players) {
    net[p.id] = p.stack - p.startStack;
    p.bet = 0;
    p.invested = 0; // the chips have moved to the winners, so pot(state) is 0 once the hand is over
    p.needAct = false;
  }
  Object.assign(state, { handOver: true, toAct: -1, street: showdown ? 4 : state.street, result: { showdown, awards, net } });
  events.push({ type: "handEnd", net, showdown });
}

// A hand is reproducible from its own starting state (deck included) plus the actions taken.
export const snapshot = (state) => structuredClone(state);
export function replay(snap, actions, upto = actions.length) {
  const state = structuredClone(snap);
  const events = [];
  for (const a of actions.slice(0, upto)) events.push(...apply(state, a));
  return { state, events };
}

// Chips as big blinds: "7.5 bb".
export function fmt(chips) {
  const bb = chips / BB;
  return (Number.isInteger(bb) ? bb : bb.toFixed(1)) + " bb";
}
