import { newDeck, cardKey, parseCards } from "../../js/core/cards.js";
import { makeRng } from "../../js/core/rng.js";
import { createGame, startHand, apply, pot } from "../../js/core/engine.js";

// Start a hand with chosen stacks, hole cards and board. hands: ["As Ks", "Qd Qc", ...] by seat; board: "2c 7d 9h Js 3s".
export function rigged({ n, stacks, dealer = 0, hands = [], board = "", seed = 1 }) {
  const state = createGame({ n });
  if (stacks) stacks.forEach((s, i) => (state.players[i].stack = s));
  state.dealer = (dealer - 1 + n) % n; // startHand moves the button on by one
  const events = startHand(state, makeRng(seed));
  const holes = hands.map((h) => (h ? parseCards(h) : null));
  const boardCards = parseCards(board);
  const used = new Set([...holes.filter(Boolean).flat(), ...boardCards].map(cardKey));
  const rest = newDeck().filter((c) => !used.has(cardKey(c)));
  state.players.forEach((p, i) => (p.cards = holes[i] || [rest.pop(), rest.pop()]));
  state.deck = [...rest, ...boardCards.slice().reverse()]; // the engine deals with deck.pop()
  for (const e of events) if (e.type === "deal") state.players.forEach((p) => (e.hands[p.id] = p.cards));
  return { state, events };
}

// Apply an action for whoever is to act. Returns the events.
export const A = (state, type, to) => apply(state, { seat: state.toAct, type, ...(to === undefined ? {} : { to }) });
export const chips = (state) => state.players.reduce((s, p) => s + p.stack, 0) + pot(state);
export const ofType = (events, type) => events.filter((e) => e.type === type);
