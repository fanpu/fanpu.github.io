import { TIER_END } from "./preflop.js";

// Names by distance clockwise from the button: index 0 is the button.
export function positionNames(n) {
  if (n === 2) return ["BTN/SB", "BB"];
  if (n === 3) return ["BTN", "SB", "BB"];
  const rest = n - 3;
  const names = ["BTN", "SB", "BB"];
  const early = ["UTG", "UTG+1", "UTG+2", "LJ", "MP", "MP+1"];
  const tail = rest >= 2 ? ["HJ", "CO"] : ["CO"];
  for (let i = 0; i < rest - tail.length; i++) names.push(early[i]);
  names.push(...tail);
  return names;
}
const fromButton = (state, seat) => (seat - state.dealer + state.config.n) % state.config.n;
export const seatPos = (state, seat) => positionNames(state.config.n)[fromButton(state, seat)];

// Seats before the button: 1 = CO, 2 = HJ, 3 = first seat at a six-player table, and so on.
export function seatsBeforeButton(posn, n) {
  if (posn === "CO") return 1;
  if (posn === "HJ") return 2;
  const i = ["UTG", "UTG+1", "UTG+2", "LJ", "MP", "MP+1"].indexOf(posn);
  if (i < 0) return 0;
  const count = Math.max(1, n - 3 - (n - 3 >= 2 ? 2 : 1));
  return 2 + (count - i);
}

// Share of hands (top x%) that a sound player opens from each position.
export function openPct(posn, n) {
  const R = Math.round;
  if (posn === "BTN/SB") return 75;
  if (posn === "BB") return 100;
  if (posn === "BTN") return R(TIER_END[7]);
  if (posn === "SB") return 40;
  const d = seatsBeforeButton(posn, n || 6);
  return d <= 1 ? R(TIER_END[6]) : d === 2 ? R(TIER_END[5]) : d === 3 ? R(TIER_END[4]) : d === 4 ? R(TIER_END[3]) : d === 5 ? R(TIER_END[2]) : 10;
}

// How wide the big blind continues against one raise, by where the raise came from.
export function bbDefendPct(raiserPos) {
  if (raiserPos === "SB" || raiserPos === "BTN/SB") return 65;
  if (raiserPos === "BTN") return 48;
  if (raiserPos === "CO") return 40;
  if (raiserPos === "HJ") return 34;
  return 30;
}

// Will the hero (seat 0) act after every listed opponent on the flop, turn and river?
// Postflop order runs clockwise from the small blind (1 from the button) round to the button (0), who is last.
// The old single-file trainer had this comparison inverted for every seat except the button and the blinds.
export function heroInPosition(state, oppSeats) {
  const hd = fromButton(state, 0);
  if (hd === 0) return true;
  return oppSeats.every((s) => {
    const od = fromButton(state, s);
    return od !== 0 && od < hd;
  });
}
