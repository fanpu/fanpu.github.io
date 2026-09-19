import { randInt } from "./rng.js";

// A card is { r, s }: r is 2..14 (14 = ace), s indexes SUITS.
export const RANKS = "23456789TJQKA";
export const SUITS = ["♠", "♥", "♦", "♣"];
export const RANK_NAMES = { 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "Ten", 11: "Jack", 12: "Queen", 13: "King", 14: "Ace" };

export function newDeck() {
  const d = [];
  for (let s = 0; s < 4; s++) for (let r = 2; r <= 14; r++) d.push({ r, s });
  return d;
}
export function shuffle(a, rng) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = randInt(rng, i + 1);
    const t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
  return a;
}
export const cardStr = (c) => RANKS[c.r - 2] + SUITS[c.s];
export const cardKey = (c) => c.r * 4 + c.s;

// "As Kd 7h" -> cards. Suit letters: s h d c.
export function parseCards(text) {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((tok) => {
      const r = RANKS.indexOf(tok[0]) + 2,
        s = "shdc".indexOf(tok[1]);
      if (tok.length !== 2 || r < 2 || s < 0) throw new Error("parseCards: bad card " + JSON.stringify(tok));
      return { r, s };
    });
}
