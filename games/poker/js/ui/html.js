import { cardCanvas } from "../stage/textures.js";
import { BB, fmt } from "../core/index.js";

// Small helpers shared by the panels.
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export const pct = (v) => Math.round(v * 100) + "%";
export { fmt };

// A signed amount of chips as big blinds: "+3.5 bb", "−1 bb", "0 bb".
export function evFmt(chips) {
  const v = chips / BB;
  return (Math.abs(v) < 0.05 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(1)) + " bb";
}
export const STREETS = ["preflop", "flop", "turn", "river"];
export const actionWord = (action, betting) => (action === "raise" ? (betting ? "bet" : "raise") : action);

// Cards in the DOM are the same paintings as on the table. `null` draws a card back.
const cache = new Map();
export function cardEl(card, fourColour) {
  const key = (card ? card.r * 4 + card.s : -1) + (fourColour ? "f" : "");
  if (!cache.has(key)) cache.set(key, cardCanvas(card, { fourColour }, 160).toDataURL());
  const img = new Image();
  img.src = cache.get(key);
  img.className = "pcard";
  img.alt = card ? "23456789TJQKA"[card.r - 2] + "shdc"[card.s] : "card";
  return img;
}
// Fill every <span data-cards="As Kd"> under root with card images.
export function mountCards(root, fourColour) {
  for (const slot of root.querySelectorAll("[data-cards]")) {
    slot.textContent = "";
    for (const tok of slot.dataset.cards.split(" ").filter(Boolean))
      slot.appendChild(cardEl({ r: "23456789TJQKA".indexOf(tok[0]) + 2, s: "shdc".indexOf(tok[1]) }, fourColour));
  }
}
export const cardsAttr = (cards) => cards.map((c) => "23456789TJQKA"[c.r - 2] + "shdc"[c.s]).join(" ");

// Re-render only when what is shown would change: `sig` is any string that captures the inputs.
export function renderer(root, draw) {
  let last = null;
  return (sig, ...args) => {
    if (sig === last) return false;
    last = sig;
    draw(root, ...args);
    return true;
  };
}
