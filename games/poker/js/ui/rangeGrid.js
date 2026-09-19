import { RANKS } from "../core/index.js";

// The 13 x 13 starting-hand grid: pairs on the diagonal, suited hands above it, offsuit below.
// alphaOf(key, fullCombos) -> 0..1 shading for that hand; `highlight` is the hero's own hand.
export function rangeGrid(alphaOf, highlight) {
  let h = '<div class="rgrid" role="img" aria-label="Range grid"><span></span>';
  for (let j = 0; j < 13; j++) h += "<span>" + RANKS[12 - j] + "</span>";
  for (let i = 0; i < 13; i++) {
    h += "<span>" + RANKS[12 - i] + "</span>";
    for (let j = 0; j < 13; j++) {
      const x = RANKS[12 - i],
        y = RANKS[12 - j];
      const key = i === j ? x + x : i < j ? x + y + "s" : y + x + "o";
      const a = alphaOf(key, i === j ? 6 : i < j ? 4 : 12);
      h += `<i title="${key}"${key === highlight ? ' class="me"' : ""}${a > 0 ? ` style="--a:${a.toFixed(2)}"` : ""}></i>`;
    }
  }
  return h + "</div>";
}
