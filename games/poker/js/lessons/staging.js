import { newDeck, cardKey, eval7, evalCat, best5 } from "../core/index.js";

// Turns a lesson page or a drill question into a scene for the table. The lesson content itself only draws rows
// of cards (remembered by content.js as it goes); this file decides who sits where.
const HERO = /^(your cards|you|hand a)$/i,
  BOARD = /^(board|flop|turn|river)$/i;

const seat = (i, name, cards, extra = {}) => ({
  seat: i,
  name,
  pos: "",
  stack: null,
  bet: 0,
  hasCards: !!cards,
  cards: cards || null,
  folded: false,
  allIn: false,
  acting: false,
  ...extra,
});
const emptyScene = (n) => ({
  n,
  dealer: -1,
  handNo: 0,
  seats: Array.from({ length: n }, (_, i) => seat(i, i ? "" : "You", null)),
  board: [],
  pot: 0,
  highlight: null,
  fan: [],
});

// rows: [[{ label, cards, pick? }]] as drawn by the page, first row first.
function fromRows(rows) {
  const groups = rows[0] || [];
  const players = groups.filter((g) => !BOARD.test(g.label || "")),
    board = groups.find((g) => BOARD.test(g.label || ""));
  if (!players.length && !board) return null;
  // The hero's cards go to seat 0; anyone else sits opposite (or round the table if there are several).
  const n = Math.max(2, players.length);
  const scene = emptyScene(n);
  const heroFirst = players.slice().sort((a, b) => HERO.test(b.label || "") - HERO.test(a.label || ""));
  heroFirst.forEach(
    (g, i) =>
      (scene.seats[i] = seat(i, HERO.test(g.label || "") ? "You" : g.label || (i ? "Opponent" : "You"), g.cards.length === 2 ? g.cards : null))
  );
  // A row that is not two cards (seven cards to name, say) is laid out as hole cards plus board.
  const odd = players.find((g) => g.cards.length !== 2);
  if (odd && !board) {
    scene.seats[0] = seat(0, "You", odd.cards.slice(0, 2));
    scene.board = odd.cards.slice(2, 7);
  }
  if (board) scene.board = board.cards.slice(0, 5);
  return scene;
}

// "The pot is 12 bb. Your opponent bets 8 bb." -> chips on the felt. Amounts are in big blinds; a big blind is two chips.
function moneyScene(q) {
  const pot = /pot (?:is|of) (\d+) bb/i.exec(q) || /into (?:a pot of )?(\d+) bb/i.exec(q),
    bet = /\b(?:bets?|bluff) (\d+) bb/i.exec(q);
  if (!pot || !bet) return null;
  const heroBets = /^(you|river\. you)/i.test(q.trim());
  const scene = emptyScene(2);
  scene.seats[1] = seat(1, "Opponent", null, { hasCards: true });
  scene.seats[0].hasCards = true;
  scene.pot = +pot[1] * 2;
  scene.seats[heroBets ? 0 : 1].bet = +bet[1] * 2;
  scene.seats[heroBets ? 1 : 0].acting = true;
  return scene;
}

// Six seats named by position; whoever is still in the hand has cards.
const POSITIONS = ["BTN", "SB", "BB", "UTG", "HJ", "CO"];
function positionScene(d, answered) {
  const scene = emptyScene(6);
  scene.dealer = 0;
  const everyone = /before the flop/i.test(d.q);
  POSITIONS.forEach((pos, i) => {
    const inHand = everyone || d.options.includes(pos);
    scene.seats[i] = seat(i, pos, null, { hasCards: inHand, folded: !inHand, acting: answered && d.options[d.answer] === pos });
  });
  if (!everyone) scene.board = [];
  return scene;
}

// The scene for a drill question; once answered, the table shows why.
export function sceneForDrill(name, d, rows, answered) {
  if (name === "position") return positionScene(d, answered);
  const scene = fromRows(rows) || moneyScene(d.q);
  if (!scene) return null;
  if (!answered) return scene;
  const hero = scene.seats[0].cards;
  if (name === "outs" && hero) {
    // Fan out the cards that complete the straight or flush.
    const all = [...hero, ...scene.board],
      seen = new Set(all.map(cardKey));
    scene.fan = newDeck().filter((c) => !seen.has(cardKey(c)) && evalCat(eval7([...all, c])) >= 4);
  } else if (name === "nameHand" && hero) scene.highlight = { cards: best5([...hero, ...scene.board]) };
  else if (name === "whoWins" && scene.seats[1]?.cards) {
    const hands = scene.seats.slice(0, 2).map((s) => [...s.cards, ...scene.board]);
    const winners = d.answer === 2 ? [0, 1] : [d.answer];
    scene.highlight = { cards: winners.flatMap((w) => best5(hands[w])) };
  }
  return scene;
}
export const sceneForPage = (rows) => fromRows(rows);
export const idleScene = () => emptyScene(6);
