import test from "node:test";
import assert from "node:assert/strict";
import { scorecardRows, agreement } from "../js/ui/scorecard.js";

test("rows are worst first, with a lesson for each; ties go to the bigger sample", () => {
  const stats = {
    decisions: 40,
    correct: 25,
    acceptable: 6,
    mistakes: 9,
    byCat: {
      "Facing a bet": { n: 10, score: 8, mistakes: 1 },
      "Bet or check": { n: 20, score: 10, mistakes: 8 },
      "Open or fold": { n: 4, score: 2, mistakes: 2 },
      Mystery: { n: 6, score: 6, mistakes: 0 },
    },
  };
  const rows = scorecardRows(stats);
  assert.deepEqual(
    rows.map((r) => [r.category, r.accuracy, r.lessonId]),
    [
      ["Bet or check", 0.5, "betting"],
      ["Open or fold", 0.5, "start"],
      ["Facing a bet", 0.8, "odds"],
      ["Mystery", 1, "equity"],
    ]
  );
  assert.equal(agreement(stats), (25 + 3) / 40);
  assert.equal(agreement({ decisions: 0 }), null);
  assert.deepEqual(scorecardRows({ byCat: {} }), []);
});
