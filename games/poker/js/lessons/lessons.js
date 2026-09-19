import { DRILLS, LESSONS, takeStaged } from "./content.js";
import { sceneForDrill, sceneForPage, idleScene } from "./staging.js";
import { mountCards } from "../ui/html.js";

// Learn: the lesson text sits in a pane beside (on a phone, beneath) the table, and the table is the illustration.
// Every page and every drill question lays its cards, chips and seats out on the felt; answering a drill makes the
// table show why (the five cards that play are lifted, the outs fan out, the seat that acts is lit).
export function createLessons(root, { store, director, rig, onExit, onSound = () => {} }) {
  const at = { lesson: null, page: 0, drill: null }; // drill: { q, rows, chosen, correct }

  function stage(scene, shot = "lesson") {
    director.applyScene(scene || idleScene());
    rig.to(scene?.n === 6 ? "top" : shot, 0.8);
  }
  function newQuestion(name, keep) {
    takeStaged();
    const q = DRILLS[name]();
    at.drill = { name, q, rows: takeStaged(), chosen: null, correct: keep?.correct || 0 };
  }
  function open(lesson, page = 0) {
    at.lesson = lesson;
    at.page = page;
    const pg = lesson == null ? null : LESSONS[lesson].pages[page];
    at.drill = null;
    if (pg?.drill) newQuestion(pg.drill);
    render();
    root.scrollTop = 0;
  }

  function listHTML() {
    const done = store.progress.done,
      next = LESSONS.find((l) => !done[l.id]);
    return `<nav class="crumb"><button data-exit>‹ Home</button><span>${LESSONS.filter((l) => done[l.id]).length} of ${
      LESSONS.length
    } done</span></nav>
      <h1>Learn poker from zero</h1>
      <p class="lead">${
        LESSONS.length
      } short lessons. Each explains one idea in plain words, shows where the numbers come from, then drills it on the table with hands dealt at random until it sticks.</p>
      ${
        next
          ? `<button class="primary wide" data-lesson="${LESSONS.indexOf(next)}">${Object.keys(done).length ? "Continue" : "Start"}: ${
              next.title
            }</button>`
          : ""
      }
      <ol class="lessons">${LESSONS.map(
        (l, i) => `<li><button data-lesson="${i}" class="${done[l.id] ? "done" : ""}"><b>${l.title}</b><span>${l.blurb}</span></button></li>`
      ).join("")}</ol>`;
  }

  function drillHTML(pg) {
    const d = at.drill,
      q = d.q,
      answered = d.chosen !== null;
    let h = `<div class="pips">${Array.from({ length: pg.need }, (_, i) => `<i class="${i < d.correct ? "on" : ""}"></i>`).join("")}<span>${
      d.correct >= pg.need ? "Drill complete. Keep going if you like." : `${d.correct} of ${pg.need} correct`
    }</span></div>`;
    h += `<div class="drill"><p class="q">${q.q}</p><div class="visual">${q.visual || ""}</div><div class="opts">${q.options
      .map(
        (o, i) =>
          `<button data-answer="${i}" class="${answered ? (i === q.answer ? "right" : i === d.chosen ? "wrong" : "") : ""}" ${
            answered ? "disabled" : ""
          }><kbd>${i + 1}</kbd>${o}</button>`
      )
      .join("")}</div>`;
    if (answered)
      h += `<div class="fb"><p class="verd ${d.chosen === q.answer ? "right" : "wrong"}">${
        d.chosen === q.answer ? "Correct." : "Not quite. The answer is " + q.options[q.answer] + "."
      }</p>${q.explain}</div><button class="primary" data-another><kbd>Space</kbd>Another question</button>`;
    return h + "</div>";
  }

  function render() {
    root.hidden = store.progress.mode !== "learn";
    if (root.hidden) return;
    if (at.lesson == null) {
      root.innerHTML = listHTML();
      stage(null, "idle");
      return;
    }
    const L = LESSONS[at.lesson],
      pg = L.pages[at.page],
      last = at.page === L.pages.length - 1,
      nextL = LESSONS[at.lesson + 1];
    takeStaged();
    const body = pg.drill ? drillHTML(pg) : pg.h();
    const rows = pg.drill ? at.drill.rows : takeStaged();
    const open = pg.drill && at.drill.correct < pg.need;
    root.innerHTML = `<nav class="crumb"><button data-list>‹ All lessons</button><span>Lesson ${at.lesson + 1} · step ${at.page + 1} of ${
      L.pages.length
    }</span></nav>
      <h1>${pg.t}</h1>${body}
      <div class="nav">${at.page > 0 ? `<button class="quiet" data-go="${at.page - 1}">Back</button>` : "<span></span>"}<button class="${
        open ? "quiet" : "primary"
      }" data-go="${last ? "finish" : at.page + 1}">${
        last ? (nextL ? "Finish, then: " + nextL.title : "Finish the course") : open ? "Skip this drill" : "Next"
      }</button></div>`;
    mountCards(root, store.settings.fourColour);
    pg.m?.(root);
    // The table illustrates whatever the page is about. With something staged, the panel's own copy of those cards steps back.
    const scene = pg.drill ? sceneForDrill(pg.drill, at.drill.q, rows, at.drill.chosen !== null) : sceneForPage(rows);
    root.classList.toggle("staged", !!scene);
    stage(scene);
  }

  root.addEventListener("click", (e) => {
    const t = e.target.closest("[data-exit],[data-list],[data-lesson],[data-go],[data-answer],[data-another]");
    if (!t || t.disabled) return;
    const d = t.dataset;
    if ("exit" in d) onExit();
    else if ("list" in d) open(null);
    else if (d.lesson) open(+d.lesson);
    else if (d.answer) answer(+d.answer);
    else if ("another" in d) another();
    else if (d.go === "finish") {
      store.completeLesson(LESSONS[at.lesson].id);
      open(at.lesson + 1 < LESSONS.length ? at.lesson + 1 : null);
    } else open(at.lesson, +d.go);
  });
  function answer(i) {
    const d = at.drill;
    if (!d || d.chosen !== null || i >= d.q.options.length) return;
    d.chosen = i;
    if (i === d.q.answer) d.correct++;
    onSound(i === d.q.answer ? "right" : "wrong");
    render();
  }
  function another() {
    const pg = LESSONS[at.lesson].pages[at.page];
    newQuestion(pg.drill, at.drill);
    render();
    root.querySelector(".drill")?.scrollIntoView({ block: "start" });
  }
  // Keys: 1-4 answer, Space or Enter for another question.
  addEventListener("keydown", (e) => {
    if (store.progress.mode !== "learn" || !at.drill || e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^[1-4]$/.test(e.key)) answer(+e.key - 1);
    else if ((e.key === " " || e.key === "Enter") && at.drill.chosen !== null && !e.target.closest?.("button")) e.preventDefault(), another();
  });

  return {
    render,
    open,
    openById: (id) =>
      open(
        Math.max(
          0,
          LESSONS.findIndex((l) => l.id === id)
        )
      ),
    get count() {
      return LESSONS.length;
    },
  };
}
