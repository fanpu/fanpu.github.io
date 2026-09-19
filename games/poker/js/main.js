import * as core from "./core/index.js";
import { Stage } from "./stage/stage.js";
import { buildTable } from "./stage/table.js";
import { createCameraRig } from "./stage/shots.js";
import { createCards } from "./stage/cards3d.js";
import { createChips } from "./stage/chips3d.js";
import { createLabels } from "./stage/labels.js";
import { createFx } from "./stage/fx.js";
import { createDirector } from "./director.js";
import { createStore } from "./ui/store.js";
import { createCoach } from "./worker/coachClient.js";
import { createSession } from "./session.js";
import { createHud } from "./ui/hud.js";
import { createDock } from "./ui/dock.js";
import { createInfo } from "./ui/info.js";
import { createSettings } from "./ui/settings.js";
import { createHome } from "./ui/home.js";

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

async function boot() {
  const store = createStore();
  if (!Stage.supported()) {
    document.body.insertAdjacentHTML(
      "beforeend",
      `<div class="notice">This table is drawn with WebGL, which this browser is not offering.<br />The lessons will still work once they are built.</div>`
    );
    return;
  }
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const stage = new Stage($("stage"), { reducedMotion: reduced || !store.settings.animations, fourColour: store.settings.fourColour });
  const parts = {
    stage,
    table: buildTable(stage),
    cards: createCards(stage),
    chips: createChips(stage),
    labels: createLabels(stage, $("stage")),
    fx: createFx(stage),
  };
  parts.table.setSeats(store.settings.players);
  parts.rig = createCameraRig(stage, parts.table);
  const director = createDirector(parts);
  parts.rig.enableOrbit(stage.renderer.domElement);
  stage.start();

  // Development views (?pose=, ?demo) bypass the app.
  if (params.has("pose") || params.has("demo")) {
    const dev = await import("./dev.js");
    if (params.get("pose") === "back") dev.showArtwork(stage);
    else if (params.has("pose")) dev.showPose(params.get("pose"), parts, director);
    else dev.autoplay(parts, director);
    document.body.dataset.ready = "1";
    return;
  }

  const seed = params.has("seed") ? +params.get("seed") : (Date.now() ^ (Math.random() * 2 ** 31)) >>> 0;
  const coach = createCoach();
  const session = createSession({ store, director, coach, rng: core.makeRng(seed) });

  // ----- panels -----
  const body = document.body;
  const openInfo = (tab) => {
    if (tab) info.setTab(tab);
    body.classList.add("info-open");
    // Opening the review of a finished hand goes straight to the first decision the coach disagreed with.
    const v = store.view;
    if (v.phase === "handOver" && v.at == null && v.decisions?.length)
      session.rewind((v.decisions.find((d) => d.grade !== "correct") || v.decisions[0]).index);
  };
  const info = createInfo($("info"), { store, onClose: () => body.classList.remove("info-open"), onRewind: (k) => session.rewind(k) });
  // Left and right step through the replay; Escape returns to the result.
  addEventListener("keydown", (e) => {
    const v = store.view;
    if (v.phase !== "handOver" || !v.timeline?.length || e.metaKey || e.ctrlKey || e.altKey) return;
    const last = v.timeline.length - 1;
    if (e.key === "ArrowRight") session.rewind(v.at == null ? 0 : v.at >= last ? null : v.at + 1);
    else if (e.key === "ArrowLeft") session.rewind(v.at == null ? last : Math.max(0, v.at - 1));
    else if (e.key === "Escape") session.rewind(null);
    else return;
    e.preventDefault();
  });
  const dock = createDock($("dock"), { store, session, openInfo });
  const settings = createSettings($("settings"), {
    store,
    onChange(key, value) {
      if (key === "fourColour") stage.textures.setFourColour(value);
      if (key === "animations") stage.anim.reduced = reduced || !value;
      if (key === "speed" || key === "animations") director.setSpeed(store.settings.speed);
    },
  });
  const hud = createHud($("hud"), {
    store,
    onHome: () => go("home"),
    onLevel: (level) => go("table", level),
    onSettings: () => settings.toggle(),
  });
  const home = createHome($("home"), { store, onTable: (level) => go("table", level) });
  addEventListener("keydown", (e) => e.key === "Escape" && body.classList.remove("info-open"));
  $("scrim").addEventListener("click", () => body.classList.remove("info-open"));

  // ----- routing: home or the table -----
  async function go(mode, level = store.progress.level) {
    if (mode === "home") await session.stop();
    store.setProgress({ mode, level });
    body.dataset.mode = mode;
    body.classList.remove("info-open");
    claimSpace();
    if (mode === "table") session.start(level), session.setLevel(level);
    else parts.rig.to("idle", 1.4);
  }

  // Panels claim their share of the screen; the stage takes the rest and the camera refits.
  function claimSpace() {
    const table = body.dataset.mode === "table",
      phone = matchMedia("(max-width: 720px)").matches;
    const px = (el) => (table ? Math.round(el.getBoundingClientRect()[el === $("info") ? "width" : "height"]) : 0) + "px";
    body.style.setProperty("--stage-top", px($("hud")));
    body.style.setProperty("--stage-bottom", px($("dock")));
    body.style.setProperty("--stage-right", phone ? "0px" : px($("info")));
  }
  const watch = new ResizeObserver(claimSpace);
  [$("hud"), $("dock"), $("info")].forEach((el) => watch.observe(el));
  addEventListener("resize", claimSpace);

  store.subscribe(() => {
    body.dataset.phase = store.view.phase;
    body.dataset.level = store.view.level || store.progress.level;
    hud.render();
    dock.render();
    info.render();
    home.render();
  });

  // A scripted hero for soak runs: ?auto=coach follows the coach, ?auto=call never folds.
  if (params.has("auto")) {
    const policy = params.get("auto");
    store.subscribe(() => {
      const v = store.view;
      if (v.phase === "feedback") session.resume();
      else if (v.phase === "handOver" && !params.has("hold")) setTimeout(() => session.next(), 50);
      else if (v.phase === "hero" && v.legal && !v.analysing && v.street < (params.has("stopAt") ? +params.get("stopAt") : 9)) {
        const rec = v.analysis?.rec;
        if (policy === "coach" && rec) session.act(rec.action === "raise" ? "raise" : rec.action, rec.size);
        else session.act(v.legal.canCheck ? "check" : "call");
      }
    });
  }

  window.__poker = { core, store, session, director, coach, timings: session.timings, ...parts };
  home.render();
  const wanted = params.get("mode");
  go(
    wanted === "train" ? "table" : wanted === "prove" ? "table" : store.progress.mode === "table" ? "table" : "home",
    wanted === "prove" ? "silent" : wanted === "train" ? "guided" : undefined
  );
  body.dataset.ready = "1";
}

boot();
