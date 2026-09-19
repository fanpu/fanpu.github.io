// Dev-only: play real hands in the real browser with a scripted hero, and fail on console errors or stalls.
//   node soak.mjs [hands=40] [mode=train|prove] [auto=coach|call]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = Object.fromEntries(process.argv.slice(2).map((a) => a.split("=")));
const hands = +(arg.hands || 40),
  mode = arg.mode || "train",
  auto = arg.auto || "coach",
  players = +(arg.players || 6);
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p.endsWith("/")) p += "index.html";
  const file = path.join(root, p);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 800 });
const problems = [];
page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
page.on(
  "console",
  (m) => m.type() === "error" && !/game-analytics|favicon|404/.test(m.text() + m.location()?.url) && problems.push("console: " + m.text())
);
// Animations off: the table runs as fast as it can think.
await page.evaluateOnNewDocument(
  (players) => localStorage.setItem("pokerTrainer.prefs", JSON.stringify({ animations: false, players, pause: "always" })),
  players
);
await page.goto(`http://127.0.0.1:${server.address().port}/?mode=${mode}&auto=${auto}&seed=${arg.seed || 21}`);

const t0 = Date.now();
let last = { handNo: 0, at: Date.now() },
  worstStall = 0;
for (;;) {
  await new Promise((r) => setTimeout(r, 500));
  const s = await page.evaluate(() => ({
    handNo: window.__poker?.store.view.handNo || 0,
    phase: document.body.dataset.phase,
    coach: window.__poker?.coach.mode,
  }));
  if (s.handNo !== last.handNo) last = { handNo: s.handNo, at: Date.now() };
  worstStall = Math.max(worstStall, Date.now() - last.at);
  if (s.handNo > hands || problems.length || Date.now() - last.at > 30000) {
    if (Date.now() - last.at > 30000) problems.push(`stalled in phase "${s.phase}" on hand ${s.handNo}`);
    break;
  }
}
const report = await page.evaluate(() => {
  const { store, coach } = window.__poker;
  const st = store.stats[store.view.level];
  return {
    coach: coach.mode + (coach.fallbackReason ? " (" + coach.fallbackReason + ")" : ""),
    level: store.view.level,
    hands: st.hands,
    decisions: st.decisions,
    correct: st.correct,
    acceptable: st.acceptable,
    mistakes: st.mistakes,
    netBB: st.net / 2,
    byCat: Object.fromEntries(Object.entries(st.byCat).map(([k, b]) => [k, `${b.n} spots, ${Math.round((100 * b.score) / b.n)}%`])),
    coachMs: (() => {
      const t = (window.__poker.timings || []).slice().sort((a, b) => a - b);
      return t.length ? { n: t.length, median: t[t.length >> 1], p90: t[Math.floor(t.length * 0.9)], max: t.at(-1) } : null;
    })(),
  };
});
console.log(JSON.stringify({ ...report, seconds: Math.round((Date.now() - t0) / 1000), worstStallMs: worstStall }, null, 1));
if (problems.length) console.log("PROBLEMS:\n  " + problems.join("\n  "));
await browser.close();
server.close();
process.exit(problems.length ? 1 : 0);
