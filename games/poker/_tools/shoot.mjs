// Dev-only: capture the table in fixed poses, at desktop and phone sizes, into _tools/out/.
//   node shoot.mjs [pose ...]          e.g. node shoot.mjs flop showdown
// Serves games/poker/ itself, so nothing else needs to be running.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = path.join(root, "_tools", "out");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json" };
const SIZES = {
  desktop: { width: 1440, height: 900, deviceScaleFactor: 1 },
  phone: { width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
  tablet: { width: 900, height: 700, deviceScaleFactor: 1 }, // a small tablet, or a narrow desktop window
};
const ALL = ["table", "cards", "preflop", "flop", "showdown", "nine", "headsup"];

const server = http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p.endsWith("/")) p += "index.html";
  const file = path.join(root, p);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-store" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}/`;

const args = process.argv.slice(2);
const sizes = args.filter((a) => SIZES[a]);
const poses = args.filter((a) => !SIZES[a]);
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--hide-scrollbars"],
});
fs.mkdirSync(out, { recursive: true });
let failed = false;
for (const pose of poses.length ? poses : ALL)
  for (const size of sizes.length ? sizes : ["desktop", "phone"]) {
    const context = await browser.createBrowserContext(); // a clean profile per capture: no saved mode or scores
    const page = await context.newPage();
    await page.setViewport(SIZES[size]);
    const problems = [];
    page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
    page.on(
      "console",
      (m) => m.type() === "error" && !/game-analytics|favicon|404/.test(m.text() + m.location()?.url) && problems.push("console: " + m.text())
    );
    const query = pose.includes("=") ? pose : `pose=${pose}&seed=1`;
    await page.goto(`${base}?${query}`, { waitUntil: "load" });
    // A query may carry directions for this tool: until=<phase> waits for the game to reach a phase,
    // then=<css selector> clicks something once it has, and analysed=1 also waits for the coach's answer.
    const q = new URLSearchParams(query);
    try {
      await page.waitForFunction(() => document.body.dataset.ready === "1", { timeout: 20000 });
      if (q.has("until")) {
        await page.waitForFunction(
          (phase, analysed, minStreet) =>
            document.body.dataset.phase === phase &&
            (!analysed || !!window.__poker?.store.view.analysis) &&
            (window.__poker?.store.view.street ?? 0) >= minStreet,
          { timeout: 90000 },
          q.get("until"),
          q.has("analysed"),
          +(q.get("minStreet") || 0)
        );
        if (q.has("then")) for (const sel of q.getAll("then")) await page.click(sel), await new Promise((r) => setTimeout(r, 350));
        if (q.has("after")) await page.waitForFunction((phase) => document.body.dataset.phase === phase, { timeout: 90000 }, q.get("after"));
        await new Promise((r) => setTimeout(r, 900));
      } else await new Promise((r) => setTimeout(r, pose.includes("=") ? +(process.env.WAIT || 6000) : 900));
    } catch (e) {
      problems.push("did not get there: " + e.message.split("\n")[0]);
    }
    const file = path.join(out, `${(q.get("name") || pose).replace(/[^a-z0-9]+/gi, "-")}-${size}.png`);
    await page.screenshot({ path: file });
    console.log((problems.length ? "PROBLEM " : "ok      ") + path.relative(root, file) + (problems.length ? "\n   " + problems.join("\n   ") : ""));
    failed ||= problems.length > 0;
    await context.close();
  }
await browser.close();
server.close();
process.exit(failed ? 1 : 0);
