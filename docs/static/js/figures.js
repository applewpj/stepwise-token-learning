// Live figures of the project page. They read docs/static/data/*.json, written by
// tools/build_site_data.py from `tokenlearning reproduce` runs.
(function () {
  "use strict";

  const P = window.TLSPlot;
  const C = P.C;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = { sig: "#e0873a", power: "#7fb0e6", specPower: "#8fcf9b", guide: "#4b4f56" };
  const NO_HOVER = { a: null, group: null, src: null };

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  /* Linear interpolation; NaN outside the data or next to a gap (null). */
  function valueAt(xs, ys, v) {
    const n = xs.length;
    if (!(v >= xs[0] && v <= xs[n - 1])) {
      return NaN;
    }
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= v) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    if (ys[lo] === null || ys[hi] === null) {
      return NaN;
    }
    const f = (v - xs[lo]) / (xs[hi] - xs[lo] || 1);
    return ys[lo] + f * (ys[hi] - ys[lo]);
  }

  function binOf(edges, v) {
    if (!(v >= edges[0] && v <= edges[edges.length - 1])) {
      return null;
    }
    let b = 0;
    while (b < edges.length - 2 && v >= edges[b + 1]) {
      b++;
    }
    return b;
  }

  function quantile(values, q) {
    const v = values.filter((x) => x !== null && Number.isFinite(x)).sort((a, b) => a - b);
    if (!v.length) {
      return NaN;
    }
    return v[Math.min(v.length - 1, Math.max(0, Math.round(q * (v.length - 1))))];
  }

  function extent(arrays, positive) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const arr of arrays) {
      for (const v of arr) {
        if (v !== null && Number.isFinite(v) && (!positive || v > 0)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
    }
    return [lo, hi];
  }

  function padded(range, log, frac) {
    const f = frac === undefined ? 0.06 : frac;
    if (log) {
      const k = Math.pow(range[1] / range[0], f) || 1.2;
      return [range[0] / k, range[1] * k];
    }
    const p = (range[1] - range[0]) * f || Math.abs(range[1]) * 0.1 || 1;
    return [range[0] - p, range[1] + p];
  }

  function linspace(lo, hi, n) {
    return Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1));
  }

  function logspace(lo, hi, n) {
    return linspace(Math.log(lo), Math.log(hi), n).map((v) => Math.exp(v));
  }

  function sigmoidAt(v, tau, lambda, start, end) {
    return end + (start - end) / (1 + Math.exp(lambda * (v - tau)));
  }

  function setPressed(buttons, attr, value) {
    for (const b of buttons) {
      b.setAttribute("aria-pressed", String(b.dataset[attr] === value));
    }
  }

  function trim(v, digits) {
    return String(Number(Number(v).toPrecision(digits || 3))).replace("-", "\u2212");
  }

  function decode(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      out[i] = bin.charCodeAt(i);
    }
    return out;
  }

  /* Rows stored as uint8 between their own minimum and maximum. */
  function rowReader(q, m) {
    const bytes = decode(q.u8);
    return (i) => {
      const out = new Array(m);
      const lo = q.lo[i];
      const span = q.hi[i] - lo;
      for (let k = 0; k < m; k++) {
        out[k] = lo + (bytes[i * m + k] / 255) * span;
      }
      return out;
    };
  }

  const AXES = {
    T: {
      file: "static/data/t.json",
      name: "Training steps",
      symbol: "T",
      unit: "steps",
      value: (v) => `${P.fmtNum(Math.round(v))} steps`,
      tick: (v) => P.fmtTick(v),
      range: (w) => `${P.fmtTick(w[0])}\u2013${P.fmtTick(w[1])} steps`,
      setup: "a 1.5B dense model, 54 checkpoints over 300B tokens",
    },
    D: {
      file: "static/data/d.json",
      name: "Data scale",
      symbol: "D",
      unit: "B tokens",
      value: (v) => `${trim(v)}B tokens`,
      tick: (v) => trim(v),
      range: (w) => `${trim(w[0])}B\u2013${trim(w[1])}B tokens`,
      setup: "the loss frontier of 50 runs of a 290M MoE model",
    },
    M: {
      file: "static/data/m.json",
      name: "Model scale",
      symbol: "M",
      unit: "GFLOPs/token",
      value: (v) => `${trim(v)} GFLOPs/token`,
      tick: (v) => trim(v),
      range: (w) => `${trim(w[0])}\u2013${trim(w[1])} GFLOPs/token`,
      setup: "the frontier of 60 models, each trained on 10B tokens",
    },
  };
  const AXIS_KEYS = ["T", "D", "M"];

  /* The figures of Part 1 show the same axis; switching it in one switches it in all. */
  const axisBus = {
    key: "T",
    listeners: [],
    set(key, from) {
      this.key = key;
      for (const fn of this.listeners) {
        if (fn !== from) {
          fn(key);
        }
      }
    },
  };

  const pending = {};

  function loadJson(file, prepare) {
    if (!pending[file]) {
      pending[file] = fetch(file)
        .then((r) => {
          if (!r.ok) {
            throw new Error(`HTTP ${r.status}`);
          }
          return r.json();
        })
        .then(prepare);
    }
    return pending[file];
  }

  function prepareAxis(d) {
    d.meta = AXES[d.key];
    d.lo = d.x[0];
    d.hi = d.x[d.x.length - 1];
    d.color = (v) => P.timeColor(clamp((v - d.lo) / (d.hi - d.lo), 0, 1));
    d.tokens.lossOf = rowReader(d.tokens.loss, d.x.length);
    const pu = d.pulses;
    const bytes = decode(pu.u8);
    const m = pu.x.length;
    pu.rows = pu.peak.map((peak, b) => Array.from({ length: m }, (_, k) => (bytes[b * m + k] / 255) * peak));
    pu.centers = pu.peak.map((_, b) => 0.5 * (pu.edges[b] + pu.edges[b + 1]));
    const g = d.groups;
    g.alignedX = g.tau.map((t) => g.x.map((v) => v - t));
    g.alignedRaw = g.raw.map((row, k) => row.map((v) => (v === null ? null : v - g.mid[k])));
    return d;
  }

  function loadAxis(key) {
    return loadJson(AXES[key].file, prepareAxis);
  }

  function loadFailed(container, error) {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent =
      location.protocol === "file:"
        ? "The live figures read JSON files, which browsers block on file:// pages. Serve the folder instead: python -m http.server -d docs"
        : `Could not load the figure data (${error.message}).`;
    container.appendChild(note);
  }

  function provenance(d) {
    const s = d.stats;
    return d.setting === "full"
      ? `All ${P.fmtNum(s.n_total)} tokens fitted at the paper setting.`
      : `A random ${P.fmtNum(s.n_fitted)} of ${P.fmtNum(s.n_total)} tokens fitted (quick setting).`;
  }

  function axisLabel(meta) {
    const name = `${meta.name.toLowerCase()} ${meta.symbol}`;
    return name.includes(meta.unit) ? name : `${name} (${meta.unit})`;
  }

  /* One loss trajectory as a small multiple: measured points, fitted sigmoid, learning time and
     its 10-90% transition. */
  function drawTile(svg, o) {
    const W = o.width;
    const H = o.height;
    P.clear(svg);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("font-family", P.FONT);
    const top = o.head ? 22 : 4;
    const box = { x: 5, y: top, w: W - 10, h: H - top - 15 };
    const sx = P.scale(false).set(o.xDomain, [box.x, box.x + box.w]);
    const sy = P.scale(false).set(o.yDomain, [box.y + box.h, box.y]);
    P.el("rect", { x: box.x, y: box.y, width: box.w, height: box.h, fill: "#f8f9fb", rx: 3 }, svg);
    if (o.tau >= o.xDomain[0] && o.tau <= o.xDomain[1]) {
      const half = Math.log(9) / o.lambda;
      const a = clamp(sx(o.tau - half), box.x, box.x + box.w);
      const b = clamp(sx(o.tau + half), box.x, box.x + box.w);
      P.el("rect", { x: a, y: box.y, width: Math.max(1.5, b - a), height: box.h, fill: o.color, "fill-opacity": 0.14 }, svg);
      P.el("line", { x1: sx(o.tau), x2: sx(o.tau), y1: box.y, y2: box.y + box.h, stroke: o.color, "stroke-width": 1.2, "stroke-dasharray": "3 2.5" }, svg);
    }
    const fitX = linspace(o.xDomain[0], o.xDomain[1], 121);
    const fit = fitX.map(o.fit);
    P.el("path", { d: P.linePath(o.xs, o.loss, sx, sy), fill: "none", stroke: "#c5cad0", "stroke-width": 0.9 }, svg);
    P.el("path", { d: P.dotsPath(o.xs, o.loss, sx, sy), fill: "none", stroke: "#7d858e", "stroke-width": 2.6, "stroke-linecap": "round" }, svg);
    P.el("path", { d: P.linePath(fitX, fit, sx, sy), fill: "none", stroke: o.color, "stroke-width": 2.2 }, svg);
    P.text(svg, box.x, H - 3, o.fmtX(o.xDomain[0]), { "font-size": 9.5, fill: C.faint });
    P.text(svg, box.x + box.w, H - 3, o.fmtX(o.xDomain[1]), { "font-size": 9.5, fill: C.faint, "text-anchor": "end" });
    if (o.head) {
      P.text(svg, 5, 14, o.head, { fill: C.ink, "font-size": 11.5, "font-weight": 650 });
      P.text(svg, W - 5, 14, o.sub, { fill: C.muted, "font-size": 10.5, "text-anchor": "end" });
    }
    if (o.title) {
      P.el("title", {}, svg).textContent = o.title;
    }
  }

  function tauLabel(d, tau) {
    if (tau < d.lo) {
      return `\u03c4 < ${d.meta.tick(d.lo)}`;
    }
    if (tau > d.hi) {
      return `\u03c4 > ${d.meta.tick(d.hi)}`;
    }
    return `\u03c4 ${d.meta.tick(tau)}`;
  }

  /* ------------------------------------------------------------------ */
  /* 1.1  Real token trajectories                                        */
  /* ------------------------------------------------------------------ */

  function initRealTokens(root, opts) {
    const o = opts || {};
    const params = new URLSearchParams(o.initial || "");
    const state = {
      axis: AXIS_KEYS.includes(params.get("axis")) ? params.get("axis") : axisBus.key,
      mode: params.get("mode") === "random" ? "random" : "clear",
      draw: clamp(Math.floor(Number(params.get("draw")) || 0), 0, 9999),
    };
    const grid = root.querySelector("[data-tiles]");
    const caption = root.querySelector("[data-caption]");
    const axisButtons = Array.from(root.querySelectorAll("[data-axis]"));
    const modeButtons = Array.from(root.querySelectorAll("[data-mode]"));
    const tiles = [];
    for (let k = 0; k < 10; k++) {
      const tile = document.createElement("div");
      tile.className = "tile";
      const svg = document.createElementNS(SVG_NS, "svg");
      svg.setAttribute("role", "img");
      tile.appendChild(svg);
      grid.appendChild(tile);
      tiles.push(svg);
    }

    function chosen(d) {
      const t = d.tokens;
      if (state.mode === "clear") {
        const out = [];
        for (let k = 0; k < t.edges.length - 1; k++) {
          const pool = [];
          for (let i = 0; i < t.n; i++) {
            if (t.pool[i] === 1 && t.interval[i] === k) {
              pool.push(i);
            }
          }
          if (pool.length) {
            out.push(pool[state.draw % pool.length]);
          }
        }
        return out;
      }
      const pool = [];
      for (let i = 0; i < t.n; i++) {
        if (t.pool[i] === 0) {
          pool.push(i);
        }
      }
      const first = (state.draw * tiles.length) % pool.length;
      return Array.from({ length: Math.min(tiles.length, pool.length) }, (_, k) => pool[(first + k) % pool.length]).sort(
        (a, b) => t.tau[a] - t.tau[b],
      );
    }

    function render(d) {
      const t = d.tokens;
      const picks = chosen(d);
      tiles.forEach((svg, k) => {
        const i = picks[k];
        if (i === undefined) {
          P.clear(svg);
          return;
        }
        const loss = t.lossOf(i);
        let lo = Infinity;
        let hi = -Infinity;
        for (let j = 0; j < loss.length; j++) {
          const f = sigmoidAt(d.x[j], t.tau[i], t.lambda[i], t.start[i], t.end[i]);
          lo = Math.min(lo, loss[j], f);
          hi = Math.max(hi, loss[j], f);
        }
        drawTile(svg, {
          width: 220,
          height: 128,
          xs: d.x,
          loss,
          fit: (v) => sigmoidAt(v, t.tau[i], t.lambda[i], t.start[i], t.end[i]),
          tau: t.tau[i],
          lambda: t.lambda[i],
          color: d.color(t.tau[i]),
          xDomain: [d.lo, d.hi],
          yDomain: padded([lo, hi], false, 0.08),
          fmtX: d.meta.tick,
          head: tauLabel(d, t.tau[i]),
          sub: `\u0394 ${trim(t.start[i] - t.end[i], 2)} \u00b7 R\u00b2 ${t.r2[i].toFixed(2)}`,
          title: `Token learned at ${d.meta.value(t.tau[i])}: loss ${trim(t.start[i])} to ${trim(t.end[i])}, R\u00b2 ${t.r2[i].toFixed(2)}`,
        });
      });
      if (caption) {
        caption.textContent =
          state.mode === "clear"
            ? `${d.meta.name} ${d.meta.symbol}: one sharp, well-fitted drop (R\u00b2 \u2265 0.5, \u0394 \u2265 0.3, sharpest tenth) from each tenth of ${d.meta.range(t.edges.length ? [t.edges[0], t.edges[t.edges.length - 1]] : d.window)}, early to late. ${provenance(d)}`
            : `${d.meta.name} ${d.meta.symbol}: tokens drawn at random from the ${P.fmtNum(d.stats.n_drops)} whose fitted loss goes down, sorted by \u03c4. ${provenance(d)}`;
      }
    }

    function update(user) {
      setPressed(axisButtons, "axis", state.axis);
      setPressed(modeButtons, "mode", state.mode);
      loadAxis(state.axis)
        .then(render)
        .catch((error) => loadFailed(grid, error));
      if (user && o.onChange) {
        o.onChange(query());
      }
    }

    function query() {
      const q = new URLSearchParams({ axis: state.axis, mode: state.mode });
      if (state.draw) {
        q.set("draw", String(state.draw));
      }
      return q.toString();
    }

    function follow(key) {
      state.axis = key;
      update(false);
    }

    for (const b of axisButtons) {
      b.addEventListener("click", () => {
        state.axis = b.dataset.axis;
        axisBus.set(state.axis, follow);
        update(true);
      });
    }
    for (const b of modeButtons) {
      b.addEventListener("click", () => {
        state.mode = b.dataset.mode;
        update(true);
      });
    }
    root.querySelector("[data-redraw]").addEventListener("click", () => {
      state.draw += 1;
      update(true);
    });
    axisBus.listeners.push(follow);
    if (params.get("axis")) {
      axisBus.set(state.axis, follow);
    }
    update(false);

    return {
      query,
      svgs: () => tiles.filter((s) => s.firstChild),
      title: () => `Token loss trajectories \u00b7 ${AXES[state.axis].name} (stepwise-token-learning)`,
    };
  }

  /* ------------------------------------------------------------------ */
  /* 1.2  From tokens to the loss curve                                  */
  /* ------------------------------------------------------------------ */

  const PANEL_KEYS = ["mean", "groups", "aligned", "pulses", "spectrum", "rate"];
  const SCALED = ["mean", "spectrum", "rate"];
  const TITLES = {
    mean: "A \u00b7 Mean loss",
    groups: "B \u00b7 Tokens grouped by learning time",
    aligned: "C \u00b7 Groups shifted to their own \u03c4",
    pulses: "D \u00b7 Learning pulses in 40 bins of \u03c4",
    spectrum: "E \u00b7 Learning-time spectrum",
    rate: "F \u00b7 Loss derivative",
  };

  function initRealPanels(root, opts) {
    const o = opts || {};
    const params = new URLSearchParams(o.initial || "");
    const state = {
      axis: AXIS_KEYS.includes(params.get("axis")) ? params.get("axis") : axisBus.key,
      scale: { mean: "lin", spectrum: "log", rate: "log" },
    };
    for (const key of SCALED) {
      const v = params.get(key);
      if (v === "lin" || v === "log") {
        state.scale[key] = v;
      }
    }
    const data = {};
    let hover = NO_HOVER;
    const axisButtons = Array.from(root.querySelectorAll("[data-axis]"));
    const scaleButtons = Array.from(root.querySelectorAll("[data-scale]"));
    const caption = root.querySelector("[data-caption]");
    const panels = {};

    function frameOptions(d, box, xLabel, yLabel) {
      return {
        xLabel,
        yLabel,
        yLabelOffset: 44,
        xFormat: d.meta.tick,
        xCount: Math.max(3, Math.floor(box.w / 62)),
        yCount: 5,
      };
    }

    function drawSeries(frame, series, sx, sy) {
      for (const s of series) {
        P.el(
          "path",
          {
            d: P.linePath(s.xs, s.ys, sx, sy),
            fill: "none",
            stroke: s.color,
            "stroke-width": s.width || 1.5,
            "stroke-dasharray": s.dash || null,
            "stroke-opacity": s.dots ? 0.8 : 1,
          },
          frame.data,
        );
        if (s.dots) {
          P.el("path", { d: P.dotsPath(s.xs, s.ys, sx, sy), fill: "none", stroke: s.color, "stroke-width": 3.4, "stroke-linecap": "round" }, frame.data);
        }
      }
    }

    const DRAW = {
      mean(panel, d, box) {
        const log = state.scale.mean === "log";
        const m = d.mean;
        const pw = m.power;
        const xs = m.x;
        const dense = (log ? logspace : linspace)(xs[0], xs[xs.length - 1], 160);
        const shift = (arr) => (log ? arr.map((v) => (v === null ? null : v - pw.E)) : arr);
        const series = [
          { name: "power law", color: COLORS.power, xs: dense, ys: shift(dense.map((a) => pw.E + pw.k * Math.pow(a, -pw.alpha))), dash: "6 3", width: 1.8 },
          { name: "sigmoid-implied", color: COLORS.sig, xs, ys: shift(m.sigmoid), dash: "4 3", width: 1.6 },
          { name: "measured", color: C.emp, xs, ys: shift(m.empirical), dots: true, width: 1.2 },
        ];
        const sx = P.scale(log).set([xs[0], xs[xs.length - 1]]);
        const sy = P.scale(log).set(padded(extent(series.map((s) => s.ys), log), log, 0.06));
        const yLabel = log ? `L \u2212 E  (E = ${pw.E.toFixed(2)})` : "mean loss L";
        const frame = P.frame(panel.stage.content, box, sx, sy, frameOptions(d, box, axisLabel(d.meta), yLabel));
        drawSeries(frame, series, sx, sy);
        return { frame, sx, sy, series, format: (v) => trim(v, 4) };
      },

      groups(panel, d, box) {
        const g = d.groups;
        const sx = P.scale(false).set(d.window);
        const sy = P.scale(false).set(padded(extent(g.raw.concat(g.fit)), false, 0.05));
        const frame = P.frame(panel.stage.content, box, sx, sy, frameOptions(d, box, axisLabel(d.meta), "mean loss of the group"));
        g.tau.forEach((t, k) => {
          const color = d.color(t);
          P.el("path", { d: P.linePath(g.fit_x, g.fit[k], sx, sy), fill: "none", stroke: color, "stroke-width": 1.2, "stroke-dasharray": "4 3" }, frame.data);
          P.el("path", { d: P.linePath(g.x, g.raw[k], sx, sy), fill: "none", stroke: color, "stroke-width": 1.6 }, frame.data);
        });
        return { frame, sx, sy };
      },

      aligned(panel, d, box) {
        const g = d.groups;
        const span = d.window[1] - d.window[0];
        const sx = P.scale(false).set([-span, span]);
        const sy = P.scale(false).set(padded(extent(g.alignedRaw.concat(g.aligned_fit)), false, 0.05));
        const frame = P.frame(panel.stage.content, box, sx, sy, frameOptions(d, box, `${d.meta.unit} from the group's \u03c4`, "loss \u2212 midpoint of the step"));
        P.el("line", { x1: sx(0), x2: sx(0), y1: box.y, y2: box.y + box.h, stroke: C.faint, "stroke-dasharray": "2 3" }, frame.over);
        P.el("line", { x1: box.x, x2: box.x + box.w, y1: sy(0), y2: sy(0), stroke: C.faint, "stroke-dasharray": "2 3" }, frame.over);
        g.tau.forEach((t, k) => {
          const color = d.color(t);
          P.el("path", { d: P.linePath(g.rel, g.aligned_fit[k], sx, sy), fill: "none", stroke: color, "stroke-width": 1.2, "stroke-dasharray": "4 3" }, frame.data);
          P.el("path", { d: P.linePath(g.alignedX[k], g.alignedRaw[k], sx, sy), fill: "none", stroke: color, "stroke-width": 1.6 }, frame.data);
        });
        return { frame, sx, sy };
      },

      pulses(panel, d, box) {
        const pu = d.pulses;
        const sx = P.scale(false).set([pu.x[0], pu.x[pu.x.length - 1]]);
        const sy = P.scale(false).set([0, Math.max(...pu.peak) * 1.06]);
        const frame = P.frame(panel.stage.content, box, sx, sy, frameOptions(d, box, axisLabel(d.meta), "\u2212d\u2113/da, mean of the bin"));
        pu.rows.forEach((row, b) => {
          P.el("path", { d: P.linePath(pu.x, row, sx, sy), fill: "none", stroke: d.color(pu.centers[b]), "stroke-width": 1.1 }, frame.data);
        });
        return { frame, sx, sy };
      },

      spectrum(panel, d, box) {
        const log = state.scale.spectrum === "log";
        const s = d.spectrum;
        const pw = s.power;
        const lo = s.edges[0];
        const hi = s.edges[s.edges.length - 1];
        const dense = (log ? logspace : linspace)(s.x[0], s.x[s.x.length - 1], 120);
        const series = [
          { name: "power law", color: COLORS.specPower, xs: dense, ys: dense.map((a) => pw.c * Math.pow(a, -pw.beta)), dash: "6 3", width: 1.8 },
          { name: "p(\u03c4)", color: C.spec, xs: s.x, ys: s.mass, dots: true, width: 1.3 },
        ];
        const sx = P.scale(log).set([lo, hi]);
        const sy = P.scale(log).set(log ? padded(extent([s.mass], true), true, 0.08) : [0, Math.max(...s.mass) * 1.08]);
        const frame = P.frame(panel.stage.content, box, sx, sy, frameOptions(d, box, `learning time \u03c4 (${d.meta.unit})`, "share of tokens"));
        drawSeries(frame, series, sx, sy);
        return { frame, sx, sy, series, format: (v) => trim(v, 3) };
      },

      rate(panel, d, box) {
        const log = state.scale.rate === "log";
        const r = d.rate;
        const pw = d.mean.power;
        const dense = (log ? logspace : linspace)(r.x[0], r.x[r.x.length - 1], 160);
        const series = [
          { name: "power law", color: COLORS.power, xs: dense, ys: dense.map((a) => pw.alpha * pw.k * Math.pow(a, -pw.alpha - 1)), dash: "6 3", width: 1.8 },
          { name: "sigmoid-implied", color: COLORS.sig, xs: r.x, ys: r.sigmoid, dash: "4 3", width: 1.5 },
          { name: "spectrum-dominated", color: C.recon, xs: r.spectrum_x, ys: r.spectrum, dash: "7 3", width: 1.9 },
          { name: "measured", color: C.emp, xs: r.x, ys: r.empirical, dots: true, width: 1.1 },
        ];
        const smooth = extent(series.slice(0, 3).map((s) => s.ys), log);
        const yRange = log
          ? [Math.min(smooth[0], quantile(r.empirical.filter((v) => v > 0), 0.04)), Math.max(smooth[1], quantile(r.empirical, 0.97))]
          : [Math.min(0, quantile(r.empirical, 0.03)), Math.max(smooth[1], quantile(r.empirical, 0.97))];
        const sx = P.scale(log).set([r.x[0], r.x[r.x.length - 1]]);
        const sy = P.scale(log).set(padded(yRange, log, 0.07));
        const frame = P.frame(panel.stage.content, box, sx, sy, frameOptions(d, box, axisLabel(d.meta), "\u2212dL/da"));
        if (!log) {
          P.el("line", { x1: box.x, x2: box.x + box.w, y1: sy(0), y2: sy(0), stroke: C.faint, "stroke-dasharray": "2 3" }, frame.over);
        }
        drawSeries(frame, series, sx, sy);
        return { frame, sx, sy, series, format: (v) => P.fmtTick(Number(v.toPrecision(2))) };
      },
    };

    function drawPanel(panel) {
      if (!panel.width) {
        return;
      }
      const W = panel.width;
      const H = Math.round(clamp(W * 0.76, 236, 300));
      panel.stage.resize(W, H);
      P.place(panel.zone, { x: 0, y: 0, w: W, h: H });
      P.text(panel.stage.content, 2, 15, TITLES[panel.key], { fill: C.ink, "font-size": 12.5, "font-weight": 650 });
      panel.view = null;
      const d = data[state.axis];
      if (!d) {
        P.text(panel.stage.content, W / 2, H / 2, "loading\u2026", { "text-anchor": "middle", fill: C.faint, "font-size": 12 });
        return;
      }
      const box = { x: 56, y: 32, w: W - 56 - 10, h: H - 32 - 42 };
      panel.view = Object.assign({ box }, DRAW[panel.key](panel, d, box));
      drawHover(panel);
    }

    function guide(layer, view, a) {
      const px = view.sx(a);
      if (Number.isFinite(px) && px >= view.box.x && px <= view.box.x + view.box.w) {
        P.el("line", { x1: px, x2: px, y1: view.box.y, y2: view.box.y + view.box.h, stroke: COLORS.guide, "stroke-opacity": 0.55, "stroke-dasharray": "3 3" }, layer);
      }
    }

    function label(layer, view, lines) {
      const x = view.box.x + view.box.w - 6;
      let y = view.box.y + 14;
      for (const line of lines) {
        P.text(layer, x, y, line.text, {
          fill: line.color || C.ink,
          "font-size": 11,
          "font-weight": line.bold ? 650 : 500,
          "text-anchor": "end",
          stroke: "#fff",
          "stroke-width": 4,
          "paint-order": "stroke",
        });
        y += 14;
      }
    }

    function drawHover(panel) {
      const layer = panel.stage.hover;
      P.clear(layer);
      const view = panel.view;
      const d = data[state.axis];
      if (!view || !d) {
        return;
      }
      const dim = (on) => view.frame.data.setAttribute("opacity", on ? 0.22 : 1);
      const g = d.groups;
      const key = panel.key;
      if (key === "groups" || key === "aligned") {
        const k = hover.group;
        dim(k !== null);
        if (k !== null) {
          const color = d.color(g.tau[k]);
          const aligned = key === "aligned";
          P.el("path", { d: P.linePath(aligned ? g.rel : g.fit_x, aligned ? g.aligned_fit[k] : g.fit[k], view.sx, view.sy), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": "4 3" }, layer);
          P.el("path", { d: P.linePath(aligned ? g.alignedX[k] : g.x, aligned ? g.alignedRaw[k] : g.raw[k], view.sx, view.sy), fill: "none", stroke: color, "stroke-width": 2.8 }, layer);
          label(layer, view, [{ text: `\u03c4 \u2248 ${d.meta.value(g.tau[k])}`, bold: true }, { text: `${P.fmtNum(g.count[k])} tokens`, color: C.muted }]);
        } else if (key === "groups" && hover.a !== null) {
          guide(layer, view, hover.a);
        }
        return;
      }
      if (hover.a === null) {
        dim(false);
        return;
      }
      guide(layer, view, hover.a);
      if (key === "pulses") {
        const b = binOf(d.pulses.edges, hover.a);
        dim(b !== null);
        if (b !== null) {
          const pu = d.pulses;
          P.el("path", { d: P.linePath(pu.x, pu.rows[b], view.sx, view.sy), fill: "none", stroke: d.color(pu.centers[b]), "stroke-width": 2.6 }, layer);
          label(layer, view, [{ text: `tokens with \u03c4 in ${d.meta.tick(pu.edges[b])}\u2013${d.meta.tick(pu.edges[b + 1])}`, bold: true }]);
        }
        return;
      }
      if (key === "spectrum") {
        const s = d.spectrum;
        const b = binOf(s.edges, hover.a);
        if (b !== null && s.mass[b] > 0) {
          P.el("circle", { cx: view.sx(s.x[b]), cy: view.sy(s.mass[b]), r: 4.6, fill: C.spec, stroke: "#fff", "stroke-width": 1.5 }, layer);
          label(layer, view, [
            { text: `\u03c4 in ${d.meta.tick(s.edges[b])}\u2013${d.meta.tick(s.edges[b + 1])}`, bold: true },
            { text: `${trim(100 * s.mass[b], 3)}% of tokens`, color: C.spec },
          ]);
        }
        return;
      }
      if (hover.src !== key) {
        return;
      }
      const lines = [{ text: d.meta.value(hover.a), bold: true }];
      for (const s of view.series.slice().reverse()) {
        const v = valueAt(s.xs, s.ys, hover.a);
        if (!Number.isFinite(v) || (view.sy.log && v <= 0)) {
          continue;
        }
        P.el("circle", { cx: view.sx(hover.a), cy: view.sy(v), r: 3, fill: s.color, stroke: "#fff", "stroke-width": 1.2 }, layer);
        lines.push({ text: `${s.name} ${view.format(v)}`, color: s.color });
      }
      label(layer, view, lines);
    }

    function setHover(h) {
      if (h.a === hover.a && h.group === hover.group && h.src === hover.src) {
        return;
      }
      hover = h;
      for (const key of PANEL_KEYS) {
        drawHover(panels[key]);
      }
    }

    function nearestGroup(panel, d, xv, py) {
      const g = d.groups;
      const aligned = panel.key === "aligned";
      let best = null;
      let bestDist = 14;
      g.tau.forEach((_, k) => {
        const y = aligned ? valueAt(g.alignedX[k], g.alignedRaw[k], xv) : valueAt(g.x, g.raw[k], xv);
        if (!Number.isFinite(y)) {
          return;
        }
        const dist = Math.abs(panel.view.sy(y) - py);
        if (dist < bestDist) {
          bestDist = dist;
          best = k;
        }
      });
      return best;
    }

    function onPointer(panel, event) {
      const d = data[state.axis];
      const view = panel.view;
      if (!d || !view) {
        return;
      }
      const p = P.pointer(panel.svg, event);
      const b = view.box;
      if (p.x < b.x - 8 || p.x > b.x + b.w + 8 || p.y < b.y - 8 || p.y > b.y + b.h + 8) {
        setHover(NO_HOVER);
        return;
      }
      const xv = view.sx.invert(clamp(p.x, b.x, b.x + b.w));
      if (panel.key === "groups" || panel.key === "aligned") {
        const k = nearestGroup(panel, d, xv, p.y);
        if (k !== null) {
          setHover({ a: d.groups.tau[k], group: k, src: panel.key });
        } else {
          setHover({ a: panel.key === "groups" ? xv : null, group: null, src: panel.key });
        }
        return;
      }
      setHover({ a: xv, group: null, src: panel.key });
    }

    for (const key of PANEL_KEYS) {
      const svg = root.querySelector(`[data-panel="${key}"]`);
      const stage = P.stage(svg);
      const panel = { key, svg, stage, zone: stage.zone(), width: 0, view: null };
      panels[key] = panel;
      panel.zone.addEventListener("pointermove", (e) => onPointer(panel, e));
      panel.zone.addEventListener("pointerdown", (e) => onPointer(panel, e));
      panel.zone.addEventListener("pointerleave", () => setHover(NO_HOVER));
      P.observeWidth(svg.parentNode, (w) => {
        panel.width = w;
        drawPanel(panel);
      });
    }

    function drawAll() {
      for (const key of PANEL_KEYS) {
        drawPanel(panels[key]);
      }
    }

    function updateCaption() {
      const d = data[state.axis];
      if (!d || !caption) {
        return;
      }
      const n = d.groups.count.reduce((a, b) => a + b, 0);
      caption.textContent =
        `${d.meta.name} ${d.meta.symbol}: ${d.meta.setup}. B and C average ${P.fmtNum(n)} tokens whose fitted loss goes down, ` +
        `in ${d.groups.tau.length} narrow bins of \u03c4 over ${d.meta.range(d.window)}. ${provenance(d)}`;
    }

    function query() {
      const q = new URLSearchParams({ axis: state.axis });
      for (const key of SCALED) {
        q.set(key, state.scale[key]);
      }
      return q.toString();
    }

    function select(key, user) {
      state.axis = key;
      hover = NO_HOVER;
      setPressed(axisButtons, "axis", key);
      drawAll();
      loadAxis(key)
        .then((d) => {
          data[key] = d;
          if (state.axis === key) {
            drawAll();
            updateCaption();
          }
        })
        .catch((error) => loadFailed(root.querySelector(".panels"), error));
      if (user && o.onChange) {
        o.onChange(query());
      }
    }

    function syncScaleButtons() {
      for (const b of scaleButtons) {
        const [key, value] = b.dataset.scale.split(":");
        b.setAttribute("aria-pressed", String(state.scale[key] === value));
      }
    }

    function follow(key) {
      select(key, false);
    }

    for (const b of axisButtons) {
      b.addEventListener("click", () => {
        axisBus.set(b.dataset.axis, follow);
        select(b.dataset.axis, true);
      });
    }
    for (const b of scaleButtons) {
      b.addEventListener("click", () => {
        const [key, value] = b.dataset.scale.split(":");
        state.scale[key] = value;
        syncScaleButtons();
        hover = NO_HOVER;
        drawPanel(panels[key]);
        if (o.onChange) {
          o.onChange(query());
        }
      });
    }
    syncScaleButtons();
    axisBus.listeners.push(follow);
    if (params.get("axis")) {
      axisBus.set(state.axis, follow);
    }
    select(state.axis, false);
    for (const key of AXIS_KEYS) {
      if (key !== state.axis) {
        loadAxis(key).then((d) => {
          data[key] = d;
        }, () => {});
      }
    }

    return {
      query,
      svgs: () => PANEL_KEYS.map((key) => panels[key].svg),
      title: () => `From tokens to the loss curve \u00b7 ${AXES[state.axis].name} (stepwise-token-learning)`,
    };
  }

  /* ------------------------------------------------------------------ */
  /* 2.1  Synthetic problems and their trajectories                      */
  /* ------------------------------------------------------------------ */

  function prepareMano(d) {
    d.lo = 0;
    d.hi = d.x[d.x.length - 1];
    d.tokens.lossOf = rowReader(d.tokens.loss, d.x.length);
    d.columns = d.edges.length + 1;
    d.color = (v) => P.timeColor(clamp(v / d.hi, 0, 1));
    return d;
  }

  function prettyExpression(q) {
    return q.replace(/\*/g, "\u00d7").replace(/ - /g, " \u2212 ");
  }

  function initMano(root, opts) {
    const o = opts || {};
    const params = new URLSearchParams(o.initial || "");
    const state = {
      mode: params.get("mode") === "random" ? "random" : "time",
      column: params.has("col") ? clamp(Math.floor(Number(params.get("col"))) || 0, 0, 9) : 3,
      draw: clamp(Math.floor(Number(params.get("draw")) || 0), 0, 9999),
    };
    const svg = root.querySelector("[data-mano-grid]");
    const stage = P.stage(svg);
    const zone = stage.zone();
    const tilesBox = root.querySelector("[data-tiles]");
    const caption = root.querySelector("[data-caption]");
    const modeButtons = Array.from(root.querySelectorAll("[data-mode]"));
    let d = null;
    let width = 0;
    let hoverColumn = null;
    let layout = null;
    let shown = [];
    const kSteps = (v) => (Math.abs(v) >= 1000 ? `${trim(v / 1000)}k` : trim(v));

    const tiles = [];
    for (let k = 0; k < 6; k++) {
      const card = document.createElement("div");
      card.className = "mtile";
      const head = document.createElement("p");
      head.className = "mq";
      const plot = document.createElementNS(SVG_NS, "svg");
      plot.setAttribute("role", "img");
      const foot = document.createElement("p");
      foot.className = "mmeta";
      card.append(head, plot, foot);
      tilesBox.appendChild(card);
      tiles.push({ card, head, plot, foot });
    }

    function columnText(c) {
      const e = d.edges;
      if (c === 0) {
        return `learned before the first checkpoint (${kSteps(e[0])} steps)`;
      }
      if (c === d.columns - 1) {
        return `not learned by ${kSteps(e[e.length - 1])} steps`;
      }
      return `learned at ${kSteps(e[c - 1])}\u2013${kSteps(e[c])} steps`;
    }

    function drawGrid() {
      if (!width || !d) {
        return;
      }
      const W = width;
      const cols = d.columns;
      const rows = d.levels.length;
      const left = 34;
      const top = 40;
      const cw = (W - left - 4) / cols;
      const ch = clamp(cw * 0.66, 15, 24);
      const H = Math.round(top + rows * ch + 34);
      stage.resize(W, H);
      P.place(zone, { x: left, y: top, w: cols * cw, h: rows * ch });
      layout = { left, top, cw, ch, cols, rows };
      const ink = stage.content;
      P.text(ink, 0, 13, "Problems by difficulty and learning time", { fill: C.ink, "font-size": 12.5, "font-weight": 650 });
      const c = hoverColumn !== null ? hoverColumn : state.mode === "time" ? state.column : null;
      if (c !== null) {
        const n = d.counts.reduce((sum, row) => sum + row[c], 0);
        P.text(ink, 0, 29, `${P.fmtNum(n)} problems ${columnText(c)}`, { fill: C.muted, "font-size": 11 });
      } else {
        P.text(ink, 0, 29, "click a column to draw problems from it", { fill: C.muted, "font-size": 11 });
      }
      let max = 0;
      for (const row of d.counts) {
        for (const v of row) {
          max = Math.max(max, v);
        }
      }
      d.counts.forEach((row, r) => {
        row.forEach((v, k) => {
          const f = Math.sqrt(v / max);
          P.el("rect", { x: left + k * cw + 0.5, y: top + r * ch + 0.5, width: cw - 1, height: ch - 1, rx: 2, fill: v ? "#4b44ce" : "#f4f5f7", "fill-opacity": v ? 0.07 + 0.93 * f : 1 }, ink);
          if (v && cw >= 30) {
            P.text(ink, left + (k + 0.5) * cw, top + (r + 0.5) * ch + 3.5, String(v), { "text-anchor": "middle", "font-size": 9, fill: f > 0.55 ? "#fff" : C.text });
          }
        });
        P.text(ink, left - 6, top + (r + 0.5) * ch + 3.5, `L${d.levels[r]}`, { "text-anchor": "end", "font-size": 10, fill: C.muted });
      });
      d.edges.forEach((e, j) => {
        const x = left + (j + 1) * cw;
        P.el("line", { x1: x, x2: x, y1: top + rows * ch, y2: top + rows * ch + 4, stroke: C.axis }, ink);
        if (cw >= 26 || j % 2 === 0) {
          P.text(ink, x, top + rows * ch + 15, kSteps(e), { "text-anchor": "middle", "font-size": 9.5, fill: C.muted });
        }
      });
      P.text(ink, left + (cols * cw) / 2, H - 2, "learning time \u03c4 (steps)", { "text-anchor": "middle", "font-size": 11, fill: C.text });
      P.text(ink, -(top + (rows * ch) / 2), 9, "difficulty", { transform: "rotate(-90)", "text-anchor": "middle", "font-size": 11, fill: C.text });
      const over = stage.hover;
      if (state.mode === "time") {
        P.el("rect", { x: left + state.column * cw, y: top - 2, width: cw, height: rows * ch + 4, fill: "none", stroke: "#4b44ce", "stroke-width": 2, rx: 3 }, over);
      } else {
        for (const i of shown) {
          const t = d.tokens;
          P.el("circle", { cx: left + (t.column[i] + 0.5) * cw, cy: top + (t.difficulty[i] - d.levels[0] + 0.5) * ch, r: 3.4, fill: "#fff", stroke: "#4b44ce", "stroke-width": 1.6 }, over);
        }
      }
      if (hoverColumn !== null && hoverColumn !== (state.mode === "time" ? state.column : -1)) {
        P.el("rect", { x: left + hoverColumn * cw, y: top - 2, width: cw, height: rows * ch + 4, fill: "none", stroke: C.faint, "stroke-width": 1.4, "stroke-dasharray": "3 2", rx: 3 }, over);
      }
    }

    function pick() {
      const t = d.tokens;
      const pool = [];
      for (let i = 0; i < t.n; i++) {
        if (state.mode === "random" ? t.pool[i] === 0 : t.pool[i] === 1 && t.column[i] === state.column) {
          pool.push(i);
        }
      }
      if (!pool.length) {
        return [];
      }
      const first = (state.draw * tiles.length) % pool.length;
      return Array.from({ length: Math.min(tiles.length, pool.length) }, (_, k) => pool[(first + k) % pool.length]).sort((a, b) => t.tau[a] - t.tau[b]);
    }

    function renderTiles() {
      const t = d.tokens;
      shown = pick();
      tiles.forEach((tile, k) => {
        const i = shown[k];
        tile.card.hidden = i === undefined;
        if (i === undefined) {
          return;
        }
        tile.head.innerHTML = "";
        const level = document.createElement("span");
        level.className = "lvl";
        level.textContent = `L${t.difficulty[i]}`;
        const expr = document.createElement("code");
        expr.textContent = `${prettyExpression(t.question[i])} = ${t.answer[i]}`;
        tile.head.append(level, expr);
        const tau = t.tau[i];
        const learned = tau < d.edges[0] ? `learned before ${kSteps(d.edges[0])} steps` : tau > d.hi ? `not learned by ${kSteps(d.hi)} steps` : `learned at \u03c4 \u2248 ${kSteps(tau)} steps`;
        tile.foot.textContent = learned;
        drawTile(tile.plot, {
          width: 240,
          height: 104,
          xs: d.x,
          loss: t.lossOf(i),
          fit: (v) => sigmoidAt(v, tau, t.lambda[i], d.start, d.end),
          tau,
          lambda: t.lambda[i],
          color: d.color(tau),
          xDomain: [0, d.hi],
          yDomain: [-0.12, d.start + 0.25],
          fmtX: kSteps,
          title: `${t.question[i]} = ${t.answer[i]} (mod 23), difficulty ${t.difficulty[i]}, ${learned}`,
        });
      });
      if (caption) {
        caption.textContent =
          state.mode === "time"
            ? `Problems ${columnText(state.column)}. Each card shows the loss on the answer token (mod 23) and its fitted sigmoid.`
            : "Problems drawn at random from all 10,000, sorted by learning time; circles mark them in the grid.";
      }
    }

    function render(user) {
      setPressed(modeButtons, "mode", state.mode);
      if (!d) {
        return;
      }
      renderTiles();
      drawGrid();
      if (user && o.onChange) {
        o.onChange(query());
      }
    }

    function query() {
      const q = new URLSearchParams({ mode: state.mode });
      if (state.mode === "time") {
        q.set("col", String(state.column));
      }
      if (state.draw) {
        q.set("draw", String(state.draw));
      }
      return q.toString();
    }

    function columnAt(event) {
      if (!layout) {
        return null;
      }
      const p = P.pointer(svg, event);
      const c = Math.floor((p.x - layout.left) / layout.cw);
      return c >= 0 && c < layout.cols ? c : null;
    }

    zone.style.cursor = "pointer";
    zone.addEventListener("pointermove", (e) => {
      const c = columnAt(e);
      if (c !== hoverColumn) {
        hoverColumn = c;
        drawGrid();
      }
    });
    zone.addEventListener("pointerleave", () => {
      hoverColumn = null;
      drawGrid();
    });
    zone.addEventListener("click", (e) => {
      const c = columnAt(e);
      if (c !== null) {
        state.mode = "time";
        state.column = c;
        state.draw = 0;
        render(true);
      }
    });
    for (const b of modeButtons) {
      b.addEventListener("click", () => {
        state.mode = b.dataset.mode;
        state.draw = 0;
        render(true);
      });
    }
    root.querySelector("[data-redraw]").addEventListener("click", () => {
      state.draw += 1;
      render(true);
    });
    P.observeWidth(svg.parentNode, (w) => {
      width = w;
      drawGrid();
    });
    setPressed(modeButtons, "mode", state.mode);
    loadJson("static/data/synthetic.json", prepareMano)
      .then((loaded) => {
        d = loaded;
        render(false);
      })
      .catch((error) => loadFailed(tilesBox, error));

    return {
      query,
      svgs: () => [svg].concat(tiles.filter((t) => !t.card.hidden).map((t) => t.plot)),
      title: () => "Synthetic problems and their loss trajectories (stepwise-token-learning)",
    };
  }

  window.TLSFigures = { AXES, loadAxis, initRealTokens, initRealPanels, initMano };
})();
