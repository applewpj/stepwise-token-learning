// Toy model for "Explore the mechanism": synthetic tokens, each learned in one sigmoid step, with
// power-law learning times. A playhead runs through training and all three panels follow it.
(function () {
  "use strict";

  // Learning times run past the end of the axis so that the derivative has no edge effect at
  // a = axisMax; the burn-in before a = burnIn is left out of the log-log panel and the slope fit.
  const TOY = {
    tokens: 2000,
    axisMax: 100,
    burnIn: 10,
    tauMin: 2,
    tauMax: 150,
    shown: 40,
    lambdaSpread: 0.25,
    spectrumBins: 14,
    histBins: 20,
    playSeconds: 8,
  };

  function mulberry32(seed) {
    let state = seed >>> 0;
    return function () {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function linspace(lo, hi, n) {
    return Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1));
  }

  function logspace(lo, hi, n) {
    return linspace(Math.log(lo), Math.log(hi), n).map((v) => Math.exp(v));
  }

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  function sampleTau(u, alpha) {
    const lo = TOY.tauMin;
    const hi = TOY.tauMax;
    if (Math.abs(1 - alpha) < 1e-6) {
      return lo * Math.pow(hi / lo, u);
    }
    const k = 1 - alpha;
    const a = Math.pow(lo, k);
    const b = Math.pow(hi, k);
    return Math.pow(a + u * (b - a), 1 / k);
  }

  /* Density of the truncated power law that sampleTau draws from. */
  function tauDensity(tau, alpha) {
    const lo = TOY.tauMin;
    const hi = TOY.tauMax;
    if (tau < lo || tau > hi) {
      return 0;
    }
    if (Math.abs(1 - alpha) < 1e-6) {
      return 1 / (tau * Math.log(hi / lo));
    }
    const k = 1 - alpha;
    return (k * Math.pow(tau, -alpha)) / (Math.pow(hi, k) - Math.pow(lo, k));
  }

  // Everything except the spectrum exponent and the step width is drawn once per seed,
  // so moving a slider morphs the same tokens instead of drawing new ones.
  function drawTokens(seed) {
    const n = TOY.tokens;
    const rand = mulberry32(seed);
    const u = new Float64Array(n);
    const z = new Float64Array(n);
    const delta = new Float64Array(n);
    const post = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      u[i] = (i + rand()) / n;
      z[i] = Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
      delta[i] = 0.4 + 1.6 * rand();
      post[i] = 0.1 + 1.1 * rand();
    }
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    return { u, z, delta, post, order };
  }

  function logSlope(xs, ys, range) {
    let n = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (let k = 0; k < xs.length; k++) {
      if (xs[k] < range[0] || xs[k] > range[1] || !(ys[k] > 0)) {
        continue;
      }
      const x = Math.log10(xs[k]);
      const y = Math.log10(ys[k]);
      n += 1;
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
    }
    return n > 1 ? (n * sxy - sx * sy) / (n * sxx - sx * sx) : NaN;
  }

  function simulate(tokens, alpha, width) {
    const n = tokens.u.length;
    const lambdaMean = (2 * Math.log(9)) / width; // width is the 10%-90% transition length
    const tau = new Float64Array(n);
    const lambda = new Float64Array(n);
    let deltaSum = 0;
    for (let i = 0; i < n; i++) {
      tau[i] = sampleTau(tokens.u[i], alpha);
      lambda[i] = lambdaMean * Math.exp(TOY.lambdaSpread * tokens.z[i]);
      deltaSum += tokens.delta[i];
    }
    const lossAt = (i, a) => tokens.post[i] + tokens.delta[i] * (1 - sigmoid(lambda[i] * (a - tau[i])));

    const x = linspace(0, TOY.axisMax, 201);
    const meanLoss = x.map((a) => {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += lossAt(i, a);
      }
      return sum / n;
    });
    const shown = tokens.order
      .slice(0, TOY.shown)
      .sort((i, j) => tau[i] - tau[j])
      .map((i) => ({ tau: tau[i], loss: x.map((a) => lossAt(i, a)) }));

    const range = [TOY.burnIn, TOY.axisMax];
    const xRate = logspace(range[0], range[1], 80);
    const rate = xRate.map((a) => {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const s = sigmoid(lambda[i] * (a - tau[i]));
        sum += tokens.delta[i] * lambda[i] * s * (1 - s);
      }
      return sum / n;
    });

    const bins = TOY.spectrumBins;
    const edges = logspace(range[0], range[1], bins + 1);
    const counts = new Array(bins).fill(0);
    const logSpan = Math.log(range[1] / range[0]);
    const histWidth = TOY.axisMax / TOY.histBins;
    const hist = new Array(TOY.histBins).fill(0);
    for (let i = 0; i < n; i++) {
      if (tau[i] >= range[0] && tau[i] < range[1]) {
        counts[Math.min(bins - 1, Math.floor((Math.log(tau[i] / range[0]) / logSpan) * bins))] += 1;
      }
      if (tau[i] < TOY.axisMax) {
        hist[Math.floor(tau[i] / histWidth)] += 1 / n;
      }
    }
    const meanDelta = deltaSum / n;
    const spectrum = {
      x: counts.map((_, b) => Math.sqrt(edges[b] * edges[b + 1])),
      y: counts.map((c, b) => (meanDelta * c) / (n * (edges[b + 1] - edges[b]))),
    };

    return {
      x,
      meanLoss,
      shown,
      tau,
      hist,
      xRate,
      rate,
      spectrum,
      slopes: { rate: logSlope(xRate, rate, range), spectrum: logSlope(spectrum.x, spectrum.y, range) },
    };
  }

  function formatSlope(value) {
    if (!Number.isFinite(value)) {
      return "n/a";
    }
    if (Math.abs(value) < 0.005) {
      return "0.00";
    }
    return `${value < 0 ? "\u2212" : "+"}${Math.abs(value).toFixed(2)}`;
  }

  function describe(alpha, slopes) {
    const head = `Log-log slope after burn-in: loss derivative ${formatSlope(slopes.rate)}, spectrum ${formatSlope(slopes.spectrum)}.`;
    if (alpha < 0.1) {
      return `${head} A flat spectrum gives a flat derivative: the loss falls linearly, with no power law.`;
    }
    return `${head} The loss derivative inherits the power law of the spectrum.`;
  }

  /* Rendering */

  function initToy(root) {
    const P = window.TLSPlot;
    const C = P.C;
    const ACCENT = "#4b44ce";
    const inputs = {
      alpha: root.querySelector("#toy-alpha"),
      width: root.querySelector("#toy-width"),
      time: root.querySelector("#toy-time"),
    };
    const outputs = {
      alpha: root.querySelector("#toy-alpha-value"),
      width: root.querySelector("#toy-width-value"),
      time: root.querySelector("#toy-time-value"),
    };
    const playButton = root.querySelector("#toy-play");
    const readout = root.querySelector("#toy-readout");
    const names = ["tokens", "spectrum", "rate"];
    const panels = {};
    for (const name of names) {
      const svg = root.querySelector(`#toy-${name}`);
      const stage = P.stage(svg);
      panels[name] = { svg, stage, zone: stage.zone(), width: 0, view: null };
    }

    let seed = 2026;
    let tokens = drawTokens(seed);
    let sim = null;
    let a = Number(inputs.time.value);
    let raf = 0;
    let play = null;
    let touched = false;

    const tauColor = (tau) => P.timeColor(Math.log(Math.max(tau, TOY.tauMin) / TOY.tauMin) / Math.log(TOY.axisMax / TOY.tauMin));

    function recompute() {
      sim = simulate(tokens, Number(inputs.alpha.value), Number(inputs.width.value));
    }

    function sizes(panel) {
      const W = panel.width;
      const H = Math.round(Math.min(300, Math.max(220, W * 0.74)));
      panel.stage.resize(W, H);
      P.place(panel.zone, { x: 0, y: 0, w: W, h: H });
      return { W, H, box: { x: 46, y: 34, w: W - 46 - 10, h: H - 34 - 40 } };
    }

    function title(panel, step, text) {
      const g = panel.stage.content;
      P.el("circle", { cx: 9, cy: 11, r: 9, fill: ACCENT }, g);
      P.text(g, 9, 15, String(step), { fill: "#fff", "font-size": 11, "font-weight": 700, "text-anchor": "middle" });
      P.text(g, 24, 15, text, { fill: C.ink, "font-size": 12.5, "font-weight": 650 });
    }

    function playhead(layer, view, value, label) {
      const px = view.sx(value);
      if (!(px >= view.box.x - 0.5 && px <= view.box.x + view.box.w + 0.5)) {
        return;
      }
      P.el("line", { x1: px, x2: px, y1: view.box.y - 4, y2: view.box.y + view.box.h, stroke: ACCENT, "stroke-width": 1.6 }, layer);
      if (label) {
        const right = px > view.box.x + view.box.w - 50;
        P.text(layer, px + (right ? -5 : 5), view.box.y + 8, label, {
          fill: ACCENT,
          "font-size": 11,
          "font-weight": 650,
          "text-anchor": right ? "end" : "start",
          stroke: "#fff",
          "stroke-width": 3.5,
          "paint-order": "stroke",
        });
      }
    }

    function drawTokensPanel() {
      const panel = panels.tokens;
      if (!panel.width) {
        return;
      }
      const { box } = sizes(panel);
      title(panel, 1, "Every token drops once");
      const sx = P.scale(false).set([0, TOY.axisMax]);
      const sy = P.scale(false).set([0, 3.2]);
      const frame = P.frame(panel.stage.content, box, sx, sy, {
        xLabel: "training progress a",
        yLabel: "loss",
        yLabelOffset: 32,
        xCount: 5,
        yCount: 4,
      });
      P.el("rect", { x: box.x, y: box.y, width: sx(TOY.burnIn) - box.x, height: box.h, fill: "#f1f2f4" }, frame.data);
      const bx = (box.x + sx(TOY.burnIn)) / 2 + 3.5;
      const by = box.y + box.h - 6;
      P.text(frame.over, bx, by, "burn-in", {
        fill: C.muted,
        "font-size": 9.5,
        stroke: "#f1f2f4",
        "stroke-width": 3,
        "paint-order": "stroke",
        transform: `rotate(-90 ${bx} ${by})`,
      });
      panel.view = { box, sx, sy };
      const layer = frame.data;
      for (const token of sim.shown) {
        const learned = token.tau <= a;
        P.el(
          "path",
          {
            d: P.linePath(sim.x, token.loss, sx, sy),
            fill: "none",
            stroke: learned ? tauColor(token.tau) : "#c9ced4",
            "stroke-width": learned ? 1.4 : 1.1,
            "stroke-opacity": learned ? 0.95 : 0.6,
          },
          layer,
        );
      }
      const past = sim.x.map((v, k) => (v <= a + 1e-9 ? sim.meanLoss[k] : null));
      const future = sim.x.map((v, k) => (v >= a - 0.5 ? sim.meanLoss[k] : null));
      P.el("path", { d: P.linePath(sim.x, future, sx, sy), fill: "none", stroke: C.faint, "stroke-width": 1.8, "stroke-dasharray": "4 3" }, layer);
      const pastPath = P.linePath(sim.x, past, sx, sy);
      P.el("path", { d: pastPath, fill: "none", stroke: "#fff", "stroke-width": 7, "stroke-opacity": 0.85 }, layer);
      P.el("path", { d: pastPath, fill: "none", stroke: C.ink, "stroke-width": 3 }, layer);
      playhead(panel.stage.hover, panel.view, a, `a = ${Math.round(a)}`);
    }

    function drawSpectrumPanel() {
      const panel = panels.spectrum;
      if (!panel.width) {
        return;
      }
      const { box } = sizes(panel);
      title(panel, 2, "Learning times thin out");
      const alpha = Number(inputs.alpha.value);
      const binW = TOY.axisMax / TOY.histBins;
      const curveX = linspace(TOY.tauMin, TOY.axisMax, 160);
      const curve = curveX.map((t) => tauDensity(t, alpha) * binW);
      const top = Math.max(...sim.hist, ...curve.slice(4)) * 1.12;
      const sx = P.scale(false).set([0, TOY.axisMax]);
      const sy = P.scale(false).set([0, top]);
      const frame = P.frame(panel.stage.content, box, sx, sy, {
        xLabel: "learning time \u03c4",
        yLabel: "share of tokens",
        yLabelOffset: 36,
        xCount: 5,
        yCount: 4,
        yFormat: (v) => `${Math.round(v * 100)}%`,
      });
      panel.view = { box, sx, sy };
      sim.hist.forEach((share, j) => {
        const x0 = j * binW;
        const x1 = x0 + binW;
        const color = tauColor(x0 + binW / 2);
        const y = sy(share);
        const h = sy(0) - y;
        P.el("rect", { x: sx(x0) + 0.6, y, width: sx(x1) - sx(x0) - 1.2, height: h, fill: color, "fill-opacity": 0.2 }, frame.data);
        const learnedTo = Math.min(x1, a);
        if (learnedTo > x0) {
          P.el("rect", { x: sx(x0) + 0.6, y, width: Math.max(0, sx(learnedTo) - sx(x0) - (learnedTo === x1 ? 1.2 : 0.6)), height: h, fill: color, "fill-opacity": 0.95 }, frame.data);
        }
      });
      P.el("path", { d: P.linePath(curveX, curve, sx, sy), fill: "none", stroke: C.ink, "stroke-width": 1.4, "stroke-dasharray": "4 3" }, frame.data);
      const lx = sx(curveX[30]) + 6;
      P.text(frame.over, lx, sy(curve[30]) - 4, "p(\u03c4) \u221d \u03c4\u207b\u1d45", { fill: C.ink, "font-size": 11, "font-style": "italic", stroke: "#fff", "stroke-width": 3.5, "paint-order": "stroke" });
      let learned = 0;
      for (let i = 0; i < sim.tau.length; i++) {
        learned += sim.tau[i] <= a ? 1 : 0;
      }
      P.text(frame.over, box.x + box.w - 4, box.y + 12, `${Math.round((100 * learned) / sim.tau.length)}% learned by a = ${Math.round(a)}`, {
        fill: ACCENT,
        "font-size": 11,
        "font-weight": 650,
        "text-anchor": "end",
        stroke: "#fff",
        "stroke-width": 3.5,
        "paint-order": "stroke",
      });
      playhead(panel.stage.hover, panel.view, a, null);
    }

    function drawRatePanel() {
      const panel = panels.rate;
      if (!panel.width) {
        return;
      }
      const { box } = sizes(panel);
      title(panel, 3, "The loss follows the spectrum");
      const sx = P.scale(true).set([TOY.burnIn, TOY.axisMax]);
      const sy = P.scale(true).set([1e-4, 1e-1]);
      const frame = P.frame(panel.stage.content, box, sx, sy, {
        xLabel: "training progress a (log)",
        yLabel: "\u2212dL/da (log)",
        yLabelOffset: 36,
        xTicks: { major: [10, 20, 50, 100], minor: [30, 40, 60, 70, 80, 90] },
        yTicks: { major: [1e-4, 1e-3, 1e-2, 1e-1], minor: [] },
      });
      panel.view = { box, sx, sy };
      const sp = sim.spectrum;
      P.el("path", { d: P.linePath(sp.x, sp.y, sx, sy), fill: "none", stroke: C.spec, "stroke-width": 1.4, "stroke-opacity": 0.8 }, frame.data);
      P.el("path", { d: P.dotsPath(sp.x, sp.y, sx, sy), fill: "none", stroke: C.spec, "stroke-width": 6.5, "stroke-linecap": "round" }, frame.data);
      P.el("path", { d: P.linePath(sim.xRate, sim.rate, sx, sy), fill: "none", stroke: C.emp, "stroke-width": 2.6 }, frame.data);
      const lastRate = sim.rate[sim.rate.length - 1];
      const lastSpec = [...sp.y].reverse().find((v) => v > 0);
      const label = (y, text, color, below) =>
        P.text(frame.over, box.x + box.w - 4, Math.min(box.y + box.h - 4, Math.max(box.y + 12, sy(y) + (below ? 16 : -9))), text, {
          fill: color,
          "font-size": 11,
          "font-weight": 650,
          "text-anchor": "end",
          stroke: "#fff",
          "stroke-width": 3.5,
          "paint-order": "stroke",
        });
      if (lastRate > 0) {
        label(lastRate, `\u2212dL/da: slope ${formatSlope(sim.slopes.rate)}`, C.emp, false);
      }
      if (lastSpec > 0) {
        label(lastSpec, `p(\u03c4): slope ${formatSlope(sim.slopes.spectrum)}`, C.spec, true);
      }
      if (a >= TOY.burnIn) {
        let k = 0;
        while (k < sim.xRate.length - 1 && sim.xRate[k + 1] < a) {
          k++;
        }
        const f = Math.min(1, Math.max(0, (Math.log(a) - Math.log(sim.xRate[k])) / (Math.log(sim.xRate[k + 1] || a) - Math.log(sim.xRate[k]) || 1)));
        const y = Math.exp(Math.log(sim.rate[k]) + f * (Math.log(sim.rate[k + 1] || sim.rate[k]) - Math.log(sim.rate[k])));
        playhead(panel.stage.hover, panel.view, a, null);
        if (y > 0) {
          P.el("circle", { cx: sx(a), cy: sy(y), r: 4.5, fill: C.emp, stroke: "#fff", "stroke-width": 1.6 }, panel.stage.hover);
        }
      }
    }

    function updateText() {
      const alpha = Number(inputs.alpha.value);
      outputs.alpha.value = `\u03b1 = ${alpha.toFixed(2)}`;
      outputs.width.value = `width = ${Number(inputs.width.value).toFixed(1)}`;
      outputs.time.value = `a = ${Math.round(a)}`;
      inputs.time.value = String(a);
      const k = Math.min(sim.x.length - 1, Math.round((a / TOY.axisMax) * (sim.x.length - 1)));
      readout.textContent =
        `Loss at a = ${Math.round(a)}: ${sim.meanLoss[k].toFixed(2)}. ` +
        (alpha < 0.1
          ? "A flat spectrum gives a flat derivative: the loss falls linearly, with no power law."
          : `After burn-in, \u2212dL/da falls with slope ${formatSlope(sim.slopes.rate)} on log-log axes and p(\u03c4) with ${formatSlope(sim.slopes.spectrum)}: ` +
            "the loss inherits the spectrum's power law, whatever the step width.");
    }

    function render() {
      raf = 0;
      if (!sim) {
        recompute();
      }
      drawTokensPanel();
      drawSpectrumPanel();
      drawRatePanel();
      updateText();
    }

    function schedule() {
      if (!raf) {
        raf = requestAnimationFrame(render);
      }
    }

    function setPlaying(on) {
      if (!on && play) {
        cancelAnimationFrame(play.frame);
        play = null;
      }
      playButton.setAttribute("aria-pressed", String(on));
      playButton.querySelector("span").textContent = on ? "Pause" : "Play";
    }

    function startPlay() {
      if (a >= TOY.axisMax - 0.5) {
        a = 0;
      }
      setPlaying(true);
      const from = a;
      const duration = (TOY.playSeconds * 1000 * (TOY.axisMax - from)) / TOY.axisMax;
      play = { start: performance.now(), frame: 0 };
      const step = (now) => {
        if (!play) {
          return;
        }
        const f = Math.min(1, (now - play.start) / duration);
        a = from + f * (TOY.axisMax - from);
        render();
        if (f < 1) {
          play.frame = requestAnimationFrame(step);
        } else {
          play = null;
          setPlaying(false);
        }
      };
      play.frame = requestAnimationFrame(step);
    }

    playButton.addEventListener("click", () => {
      touched = true;
      if (play) {
        setPlaying(false);
      } else {
        startPlay();
      }
    });
    inputs.time.addEventListener("input", () => {
      touched = true;
      setPlaying(false);
      a = Number(inputs.time.value);
      schedule();
    });
    for (const key of ["alpha", "width"]) {
      inputs[key].addEventListener("input", () => {
        touched = true;
        recompute();
        schedule();
      });
    }
    root.querySelector("#toy-resample").addEventListener("click", () => {
      touched = true;
      seed += 1;
      tokens = drawTokens(seed);
      recompute();
      schedule();
    });

    for (const name of names) {
      const panel = panels[name];
      const scrub = (event) => {
        const view = panel.view;
        if (!view) {
          return;
        }
        const p = P.pointer(panel.svg, event);
        const v = view.sx.invert(Math.min(view.box.x + view.box.w, Math.max(view.box.x, p.x)));
        touched = true;
        setPlaying(false);
        a = Math.round(Math.min(TOY.axisMax, Math.max(name === "rate" ? TOY.burnIn : 0, v)) * 2) / 2;
        schedule();
      };
      panel.zone.style.cursor = "ew-resize";
      panel.zone.addEventListener("pointerdown", (event) => {
        panel.zone.setPointerCapture(event.pointerId);
        scrub(event);
      });
      panel.zone.addEventListener("pointermove", (event) => {
        if (event.buttons & 1) {
          scrub(event);
        }
      });
      P.observeWidth(panel.svg.parentNode, (w) => {
        panel.width = w;
        schedule();
      });
    }

    recompute();
    render();

    const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!still && typeof IntersectionObserver === "function") {
      const watcher = new IntersectionObserver(
        (entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            watcher.disconnect();
            if (!touched) {
              a = 0;
              startPlay();
            }
          }
        },
        { threshold: 0.45 },
      );
      watcher.observe(root.querySelector(".toy-panels"));
    }
  }

  if (typeof module === "object" && module.exports) {
    module.exports = { TOY, drawTokens, simulate, describe };
  }

  if (typeof document !== "undefined") {
    const start = () => {
      const toy = document.getElementById("toy");
      if (toy && window.TLSPlot) {
        initToy(toy);
      }
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  }
})();
