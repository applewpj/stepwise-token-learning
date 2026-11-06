// Minimal SVG plotting helpers shared by the figures on the project page.
(function (root) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif';
  const C = {
    ink: "#16181d",
    text: "#3d434b",
    muted: "#646b74",
    faint: "#9aa1a9",
    grid: "#edeff2",
    axis: "#8f969e",
    emp: "#3b6db3",
    recon: "#d1453b",
    spec: "#3f9a55",
    sig: "#7d8590",
  };
  const VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"];
  const SUP = { "-": "\u207b", 0: "\u2070", 1: "\u00b9", 2: "\u00b2", 3: "\u00b3", 4: "\u2074", 5: "\u2075", 6: "\u2076", 7: "\u2077", 8: "\u2078", 9: "\u2079" };

  let uid = 0;

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    if (attrs) {
      for (const key in attrs) {
        if (attrs[key] !== undefined && attrs[key] !== null) {
          node.setAttribute(key, attrs[key]);
        }
      }
    }
    if (parent) {
      parent.appendChild(node);
    }
    return node;
  }

  function text(parent, x, y, content, attrs) {
    const node = el("text", Object.assign({ x, y, fill: C.muted, "font-size": 11 }, attrs), parent);
    node.textContent = content;
    return node;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  // A scale maps data to pixels; `set` updates it in place so closures stay valid during animations.
  function scale(log) {
    const t = log ? Math.log10 : (v) => v;
    const s = (v) => s.r0 + ((t(v) - s.t0) / (s.t1 - s.t0)) * (s.r1 - s.r0);
    s.log = Boolean(log);
    s.set = (domain, range) => {
      s.domain = domain.slice();
      s.t0 = t(domain[0]);
      s.t1 = t(domain[1]);
      if (range) {
        s.r0 = range[0];
        s.r1 = range[1];
      }
      return s;
    };
    s.invert = (px) => {
      const v = s.t0 + ((px - s.r0) / (s.r1 - s.r0)) * (s.t1 - s.t0);
      return log ? Math.pow(10, v) : v;
    };
    return s;
  }

  function sup(n) {
    return String(n)
      .split("")
      .map((ch) => SUP[ch] || ch)
      .join("");
  }

  function trim(v, digits) {
    return String(Number(Number(v).toPrecision(digits || 3)));
  }

  function fmtTick(v) {
    const a = Math.abs(v);
    if (a < 1e-12) {
      return "0";
    }
    if (a >= 1e4 && a < 1e7) {
      return `${trim(v / 1e3)}k`.replace("-", "\u2212");
    }
    if (a >= 1e-2 && a < 1e4) {
      return trim(v).replace("-", "\u2212");
    }
    const e = Math.floor(Math.log10(a) + 1e-9);
    const m = v / Math.pow(10, e);
    return `${Math.abs(m - 1) < 1e-6 ? "" : `${trim(m, 2)}\u00d7`}10${sup(e)}`.replace("-", "\u2212");
  }

  // Human-readable numbers for readouts: 31,234 / 2.05 / 0.00412.
  function fmtNum(v, digits) {
    if (!Number.isFinite(v)) {
      return "n/a";
    }
    const a = Math.abs(v);
    if (a >= 1000) {
      return Math.round(v).toLocaleString("en-US");
    }
    return trim(v, digits || 3).replace("-", "\u2212");
  }

  function linearTicks(lo, hi, count) {
    const span = hi - lo;
    if (!(span > 0)) {
      return [lo];
    }
    let step = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = span / count / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    const ticks = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
      ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return ticks;
  }

  // Labels every multiple on narrow ranges, 1-2-5 on medium ranges, decades on wide ranges.
  function logTicks(lo, hi, maxLabels) {
    const decades = Math.log10(hi / lo);
    const mantissas = decades < 1.1 ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : decades < 2.3 ? [1, 2, 5] : [1];
    const major = [];
    const minor = [];
    for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) {
      for (let m = 1; m <= 9; m++) {
        const v = m * Math.pow(10, e);
        if (v < lo * (1 - 1e-9) || v > hi * (1 + 1e-9)) {
          continue;
        }
        (mantissas.includes(m) ? major : minor).push(v);
      }
    }
    let labelled = major;
    const limit = maxLabels || 7;
    if (labelled.length > limit) {
      const decadeTicks = major.filter((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
      const stride = Math.ceil(labelled.length / limit);
      labelled = decades >= 1.1 && decadeTicks.length >= 2 ? decadeTicks : major.filter((_, k) => k % stride === 0);
    }
    return { major: labelled, minor: minor.concat(major.filter((v) => !labelled.includes(v))) };
  }

  function ticksFor(s, count) {
    const [lo, hi] = s.domain[0] < s.domain[1] ? s.domain : [s.domain[1], s.domain[0]];
    if (s.log) {
      return logTicks(lo, hi, count);
    }
    return { major: linearTicks(lo, hi, count || 5), minor: [] };
  }

  // Draws a plot frame with ticks and labels and returns layers for data (clipped) and overlays.
  function frame(svg, box, sx, sy, opt) {
    const o = opt || {};
    const g = el("g", {}, svg);
    const x0 = box.x;
    const x1 = box.x + box.w;
    const y0 = box.y + box.h;
    const y1 = box.y;
    sx.set(sx.domain, [x0, x1]);
    sy.set(sy.domain, [y0, y1]);
    const grid = el("g", { stroke: C.grid, "stroke-width": 1 }, g);
    const axes = el("g", { "font-size": o.fontSize || 11, fill: C.muted }, g);
    const xt = o.xTicks || ticksFor(sx, o.xCount || 5);
    const yt = o.yTicks || ticksFor(sy, o.yCount || 5);
    const fx = o.xFormat || fmtTick;
    const fy = o.yFormat || fmtTick;
    for (const v of xt.major) {
      const px = sx(v);
      if (o.grid !== false) {
        el("line", { x1: px, x2: px, y1, y2: y0 }, grid);
      }
      el("line", { x1: px, x2: px, y1: y0, y2: y0 + 4, stroke: C.axis }, axes);
      if (!o.hideXLabels) {
        text(axes, px, y0 + 15, fx(v), { "text-anchor": "middle" });
      }
    }
    for (const v of xt.minor) {
      el("line", { x1: sx(v), x2: sx(v), y1: y0, y2: y0 + 2.5, stroke: C.faint }, axes);
    }
    for (const v of yt.major) {
      const py = sy(v);
      if (o.grid !== false) {
        el("line", { x1: x0, x2: x1, y1: py, y2: py }, grid);
      }
      el("line", { x1: x0 - 4, x2: x0, y1: py, y2: py, stroke: C.axis }, axes);
      if (!o.hideYLabels) {
        text(axes, x0 - 7, py + 3.5, fy(v), { "text-anchor": "end" });
      }
    }
    for (const v of yt.minor) {
      el("line", { x1: x0 - 2.5, x2: x0, y1: sy(v), y2: sy(v), stroke: C.faint }, axes);
    }
    el("path", { d: `M${x0},${y1}V${y0}H${x1}`, fill: "none", stroke: C.axis }, axes);
    if (o.xLabel) {
      text(axes, (x0 + x1) / 2, y0 + 31, o.xLabel, { "text-anchor": "middle", fill: C.text, "font-size": 11.5 });
    }
    if (o.yLabel) {
      text(axes, -(y0 + y1) / 2, box.x - (o.yLabelOffset || 44), o.yLabel, {
        transform: "rotate(-90)",
        "text-anchor": "middle",
        fill: C.text,
        "font-size": 11.5,
      });
    }
    uid += 1;
    const clipId = `clip-${uid}`;
    const clip = el("clipPath", { id: clipId }, el("defs", {}, g));
    el("rect", { x: x0, y: y1 - 1, width: x1 - x0, height: y0 - y1 + 2 }, clip);
    const data = el("g", { "clip-path": `url(#${clipId})` }, g);
    const over = el("g", {}, g);
    return { g, data, over, x0, x1, y0, y1 };
  }

  function linePath(xs, ys, sx, sy) {
    let d = "";
    let pen = false;
    for (let k = 0; k < xs.length; k++) {
      const x = xs[k];
      const y = ys[k];
      if (y === null || !Number.isFinite(y) || !Number.isFinite(x) || (sy.log && y <= 0) || (sx.log && x <= 0)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
      pen = true;
    }
    return d;
  }

  function dotsPath(xs, ys, sx, sy) {
    let d = "";
    for (let k = 0; k < xs.length; k++) {
      const y = ys[k];
      if (y === null || !Number.isFinite(y) || (sy.log && y <= 0)) {
        continue;
      }
      d += `M${sx(xs[k]).toFixed(1)},${sy(y).toFixed(1)}h0`;
    }
    return d;
  }

  function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  const VIRIDIS_RGB = VIRIDIS.map(hexToRgb);

  function viridis(t) {
    const s = Math.min(1, Math.max(0, t)) * (VIRIDIS_RGB.length - 1);
    const k = Math.min(VIRIDIS_RGB.length - 2, Math.floor(s));
    const f = s - k;
    const c = VIRIDIS_RGB[k].map((v, j) => Math.round(v + f * (VIRIDIS_RGB[k + 1][j] - v)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  // Learning time to colour, shared by every figure: early is dark purple, late is green-yellow.
  function timeColor(frac) {
    return viridis(0.04 + 0.82 * frac);
  }

  function gradientCss() {
    return `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((f) => timeColor(f)).join(", ")})`;
  }

  function svgRoot(svg, width, height) {
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.setAttribute("font-family", FONT);
    return svg;
  }

  // Persistent layers for interactive charts: `content` and `hover` are redrawn, `zones` keep
  // their listeners (and pointer capture) across redraws.
  function stage(svg) {
    svg.setAttribute("font-family", FONT);
    const content = el("g", {}, svg);
    const hover = el("g", { "pointer-events": "none" }, svg);
    const zones = el("g", { "data-zones": "" }, svg);
    return {
      svg,
      content,
      hover,
      zones,
      resize(width, height) {
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.setAttribute("width", width);
        svg.setAttribute("height", height);
        clear(content);
        clear(hover);
      },
      zone(cls) {
        return el("rect", { fill: "transparent", class: cls || null }, zones);
      },
    };
  }

  function place(rect, box) {
    rect.setAttribute("x", box.x);
    rect.setAttribute("y", box.y);
    rect.setAttribute("width", Math.max(0, box.w));
    rect.setAttribute("height", Math.max(0, box.h));
  }

  // Pointer position in the SVG's own coordinates.
  function pointer(svg, event) {
    const rect = svg.getBoundingClientRect();
    const vb = svg.viewBox.baseVal;
    const kx = vb && vb.width ? vb.width / rect.width : 1;
    const ky = vb && vb.height ? vb.height / rect.height : 1;
    return { x: (event.clientX - rect.left) * kx, y: (event.clientY - rect.top) * ky };
  }

  // Calls `render(width)` now and whenever the element's width changes.
  function observeWidth(node, render) {
    let last = -1;
    let frame = 0;
    const run = () => {
      frame = 0;
      const width = Math.floor(node.clientWidth);
      if (width > 0 && width !== last) {
        last = width;
        render(width);
      }
    };
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => {
        if (!frame) {
          frame = requestAnimationFrame(run);
        }
      }).observe(node);
    } else {
      window.addEventListener("resize", () => {
        if (!frame) {
          frame = requestAnimationFrame(run);
        }
      });
    }
    run();
    return () => {
      last = -1;
      run();
    };
  }

  // Saves one or more chart SVGs as a single file, laid out as they appear on the page.
  function downloadSvg(svgs, filename, title) {
    const list = (Array.isArray(svgs) ? svgs : [svgs]).filter((s) => s.getBoundingClientRect().width > 0);
    if (!list.length) {
      return;
    }
    const rects = list.map((s) => s.getBoundingClientRect());
    const left = Math.min(...rects.map((r) => r.left));
    const top = Math.min(...rects.map((r) => r.top));
    const head = title ? 24 : 0;
    const width = Math.ceil(Math.max(...rects.map((r) => r.right)) - left);
    const height = Math.ceil(Math.max(...rects.map((r) => r.bottom)) - top) + head;
    const out = document.createElementNS(NS, "svg");
    out.setAttribute("xmlns", NS);
    out.setAttribute("viewBox", `0 0 ${width} ${height}`);
    out.setAttribute("width", width);
    out.setAttribute("height", height);
    out.setAttribute("font-family", FONT);
    el("rect", { width, height, fill: "#fff" }, out);
    if (title) {
      text(out, 0, 14, title, { fill: C.ink, "font-size": 12, "font-weight": 600 });
    }
    list.forEach((svg, k) => {
      const r = rects[k];
      const vb = svg.viewBox.baseVal;
      const box = vb && vb.width ? vb : { x: 0, y: 0, width: r.width };
      const factor = r.width / box.width;
      const copy = svg.cloneNode(true);
      copy.querySelectorAll("[data-zones]").forEach((node) => node.remove());
      const g = el("g", { transform: `translate(${(r.left - left).toFixed(1)},${(r.top - top + head).toFixed(1)}) scale(${factor.toFixed(4)}) translate(${-box.x},${-box.y})` }, out);
      while (copy.firstChild) {
        g.appendChild(copy.firstChild);
      }
    });
    const blob = new Blob([new XMLSerializer().serializeToString(out)], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  root.TLSPlot = {
    C,
    FONT,
    el,
    text,
    clear,
    scale,
    fmtTick,
    fmtNum,
    linearTicks,
    logTicks,
    frame,
    linePath,
    dotsPath,
    viridis,
    timeColor,
    gradientCss,
    svgRoot,
    stage,
    place,
    pointer,
    observeWidth,
    downloadSvg,
  };
})(typeof window !== "undefined" ? window : globalThis);
