#!/usr/bin/env python3
"""Draw the images of the README and the dataset card, in the style of the project page.

    python tools/make_readme_assets.py                 # everything in assets/readme/
    python tools/make_readme_assets.py --only preview  # banner, buttons or preview

The banner plots sigmoid fits from the paper-setting reproduction of Figure 3, so run
`tokenlearning reproduce fig3` first. The preview replays the mechanism toy of
docs/static/js/toy.js with the page's seed and default sliders, so it shows the tokens the
page shows.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors, patheffects  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "assets" / "readme"
FIT_CACHE = REPO_ROOT / "runs" / "step_axis_1m" / "cache"

# Colours of docs/static/css/index.css (page) and docs/static/js/plot.js (charts).
INK = "#1f2328"
MUTED = "#5d6670"
FAINT = "#8c959f"
LINE = "#e3e6ea"
ACCENT = "#4b44ce"
ACCENT_SOFT = "#efeefc"
C = {
    "ink": "#16181d",
    "text": "#3d434b",
    "muted": "#646b74",
    "faint": "#9aa1a9",
    "grid": "#edeff2",
    "axis": "#8f969e",
    "emp": "#3b6db3",
    "spec": "#3f9a55",
}
VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"]
SANS = ["Cantarell", "DejaVu Sans"]
SERIF = ["STIXGeneral"]
MONO = ["DejaVu Sans Mono"]

plt.rcParams.update({"font.family": SANS, "mathtext.fontset": "cm", "lines.scale_dashes": False})


def time_color(frac: float) -> tuple:
    """Learning time to colour, as `timeColor` in plot.js: early is dark purple, late is green-yellow."""
    stops = np.array([colors.to_rgb(c) for c in VIRIDIS])
    s = min(1.0, max(0.0, 0.04 + 0.82 * frac)) * (len(stops) - 1)
    k = min(len(stops) - 2, int(s))
    return tuple(np.round((stops[k] + (s - k) * (stops[k + 1] - stops[k])) * 255) / 255)


# Drawing in CSS pixels


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float


class Canvas:
    """A figure in CSS pixels with y pointing down, like the page's SVG; 1 pt is 1 CSS pixel."""

    def __init__(self, width: float, height: float, scale: float) -> None:
        self.width, self.height, self.scale = width, height, scale
        self.fig = plt.figure(figsize=(width / 72, height / 72), dpi=72 * scale)
        self.fig.patch.set_alpha(0)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_axis_off()
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)

    def text(self, x, y, s, size, color, *, family=SANS, weight="normal", style="normal", anchor="start",
             va="baseline", halo=None, rotation=0, z=6):
        ha = {"start": "left", "middle": "center", "end": "right"}[anchor]
        t = self.ax.text(x, y, s, fontsize=size, color=color, family=family, fontweight=weight, fontstyle=style,
                         ha=ha, va=va, rotation=rotation, rotation_mode="anchor", zorder=z)
        if halo:
            t.set_path_effects([patheffects.withStroke(linewidth=halo[1], foreground=halo[0])])
        return t

    def measure(self, s, size, *, family=SANS, weight="normal", style="normal") -> float:
        """Advance width in CSS pixels, counting leading and trailing spaces."""
        renderer = self.fig.canvas.get_renderer()
        widths = []
        for probe in ("n" + s + "n", "nn"):
            t = self.ax.text(0, 0, probe, fontsize=size, family=family, fontweight=weight, fontstyle=style)
            widths.append(t.get_window_extent(renderer).width)
            t.remove()
        return (widths[0] - widths[1]) / self.scale

    def rich(self, x, y, parts, size, color, *, anchor="start", z=6) -> float:
        """Text in several styles on one baseline; parts are (text, weight, style). Returns the width."""
        widths = [self.measure(s, size, weight=w, style=st) for s, w, st in parts]
        x0 = x - {"start": 0, "middle": sum(widths) / 2, "end": sum(widths)}[anchor]
        for (s, w, st), width in zip(parts, widths):
            self.text(x0, y, s, size, color, weight=w, style=st, z=z)
            x0 += width
        return sum(widths)

    def rrect(self, x, y, w, h, r, fc, ec="none", lw=0.0, alpha=1.0, z=1):
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(r, w / 2, h / 2)}",
                               fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
        self.ax.add_patch(patch)
        return patch

    def rect(self, x, y, w, h, fc, alpha=1.0, z=2, clip=None):
        patch = Rectangle((x, y), w, h, fc=fc, ec="none", alpha=alpha, zorder=z)
        self.ax.add_patch(patch)
        if clip is not None:
            patch.set_clip_path(clip)
        return patch

    def line(self, xs, ys, color, lw, *, dash=None, alpha=1.0, clip=None, z=3, cap="round"):
        (ln,) = self.ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=z, solid_capstyle=cap,
                             solid_joinstyle="round", dash_capstyle="butt")
        if dash:
            ln.set_dashes(dash)
        if clip is not None:
            ln.set_clip_path(clip)
        return ln

    def dot(self, x, y, d, fc, ec="none", lw=0.0, z=7, clip=None):
        patch = Circle((x, y), d / 2, fc=fc, ec=ec, lw=lw, zorder=z)
        self.ax.add_patch(patch)
        if clip is not None:
            patch.set_clip_path(clip)
        return patch

    def gradient(self, box: Box, top: str, bottom: str, clip=None, z=0):
        ramp = np.linspace(0, 1, 256)[:, None, None]
        img = np.array(colors.to_rgb(top)) + ramp * (np.array(colors.to_rgb(bottom)) - np.array(colors.to_rgb(top)))
        im = self.ax.imshow(img, extent=(box.x, box.x + box.w, box.y + box.h, box.y), aspect="auto", zorder=z,
                            interpolation="bilinear")
        if clip is not None:
            im.set_clip_path(clip)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(self.height, 0)

    def icon(self, d_list, x, y, size, color, *, lw=2.0, fill=False, z=8):
        """SVG icon paths in a 24-unit box, drawn with the box's top-left corner at (x, y)."""
        k = size / 24
        transform = Affine2D().scale(k).translate(x, y) + self.ax.transData
        for d in d_list:
            patch = PathPatch(svg_path(d), transform=transform, zorder=z, capstyle="round", joinstyle="round",
                              fc=color if fill else "none", ec="none" if fill else color, lw=0 if fill else lw * k)
            self.ax.add_patch(patch)

    def clip_box(self, x0, y0, w, h):
        return Rectangle((x0, y0), w, h, transform=self.ax.transData)

    def rgba(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba()).copy()

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=72 * self.scale, transparent=True)
        plt.close(self.fig)


_TOKEN = re.compile(r"[MmLlHhVvCcSsAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "A": 7}


def _arc(x1, y1, rx, ry, phi, large, sweep, x2, y2):
    """Points along an SVG elliptical arc (endpoint to centre parameterization, SVG spec F.6.5)."""
    if rx == 0 or ry == 0:
        return [(x2, y2)]
    cos, sin = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    xp, yp = cos * dx + sin * dy, -sin * dx + cos * dy
    rx, ry = abs(rx), abs(ry)
    grow = xp**2 / rx**2 + yp**2 / ry**2
    if grow > 1:
        rx, ry = rx * math.sqrt(grow), ry * math.sqrt(grow)
    num = rx**2 * ry**2 - rx**2 * yp**2 - ry**2 * xp**2
    den = rx**2 * yp**2 + ry**2 * xp**2
    coef = math.sqrt(max(0.0, num / den)) if den else 0.0
    if bool(large) == bool(sweep):
        coef = -coef
    cxp, cyp = coef * rx * yp / ry, -coef * ry * xp / rx
    cx, cy = cos * cxp - sin * cyp + (x1 + x2) / 2, sin * cxp + cos * cyp + (y1 + y2) / 2
    ux, uy = (xp - cxp) / rx, (yp - cyp) / ry
    vx, vy = (-xp - cxp) / rx, (-yp - cyp) / ry
    t1 = math.atan2(uy, ux)
    dt = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(4, int(abs(dt) / (math.pi / 16)))
    pts = []
    for j in range(1, n + 1):
        t = t1 + dt * j / n
        pts.append((cos * rx * math.cos(t) - sin * ry * math.sin(t) + cx, sin * rx * math.cos(t) + cos * ry * math.sin(t) + cy))
    return pts


def svg_path(d: str) -> MplPath:
    """The subset of SVG path syntax that the page's icons use."""
    tokens = _TOKEN.findall(d)
    verts, codes = [], []
    x = y = x0 = y0 = 0.0
    ctrl = None
    cmd = ""
    i = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                verts.append((x0, y0))
                codes.append(MplPath.CLOSEPOLY)
                x, y, ctrl = x0, y0, None
                continue
        op = cmd.upper()
        args = [float(t) for t in tokens[i:i + _ARITY[op]]]
        i += _ARITY[op]
        ox, oy = (x, y) if cmd.islower() else (0.0, 0.0)
        if op == "M":
            x, y = ox + args[0], oy + args[1]
            x0, y0 = x, y
            verts.append((x, y))
            codes.append(MplPath.MOVETO)
            cmd = "l" if cmd.islower() else "L"
        elif op in "LHV":
            if op == "L":
                x, y = ox + args[0], oy + args[1]
            elif op == "H":
                x = ox + args[0]
            else:
                y = oy + args[0]
            verts.append((x, y))
            codes.append(MplPath.LINETO)
        elif op in "CS":
            if op == "C":
                c1 = (ox + args[0], oy + args[1])
                rest = args[2:]
            else:
                c1 = (2 * x - ctrl[0], 2 * y - ctrl[1]) if ctrl else (x, y)
                rest = args
            c2 = (ox + rest[0], oy + rest[1])
            x, y = ox + rest[2], oy + rest[3]
            verts += [c1, c2, (x, y)]
            codes += [MplPath.CURVE4] * 3
            ctrl = c2
            continue
        elif op == "A":
            end = (ox + args[5], oy + args[6])
            for p in _arc(x, y, *args[:5], *end):
                verts.append(p)
                codes.append(MplPath.LINETO)
            x, y = end
        ctrl = None
    return MplPath(verts, codes)


ICONS = {
    "paper": ["M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z", "M14 3v5h5M9 13h6M9 17h6"],
    "code": ["M16 18l6-6-6-6M8 6l-6 6 6 6"],
    "data": ["M4 5a8 3 0 1 0 16 0a8 3 0 1 0-16 0", "M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5",
             "M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"],
    "colab": ["M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0", "M10 8.5l5.5 3.5-5.5 3.5z"],
    "bibtex": ["M6 3h12v18l-6-4-6 4z"],
    "sliders": ["M3 6h18M3 12h18M3 18h18"],
    "play": ["M8 5.5v13l10.5-6.5z"],
    "pause": ["M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"],
}
SLIDER_KNOBS = [(15, 6), (8, 12), (16.5, 18)]


# Banner


def banner(path: Path) -> None:
    fit = np.load(FIT_CACHE / "fit_params.npz")
    kept = np.load(FIT_CACHE / "filtered_mask.npy")
    tau, lam, delta, r2 = (np.asarray(fit[k], float) for k in ("tau", "lambda_", "delta", "r2"))
    # The social card's selection: tokens learned inside the interior window of the step axis in
    # a sharp step (10-90% width under a tenth of the window), with a clear drop and a good fit.
    window, view = (25000.0, 60000.0), (20000.0, 65000.0)
    width = 2 * np.log(9) / lam
    good = (kept & np.isfinite(tau) & np.isfinite(lam) & (delta > 0.3) & (r2 > 0.6) & (tau >= window[0])
            & (tau < window[1]) & (width <= 0.1 * (window[1] - window[0])))
    idx = np.flatnonzero(good)
    idx = idx[np.argsort(tau[idx])]
    pick = idx[np.linspace(0, len(idx) - 1, 46).astype(int)]
    xs = np.linspace(*view, 600)

    def curves(rows):
        z = np.clip(lam[rows, None] * (xs[None, :] - tau[rows, None]), -40, 40)
        return 1.0 - 1.0 / (1.0 + np.exp(-z))

    avg = curves(idx).mean(axis=0)

    W, H = 880, 284
    cv = Canvas(W, H, 2.5)
    card = cv.rrect(0.75, 0.75, W - 1.5, H - 1.5, 20, "none", ec=LINE, lw=1.5, z=3)
    cv.gradient(Box(0, 0, W, H), "#f2f0ff", "#ffffff", clip=card)

    venue = "NeurIPS 2026  ·  Spotlight"
    vw = cv.measure(venue, 13.5, weight="bold") + 30
    cv.rrect(48, 38, vw, 28, 14, ACCENT_SOFT, z=4)
    cv.text(48 + vw / 2, 52, venue, 13.5, ACCENT, weight="bold", anchor="middle", va="center_baseline")
    cv.text(46, 118, "Smooth Scaling Laws Hide", 39, INK, family=SERIF, weight="bold")
    cv.text(46, 164, "Stepwise Token Learning", 39, INK, family=SERIF, weight="bold")
    cv.text(48, 210, "Each token's loss falls in one sigmoid-shaped step.", 15.5, MUTED)
    cv.text(48, 234, "When those steps happen sets the smooth scaling law.", 15.5, MUTED)

    box = Box(560, 50, 282, 180)
    sx = lambda v: box.x + (v - view[0]) / (view[1] - view[0]) * box.w  # noqa: E731
    sy = lambda v: box.y + (1.06 - v) / 1.10 * box.h  # noqa: E731
    cnorm = colors.Normalize(*window)
    viridis = matplotlib.colormaps["viridis"]
    for row, curve in zip(pick, curves(pick)):
        cv.line(sx(xs), sy(curve), viridis(cnorm(tau[row])), 1.0, alpha=0.85, z=4)
    cv.line(sx(xs), sy(avg), "#ffffff", 6.5, alpha=0.85, z=5)
    cv.line(sx(xs), sy(avg), C["ink"], 3.2, z=6)
    cv.line([box.x, box.x, box.x + box.w], [box.y - 6, box.y + box.h, box.y + box.h], "#9aa0a6", 1.0, cap="butt", z=4)
    cv.text(box.x + box.w / 2, box.y + box.h + 22, "training step", 12, MUTED, anchor="middle")
    cv.text(box.x - 10, box.y + box.h / 2, "normalized loss", 12, MUTED, anchor="middle", rotation=90)
    cv.text(box.x + box.w, box.y - 14, "individual tokens: one sharp step each", 12, "#2a788e", anchor="end")
    k = int(np.argmin(np.abs(avg - 0.42)))
    cv.line([sx(xs[k]) + 4, sx(xs[k]) + 40], [sy(avg[k]) - 4, sy(avg[k]) - 40], C["ink"], 0.9, z=6)
    cv.text(sx(xs[k]) + 44, sy(avg[k]) - 42, "their average: smooth", 12.5, C["ink"], weight="bold", z=7,
            halo=("#ffffff", 3.5))
    cv.save(path)
    print(f"banner: {len(idx)} tokens averaged, {len(pick)} drawn -> {path.relative_to(REPO_ROOT)}")


# Buttons


BUTTONS = {
    "page": ("Interactive project page", "sliders", ACCENT),
    "paper": ("Paper", "paper", INK),
    "data": ("Data", "data", INK),
    "code": ("Code", "code", INK),
    "colab": ("Colab", "colab", INK),
    "bibtex": ("BibTeX", "bibtex", INK),
}


def path_data(path: MplPath, x: float, y: float) -> str:
    """SVG path data of a y-up matplotlib path, with its origin moved to (x, y) in y-down SVG space."""
    def num(v: float) -> str:
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return "0" if s == "-0" else s

    letters = {MplPath.MOVETO: "M", MplPath.LINETO: "L", MplPath.CURVE3: "Q", MplPath.CURVE4: "C"}
    out = []
    for verts, code in path.iter_segments(curves=True, simplify=False):
        if code == MplPath.CLOSEPOLY:
            out.append("Z")
            continue
        out.append(letters[code] + " ".join(f"{num(x + px)} {num(y - py)}" for px, py in zip(verts[::2], verts[1::2])))
    return "".join(out)


def buttons(out_dir: Path) -> None:
    """The page's `.btn`: 0.95rem semibold label, 1.05em icon, 0.5em x 1.15em padding, pill shape.

    SVG with the label as outlines and its own 40 px height, so it keeps its size in Markdown
    renderers that drop `height` attributes and stays sharp on any screen.
    """
    font, icon, gap, pad_x, height = 16, 17, 7.3, 18.6, 40
    prop = FontProperties(family=SANS, weight="bold")
    baseline = height / 2 + TextPath((0, 0), "H", size=font, prop=prop).get_extents().y1 / 2
    for name, (label, glyph, fill) in BUTTONS.items():
        text = TextPath((0, 0), label, size=font, prop=prop)
        ink = text.get_extents()
        width = round(2 * pad_x + icon + gap + ink.width)
        left = (width - icon - gap - ink.width) / 2
        knobs = "".join(f'<circle cx="{kx}" cy="{ky}" r="3.1" fill="{fill}"/>' for kx, ky in SLIDER_KNOBS)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" rx="{height / 2:g}" fill="{fill}"/>'
            f'<g transform="translate({left:.2f} {(height - icon) / 2:g}) scale({icon / 24:.4f})" fill="none" '
            'stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            + "".join(f'<path d="{d}"/>' for d in ICONS[glyph])
            + (knobs if glyph == "sliders" else "")
            + "</g>"
            f'<path fill="#fff" d="{path_data(text, left + icon + gap - ink.x0, baseline)}"/>'
            "</svg>\n"
        )
        (out_dir / f"btn-{name}.svg").write_text(svg, encoding="utf-8")
    print(f"buttons: {', '.join(BUTTONS)} -> {out_dir.relative_to(REPO_ROOT)}")


# Mechanism preview: a port of docs/static/js/toy.js

TOY = {"tokens": 2000, "axis_max": 100, "burn_in": 10, "tau_min": 2, "tau_max": 150, "shown": 40,
       "lambda_spread": 0.25, "spectrum_bins": 14, "hist_bins": 20}
SEED = 2026
M32 = 0xFFFFFFFF


def mulberry32(seed: int):
    state = seed & M32

    def rand() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & M32
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


def draw_tokens(seed: int) -> dict:
    n = TOY["tokens"]
    rand = mulberry32(seed)
    u, z, delta, post = (np.zeros(n) for _ in range(4))
    for i in range(n):
        u[i] = (i + rand()) / n
        r1 = rand()
        r2 = rand()
        z[i] = math.sqrt(-2 * math.log(1 - r1)) * math.cos(2 * math.pi * r2)
        delta[i] = 0.4 + 1.6 * rand()
        post[i] = 0.1 + 1.1 * rand()
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(rand() * (i + 1))
        order[i], order[j] = order[j], order[i]
    return {"u": u, "z": z, "delta": delta, "post": post, "order": order}


def sample_tau(u, alpha):
    lo, hi = TOY["tau_min"], TOY["tau_max"]
    if abs(1 - alpha) < 1e-6:
        return lo * (hi / lo) ** u
    k = 1 - alpha
    return (lo**k + u * (hi**k - lo**k)) ** (1 / k)


def tau_density(tau, alpha):
    lo, hi = TOY["tau_min"], TOY["tau_max"]
    tau = np.asarray(tau, float)
    if abs(1 - alpha) < 1e-6:
        dens = 1 / (tau * math.log(hi / lo))
    else:
        k = 1 - alpha
        dens = k * tau ** (-alpha) / (hi**k - lo**k)
    return np.where((tau < lo) | (tau > hi), 0.0, dens)


def log_slope(xs, ys, lo, hi):
    xs, ys = np.asarray(xs), np.asarray(ys)
    ok = (xs >= lo) & (xs <= hi) & (ys > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log10(xs[ok]), np.log10(ys[ok]), 1)[0])


def simulate(tokens: dict, alpha: float, width: float) -> dict:
    n = TOY["tokens"]
    lam = 2 * math.log(9) / width * np.exp(TOY["lambda_spread"] * tokens["z"])
    tau = sample_tau(tokens["u"], alpha)
    delta, post = tokens["delta"], tokens["post"]
    x = np.linspace(0, TOY["axis_max"], 201)
    loss = post[:, None] + delta[:, None] * (1 - 1 / (1 + np.exp(-lam[:, None] * (x[None, :] - tau[:, None]))))
    shown = sorted(tokens["order"][:TOY["shown"]], key=lambda i: tau[i])
    lo, hi = TOY["burn_in"], TOY["axis_max"]
    x_rate = np.exp(np.linspace(math.log(lo), math.log(hi), 80))
    s = 1 / (1 + np.exp(-lam[:, None] * (x_rate[None, :] - tau[:, None])))
    rate = (delta[:, None] * lam[:, None] * s * (1 - s)).mean(axis=0)
    bins = TOY["spectrum_bins"]
    edges = np.exp(np.linspace(math.log(lo), math.log(hi), bins + 1))
    inside = (tau >= lo) & (tau < hi)
    which = np.minimum(bins - 1, np.floor(np.log(tau[inside] / lo) / math.log(hi / lo) * bins).astype(int))
    counts = np.bincount(which, minlength=bins)
    hist_w = TOY["axis_max"] / TOY["hist_bins"]
    early = tau < TOY["axis_max"]
    hist = np.bincount(np.floor(tau[early] / hist_w).astype(int), minlength=TOY["hist_bins"]) / n
    spec_x = np.sqrt(edges[:-1] * edges[1:])
    spec_y = delta.mean() * counts / (n * np.diff(edges))
    return {
        "x": x,
        "mean_loss": loss.mean(axis=0),
        "shown": [(tau[i], loss[i]) for i in shown],
        "tau": tau,
        "hist": hist,
        "x_rate": x_rate,
        "rate": rate,
        "spec_x": spec_x,
        "spec_y": spec_y,
        "slope_rate": log_slope(x_rate, rate, lo, hi),
        "slope_spec": log_slope(spec_x, spec_y, lo, hi),
    }


def fmt_slope(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    if abs(v) < 0.005:
        return "0.00"
    return f"{'−' if v < 0 else '+'}{abs(v):.2f}"


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_tick(v: float) -> str:
    a = abs(v)
    if a < 1e-12:
        return "0"
    if 1e-2 <= a < 1e4:
        return f"{float(f'{v:.3g}'):g}".replace("-", "−")
    e = math.floor(math.log10(a) + 1e-9)
    return f"10{str(e).translate(SUPERSCRIPT)}"


def linear_ticks(lo: float, hi: float, count: int) -> list[float]:
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / count))
    err = span / count / step
    step *= 10 if err >= 7.5 else 5 if err >= 3.5 else 2 if err >= 1.5 else 1
    ticks, v = [], math.ceil(lo / step - 1e-9) * step
    while v <= hi + step * 1e-9:
        ticks.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return ticks


class Scale:
    def __init__(self, d0, d1, r0, r1, log=False):
        self.log = log
        f = math.log10 if log else float
        self.t0, self.t1, self.r0, self.r1, self.f = f(d0), f(d1), r0, r1, f

    def __call__(self, v):
        v = np.asarray(v, float)
        t = np.log10(np.where(v > 0, v, np.nan)) if self.log else v
        return self.r0 + (t - self.t0) / (self.t1 - self.t0) * (self.r1 - self.r0)


def frame(cv: Canvas, box: Box, sx: Scale, sy: Scale, *, xt, yt, x_label, y_label, y_off, fy=fmt_tick,
          x_minor=(), y_minor=()):
    """plot.js `frame`: grid, ticks, tick labels, L-shaped axes and axis labels; returns the data clip."""
    x0, x1, y0, y1 = box.x, box.x + box.w, box.y + box.h, box.y
    for v in xt:
        px = float(sx(v))
        cv.line([px, px], [y1, y0], C["grid"], 1, cap="butt", z=1)
        cv.line([px, px], [y0, y0 + 4], C["axis"], 1, cap="butt", z=2)
        cv.text(px, y0 + 15, fmt_tick(v), 11, C["muted"], anchor="middle")
    for v in x_minor:
        cv.line([float(sx(v))] * 2, [y0, y0 + 2.5], C["faint"], 1, cap="butt", z=2)
    for v in yt:
        py = float(sy(v))
        cv.line([x0, x1], [py, py], C["grid"], 1, cap="butt", z=1)
        cv.line([x0 - 4, x0], [py, py], C["axis"], 1, cap="butt", z=2)
        cv.text(x0 - 7, py + 3.5, fy(v), 11, C["muted"], anchor="end")
    for v in y_minor:
        cv.line([x0 - 2.5, x0], [float(sy(v))] * 2, C["faint"], 1, cap="butt", z=2)
    cv.line([x0, x0, x1], [y1, y0, y0], C["axis"], 1, cap="butt", z=5)
    cv.text((x0 + x1) / 2, y0 + 31, x_label, 11.5, C["text"], anchor="middle")
    cv.text(box.x - y_off, (y0 + y1) / 2, y_label, 11.5, C["text"], anchor="middle", rotation=90)
    return cv.clip_box(x0, y1 - 1, x1 - x0, y0 - y1 + 2)


def panel_title(cv: Canvas, ox, oy, step: int, title: str) -> None:
    cv.dot(ox + 9, oy + 11, 18, ACCENT, z=6)
    cv.text(ox + 9, oy + 15, str(step), 11, "#ffffff", weight="bold", anchor="middle", z=7)
    cv.text(ox + 24, oy + 15, title, 12.5, C["ink"], weight="bold")


def playhead(cv: Canvas, box: Box, sx: Scale, a: float, label: str | None = None) -> None:
    px = float(sx(a))
    if not (box.x - 0.5 <= px <= box.x + box.w + 0.5):
        return
    cv.line([px, px], [box.y - 4, box.y + box.h], ACCENT, 1.6, cap="butt", z=8)
    if label:
        right = px > box.x + box.w - 50
        cv.text(px + (-5 if right else 5), box.y + 8, label, 11, ACCENT, weight="bold",
                anchor="end" if right else "start", halo=("#ffffff", 3.5), z=9)


def tau_color(tau: float) -> tuple:
    lo, top = TOY["tau_min"], TOY["axis_max"]
    return time_color(math.log(max(tau, lo) / lo) / math.log(top / lo))


def tokens_panel(cv: Canvas, ox, oy, W, H, sim, a) -> None:
    box = Box(ox + 46, oy + 34, W - 56, H - 74)
    panel_title(cv, ox, oy, 1, "Every token drops once")
    sx = Scale(0, TOY["axis_max"], box.x, box.x + box.w)
    sy = Scale(0, 3.2, box.y + box.h, box.y)
    clip = frame(cv, box, sx, sy, xt=linear_ticks(0, 100, 5), yt=linear_ticks(0, 3.2, 4),
                 x_label="training progress a", y_label="loss", y_off=32)
    burn = float(sx(TOY["burn_in"]))
    cv.rect(box.x, box.y, burn - box.x, box.h, "#f1f2f4", z=1.5)
    cv.text((box.x + burn) / 2 + 3.5, box.y + box.h - 6, "burn-in", 9.5, C["muted"], rotation=90,
            halo=("#f1f2f4", 3), z=4)
    for tau, loss in sim["shown"]:
        learned = tau <= a
        cv.line(sx(sim["x"]), sy(loss), tau_color(tau) if learned else "#c9ced4", 1.4 if learned else 1.1,
                alpha=0.95 if learned else 0.6, clip=clip, z=3)
    x, mean = sim["x"], sim["mean_loss"]
    future = np.where(x >= a - 0.5, mean, np.nan)
    past = np.where(x <= a + 1e-9, mean, np.nan)
    cv.line(sx(x), sy(future), C["faint"], 1.8, dash=(4, 3), clip=clip, z=4)
    cv.line(sx(x), sy(past), "#ffffff", 7, alpha=0.85, clip=clip, z=4)
    cv.line(sx(x), sy(past), C["ink"], 3, clip=clip, z=5)
    playhead(cv, box, sx, a, f"a = {round(a)}")


def spectrum_panel(cv: Canvas, ox, oy, W, H, sim, a, alpha) -> None:
    box = Box(ox + 46, oy + 34, W - 56, H - 74)
    panel_title(cv, ox, oy, 2, "Learning times thin out")
    bin_w = TOY["axis_max"] / TOY["hist_bins"]
    curve_x = np.linspace(TOY["tau_min"], TOY["axis_max"], 160)
    curve = tau_density(curve_x, alpha) * bin_w
    top = max(sim["hist"].max(), curve[4:].max()) * 1.12
    sx = Scale(0, TOY["axis_max"], box.x, box.x + box.w)
    sy = Scale(0, top, box.y + box.h, box.y)
    clip = frame(cv, box, sx, sy, xt=linear_ticks(0, 100, 5), yt=linear_ticks(0, top, 4),
                 x_label="learning time τ", y_label="share of tokens", y_off=36, fy=lambda v: f"{round(v * 100)}%")
    for j, share in enumerate(sim["hist"]):
        x0, x1 = j * bin_w, (j + 1) * bin_w
        color = tau_color(x0 + bin_w / 2)
        y, base = float(sy(share)), float(sy(0))
        left = float(sx(x0)) + 0.6
        cv.rect(left, y, float(sx(x1) - sx(x0)) - 1.2, base - y, color, alpha=0.2, z=2, clip=clip)
        learned_to = min(x1, a)
        if learned_to > x0:
            w = max(0.0, float(sx(learned_to) - sx(x0)) - (1.2 if learned_to == x1 else 0.6))
            cv.rect(left, y, w, base - y, color, alpha=0.95, z=2.5, clip=clip)
    cv.line(sx(curve_x), sy(curve), C["ink"], 1.4, dash=(4, 3), clip=clip, z=4)
    learned = f"{round(100 * np.mean(sim['tau'] <= a))}% learned by a = {round(a)}"
    cv.text(box.x + box.w - 4, box.y + 12, learned, 11, ACCENT, weight="bold", anchor="end", halo=("#ffffff", 3.5),
            z=7)
    # Same rule as toy.js: a near-flat spectrum puts the label in the top row, on the share learned.
    lx, ly = float(sx(curve_x[30])) + 6, float(sy(curve[30])) - 4
    if ly < box.y + 30:
        ly += 18
    cv.text(lx, ly, "p(τ) ∝ τ⁻ᵅ", 11, C["ink"], style="italic", halo=("#ffffff", 3.5), z=7)
    playhead(cv, box, sx, a)


def rate_panel(cv: Canvas, ox, oy, W, H, sim, a) -> None:
    box = Box(ox + 46, oy + 34, W - 56, H - 74)
    panel_title(cv, ox, oy, 3, "The loss follows the spectrum")
    lo, hi = TOY["burn_in"], TOY["axis_max"]
    sx = Scale(lo, hi, box.x, box.x + box.w, log=True)
    sy = Scale(1e-4, 1e-1, box.y + box.h, box.y, log=True)
    clip = frame(cv, box, sx, sy, xt=[10, 20, 50, 100], yt=[1e-4, 1e-3, 1e-2, 1e-1],
                 x_minor=[30, 40, 60, 70, 80, 90], x_label="training progress a (log)", y_label="−dL/da (log)",
                 y_off=36)
    spec_x, spec_y = sim["spec_x"], np.where(sim["spec_y"] > 0, sim["spec_y"], np.nan)
    cv.line(sx(spec_x), sy(spec_y), C["spec"], 1.4, alpha=0.8, clip=clip, z=3)
    for px, py in zip(sx(spec_x), sy(spec_y)):
        if np.isfinite(py):
            cv.dot(float(px), float(py), 6.5, C["spec"], z=3.5, clip=clip)
    cv.line(sx(sim["x_rate"]), sy(sim["rate"]), C["emp"], 2.6, clip=clip, z=4)

    def label(y, text, color, below):
        py = min(box.y + box.h - 4, max(box.y + 12, float(sy(y)) + (16 if below else -9)))
        cv.text(box.x + box.w - 4, py, text, 11, color, weight="bold", anchor="end", halo=("#ffffff", 3.5), z=7)

    label(sim["rate"][-1], f"−dL/da: slope {fmt_slope(sim['slope_rate'])}", C["emp"], False)
    last_spec = sim["spec_y"][sim["spec_y"] > 0]
    if last_spec.size:
        label(last_spec[-1], f"p(τ): slope {fmt_slope(sim['slope_spec'])}", C["spec"], True)
    if a >= lo:
        playhead(cv, box, sx, a)
        y = math.exp(np.interp(math.log(a), np.log(sim["x_rate"]), np.log(sim["rate"])))
        cv.dot(float(sx(a)), float(sy(y)), 9, C["emp"], ec="#ffffff", lw=1.6, z=9)


def legend(cv: Canvas, cx: float, top: float, width: float, items) -> None:
    """figcaption.legend: centred, wrapping items of a swatch and a label, 0.78rem at line-height 1.6."""
    size, line_h, gap = 13.26, 21.2, 14
    sized = []
    for kind, color, parts in items:
        sw = {"ramp": 40, "pt": 7}.get(kind, 18) + 6
        sized.append((kind, color, parts, sw + sum(cv.measure(s, size, weight=w, style=st) for s, w, st in parts)))
    rows, row = [], []
    for item in sized:
        if row and sum(it[3] for it in row) + gap * len(row) + item[3] > width:
            rows.append(row)
            row = []
        row.append(item)
    rows.append(row)
    for r, row in enumerate(rows):
        total = sum(it[3] for it in row) + gap * (len(row) - 1)
        x = cx - total / 2
        base = top + line_h * r + 15
        mid = base - 4.2
        for kind, color, parts, w in row:
            if kind == "ramp":
                for j in range(20):
                    cv.rect(x + 2 * j, mid - 3.5, 2.05, 7, time_color(j / 19), z=6)
            elif kind == "pt":
                cv.dot(x + 3.5, mid, 7, color, z=6)
            else:
                lw = {"thick": 3.5}.get(kind, 2)
                cv.line([x, x + 18], [mid, mid], color, lw, dash=(4, 2.5) if kind == "dash" else None, cap="butt", z=6)
            sw = {"ramp": 40, "pt": 7}.get(kind, 18) + 6
            text_color = color if kind != "ramp" else MUTED
            cv.rich(x + sw, base, parts, size, text_color)
            x += w + gap


def slider(cv: Canvas, x, y, w, frac) -> tuple[float, float]:
    """A range input with `accent-color`: filled track up to the thumb."""
    frac = min(1.0, max(0.0, frac))
    cv.rrect(x, y - 4, w, 8, 4, "#eceef1", ec="#c3c8cf", lw=0.8, z=3)
    tx = x + 8 + frac * (w - 16)
    cv.rrect(x, y - 4, tx - x, 8, 4, ACCENT, z=4)
    cv.dot(tx, y, 17, ACCENT, ec="#ffffff", lw=1.2, z=5)
    return tx, y


def control(cv: Canvas, x, w, label_parts, value, frac, out_text):
    cv.rich(x, 37, label_parts, 13.8, INK)
    out_w = 88
    knob = slider(cv, x, 54, w - out_w - 10, frac)
    cv.text(x + w, 58.5, out_text, 12, MUTED, family=MONO, anchor="end")
    return knob


CURSOR = [(0, 0), (0, 17), (4.2, 13), (7.2, 19.6), (10, 18.4), (7.2, 12), (12.4, 12)]

PREVIEW_W, PREVIEW_H = 960, 480


def preview_frame(sim, a: float, alpha: float, width: float, playing: bool, cursor_on: str | None) -> np.ndarray:
    W, H = PREVIEW_W, PREVIEW_H
    cv = Canvas(W, H, 1)
    cv.rect(0, 0, W, H, "#ffffff", z=-1)
    cv.gradient(Box(0, 0, W, H), "#fbfbff", "#f6f7f9")
    cv.line([0.5, W - 0.5, W - 0.5, 0.5, 0.5], [0.5, 0.5, H - 0.5, H - 0.5, 0.5], LINE, 1, cap="butt", z=1)

    play_w, play_x = 98, 20
    cv.ax.add_patch(FancyBboxPatch((play_x, 26), play_w, 38, boxstyle="round,pad=0,rounding_size=19", fc=ACCENT,
                                   ec="none", zorder=3, path_effects=[patheffects.SimplePatchShadow(
                                       offset=(0, -2), shadow_rgbFace=ACCENT, alpha=0.25), patheffects.Normal()]))
    cv.icon(ICONS["pause" if playing else "play"], play_x + 16, 37.5, 15, "#ffffff", fill=True)
    cv.text(play_x + 37, 45, "Pause" if playing else "Play", 15, "#ffffff", weight="bold", va="center_baseline")

    gap, inner = 22, W - 40
    fr = (inner - play_w - 3 * gap) / 3.25
    x_time = play_x + play_w + gap
    x_alpha = x_time + 1.25 * fr + gap
    x_width = x_alpha + fr + gap
    control(cv, x_time, 1.25 * fr, [("Training progress", "normal", "normal")], a, a / 100, f"a = {round(a)}")
    knob_alpha = control(cv, x_alpha, fr, [("When", "bold", "normal"), (" tokens learn: exponent ", "normal", "normal"),
                                           ("α", "normal", "italic")], alpha, alpha / 2, f"α = {alpha:.2f}")
    control(cv, x_width, fr, [("How abruptly", "bold", "normal"), (": step width", "normal", "normal")], width,
            (width - 1) / 14, f"width = {width:.1f}")

    col_w = (inner - 24) / 3
    chart_w, chart_h = col_w - 22, 220
    fig_top, fig_h = 82, 1 + 10 + chart_h + 4 + 2 * 21.2 + 6 + 1
    legends = [
        [("ramp", None, [("one token each, lit once learned", "normal", "normal")]),
         ("thick", C["ink"], [("average loss ", "normal", "normal"), ("L", "normal", "italic")])],
        [("ramp", None, [("share learned per interval", "normal", "normal")]),
         ("dash", C["ink"], [("p", "normal", "italic"), ("(", "normal", "normal"), ("τ", "normal", "italic"),
                             (") ∝ ", "normal", "normal"), ("τ", "normal", "italic"), ("⁻ᵅ", "normal", "normal")])],
        [("line", C["emp"], [("−d", "normal", "normal"), ("L", "normal", "italic"), ("/d", "normal", "normal"),
                             ("a", "normal", "italic")]),
         ("pt", C["spec"], [("p", "normal", "italic"), ("(", "normal", "normal"), ("τ", "normal", "italic"),
                            (") × mean drop", "normal", "normal")])],
    ]
    for k in range(3):
        fx = 20 + k * (col_w + 12)
        cv.rrect(fx + 0.5, fig_top + 0.5, col_w - 1, fig_h - 1, 12, "#ffffff", ec=LINE, lw=1, z=0.5)
        ox, oy = fx + 11, fig_top + 11
        if k == 0:
            tokens_panel(cv, ox, oy, chart_w, chart_h, sim, a)
        elif k == 1:
            spectrum_panel(cv, ox, oy, chart_w, chart_h, sim, a, alpha)
        else:
            rate_panel(cv, ox, oy, chart_w, chart_h, sim, a)
        legend(cv, fx + col_w / 2, oy + chart_h + 4, chart_w, legends[k])

    k = min(len(sim["x"]) - 1, round(a / TOY["axis_max"] * (len(sim["x"]) - 1)))
    head = f"Loss at a = {round(a)}: {sim['mean_loss'][k]:.2f}."
    if alpha < 0.1:
        lines = [f"{head} A flat spectrum gives a flat derivative: the loss falls linearly, with no power law."]
    else:
        lines = [f"{head} After burn-in, −dL/da falls with slope {fmt_slope(sim['slope_rate'])} on log-log axes "
                 f"and p(τ) with {fmt_slope(sim['slope_spec'])}:",
                 "the loss inherits the spectrum's power law, whatever the step width."]
    base = fig_top + fig_h + 34 if len(lines) == 2 else fig_top + fig_h + 46
    for j, line in enumerate(lines):
        cv.text(W / 2, base + 24 * j, line, 14.5, INK, anchor="middle")
    cv.rich(W / 2, H - 17, [("Preview of the interactive toy.  ", "normal", "normal"),
                            ("Open the project page to drag the sliders yourself  →", "bold", "normal")], 13.5, ACCENT,
            anchor="middle")

    if cursor_on == "alpha":
        tx, ty = knob_alpha
        cv.ax.add_patch(Polygon([(tx + 2 + px, ty + 3 + py) for px, py in CURSOR], closed=True, fc="#111111",
                                ec="#ffffff", lw=1.3, zorder=20, joinstyle="round"))
    img = cv.rgba()[..., :3]
    plt.close(cv.fig)
    return img


def save_gif(frames: list[np.ndarray], durations: list[int], path: Path, n_colors: int = 200) -> None:
    """One shared palette; pixels that did not change are left transparent, so a frame stores only what moved."""
    sample = np.concatenate(frames[:: max(1, len(frames) // 16)], axis=0)
    palette = Image.fromarray(sample).quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT,
                                                dither=Image.Dither.NONE)
    entries = palette.getpalette()[: 3 * n_colors]
    clear = len(entries) // 3
    full = entries + [255, 0, 255] + [0, 0, 0] * (255 - clear)
    indexed = [np.asarray(Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)) for f in frames]
    images, prev = [], None
    for idx in indexed:
        out = idx.copy()
        if prev is not None:
            out[idx == prev] = clear
        prev = idx
        im = Image.fromarray(out.astype(np.uint8), mode="P")
        im.putpalette(full)
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, disposal=1,
                   transparency=clear, optimize=False)


def ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def preview(path: Path) -> None:
    """Play from a = 0 to 100, then drag the spectrum exponent down to a flat spectrum, up to 2 and back."""
    tokens = draw_tokens(SEED)
    alpha0, width0 = 1.2, 4.0
    sim0 = simulate(tokens, alpha0, width0)
    frames, durations = [], []
    for j in range(46):
        a = 100 * j / 45
        frames.append(preview_frame(sim0, a, alpha0, width0, playing=j < 45, cursor_on=None))
        durations.append(1500 if j == 45 else 70)
    for start, end, n, hold in ((1.2, 0.0, 14, 1600), (0.0, 2.0, 20, 1000), (2.0, 1.2, 10, 1800)):
        for j in range(1, n + 1):
            alpha = round((start + (end - start) * ease(j / n)) * 20) / 20
            sim = simulate(tokens, alpha, width0)
            frames.append(preview_frame(sim, 100, alpha, width0, playing=False, cursor_on="alpha"))
            durations.append(hold if j == n else 70)
    save_gif(frames, durations, path)
    print(f"preview: {len(frames)} frames, {sum(durations) / 1000:.1f} s, "
          f"{path.stat().st_size / 1e6:.2f} MB -> {path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["banner", "buttons", "preview"])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "banner"):
        banner(OUT_DIR / "banner.png")
    if args.only in (None, "buttons"):
        buttons(OUT_DIR)
    if args.only in (None, "preview"):
        preview(OUT_DIR / "mechanism.gif")


if __name__ == "__main__":
    main()
