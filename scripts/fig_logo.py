"""Generate the CellPack logo and its animation as standalone SVGs.

Outputs:
  logo.svg                the static box mark (logo_box.py)
  cellpack-animation.svg  the mark being packed, played once and frozen on the
                          logo: cells rain into the open box and pile up, the
                          flaps fall shut and press the pile down, tape runs
                          over the seam and down the side, and the cell
                          sticker is laid on from the back corner forward

The cells come from a rigid-sphere simulation (cell_sim.py), rendered as
cel-shaded spheres: each one's path, spin and depth order follow the physics.
Every animated attribute is a keyframe track on one global timeline, so the
same tracks render as SMIL or as a still at any moment (--frames) for preview.

Usage: python scripts/fig_logo.py [--out-dir assets] [--palette teal|kraft]
                                  [--frames DIR] [--times 0.5,2,4]
                                  [--search 0-9]   # score simulation seeds
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from cell_sim import FPS, G, Recording, quat_rotate, score, simulate
from logo_box import (BAND, HEX_R, N, PALETTES, SIZE, STICKER_R, TAPE_DROP, TAPE_W, build_logo,
                      corrugation, corrugation_side, face_matrix, face_right, flap_shade, ground_shadow, hexagon,
                      left_base, mix, sticker, tape_drop, top_base)

ROOT = Path(__file__).resolve().parents[1]

# The open flaps stand above the box, so the stage reaches higher than the logo.
VIEW_TOP = -88.0
VIEW = (0.0, VIEW_TOP, float(SIZE), SIZE - VIEW_TOP)

# Timeline in seconds.
END = 6.65
# Cells fall as a Poisson process whose rate follows a logistic curve over
# RAIN: a long thin tail at the start (a lone cell or two), then a short climb
# to a downpour. Where they fall follows the same curve, opening out from the
# opening's screen-vertical diagonal (back corner to front corner, where a cell
# shows best) to the whole opening.
RAIN = (0.1, 2.6)
RAIN_MID, RAIN_WIDTH = 1.35, 0.12  # logistic centre and width, seconds
RAIN_FLOOR = 0.006        # the long tail: a trickle relative to the full downpour
AIM_START = 0.1           # spread around the diagonal at first, as a fraction of the opening
FLAP_A = (3.3, 3.8)       # back flap closes
FLAP_B = (3.55, 4.05)     # front flap closes on top of it
CLOSED = FLAP_B[1]
SHADE_IN = (FLAP_B[1] - 0.25, CLOSED)  # the flaps' shade on the left face grows in as they settle
TAPE_TOP = (4.15, 4.6)    # tape runs along the seam, back to front
TAPE_DOWN = (4.6, 4.85)   # and down the left face
STICK_IN = (4.85, 5.3)    # sticker is carried in from off stage and set down on the back edge
STICK_LAY = (5.3, 6.4)    # then smoothed down from the back edge forward

FLAP_OPEN_A, FLAP_OPEN_B = 118.0, 112.0  # open angles from closed, degrees
RIM_SHADE = 0.05          # depth of the shade the front walls cast inside, box units
BOARD = BAND / N          # board thickness in box units, matching the logo's face units
STICKER_LIFT = 80.0       # how far the free end of the sticker curls up, degrees
STICKER_CURL = 60.0       # flat units over which the curl builds up from the lay line
STICKER_STRIPS = 12       # bands the curled sheet is drawn in
STICKER_FROM = (230.0, -270.0, 14.0)  # carried in from this screen offset and tilt (degrees)
CARRY_IN = 2.0            # entry speed of the carry, relative to its average
LAY_IN = 1.4              # starting speed of the smoothing, relative to its average
STICKER_BACK = ("#F4F1F8", "#D8D1E6")  # backing paper and its edge

N_CELLS = 360             # enough to heap over the rim before the flaps press it in
CELL_R = 0.075            # median cell radius, box units
SEED = 9
CELL_COLORS = (  # (cytoplasm, membrane), purple-weighted to match the sticker
    ("#EEE8FF", "#7B5CD6"), ("#EEE8FF", "#7B5CD6"), ("#F1EAFF", "#9370E0"),
    ("#FFE6EF", "#D9628E"), ("#FFEBDD", "#E08A55"), ("#E4F0FF", "#4F7FD9"),
)
SQUASH = 0.12  # most a cell flattens on impact: rubbery, not jelly
SQUASH_FULL = 0.2  # impact (box units per frame^2, gravity removed) that reaches SQUASH
KEY_TOL = 1.0  # px; keyframes are dropped where linear interpolation stays this close
SPIN_TOL = 1.6  # the same for the nucleus, which only hints at the spin
HOLD_MIN = 0.2  # s; motion that stays within tolerance this long is held still
SQUASH_MIN = 1 / 60  # flattening that moves a cell's outline well under a pixel is left out

Value = tuple[float, ...] | str


# --------------------------------------------------------------- tracks

class Track:
    """Keyframes for one attribute over [0, END]. Tuples interpolate linearly;
    strings switch discretely. live is when the element is rendered at all:
    outside it the value is free."""

    def __init__(self, attr: str, keys: list[tuple[float, Value]], fmt=None,
                 transform: str | None = None, live: tuple[float, float] = (0.0, END)):
        keys = sorted(keys, key=lambda k: k[0])
        if keys[0][0] > 0:
            keys.insert(0, (0.0, keys[0][1]))
        if keys[-1][0] < END:
            keys.append((END, keys[-1][1]))
        self.attr, self.keys, self.transform, self.live = attr, keys, transform, live
        self.fmt = fmt or (lambda v: v if isinstance(v, str) else " ".join(f"{x:.1f}" for x in v))
        self.discrete = isinstance(keys[0][1], str)

    def at(self, t: float) -> str:
        keys = self.keys
        if t <= keys[0][0]:
            return self.fmt(keys[0][1])
        for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
            if t < t1 or (t == t1 and not self.discrete):
                if self.discrete or t1 == t0:
                    return self.fmt(v0)
                assert isinstance(v0, tuple) and isinstance(v1, tuple)
                u = (t - t0) / (t1 - t0)
                return self.fmt(tuple(a + (b - a) * u for a, b in zip(v0, v1)))
        return self.fmt(keys[-1][1])

    def _attr(self, v: str) -> str:
        return f'transform="{self.transform}({v})"' if self.transform else f'{self.attr}="{v}"'

    def static_attr(self, t: float) -> str:
        return self._attr(self.at(t))

    def base_attr(self) -> str:
        """The plain attribute the SMIL animations sit on."""
        return self._attr((self._switches() if self.discrete else self._windows())[0])

    def _windows(self) -> tuple[str, list[list[tuple[float, str]]]]:
        """(base value, keyframe runs to animate), all within live. The base is
        the value held longest while live; only the stretches that differ from
        it are animated."""
        lo, hi = self.live
        keys = ([(lo, self.at(lo))] + [(k, self.fmt(v)) for k, v in self.keys if lo < k < hi]
                + [(hi, self.at(hi))])
        held: dict[str, float] = {}
        for (k0, v0), (k1, v1) in zip(keys, keys[1:]):
            if v0 == v1:
                held[v0] = held.get(v0, 0.0) + k1 - k0
        base = max(held, key=held.__getitem__) if held else keys[-1][1]
        runs, start = [], None
        for k in range(len(keys) - 1):
            still = keys[k][1] == base == keys[k + 1][1]
            if not still and start is None:
                start = k
            elif still and start is not None:
                runs.append(keys[start:k + 1])
                start = None
        if start is not None:
            runs.append(keys[start:])
        return base, runs

    def smil(self) -> str:
        """SMIL over the base attribute. An animation is only active while the
        value differs from the base, and is removed afterwards: a browser keeps
        re-applying every active or frozen animation on each frame, so this
        leaves a phone only the few tracks moving at that moment."""
        if self.discrete:
            return self._sets()
        out = []
        for run in self._windows()[1]:
            t0, t1 = run[0][0], run[-1][0]
            if t1 <= t0:
                continue
            values = ";".join(v for _, v in run)
            times = ";".join(f"{(k - t0) / (t1 - t0):.4f}" for k, _ in run)
            # A span ending before END returns to the base value; one reaching it holds the last frame.
            fill = "freeze" if t1 >= END else "remove"
            span = f'begin="{t0:.4f}s" dur="{t1 - t0:.4f}s" fill="{fill}"'
            if self.transform:
                out.append(f'<animateTransform attributeName="transform" type="{self.transform}" '
                           f'values="{values}" keyTimes="{times}" {span}/>')
            else:
                out.append(f'<animate attributeName="{self.attr}" values="{values}" keyTimes="{times}" {span}/>')
        return "".join(out)

    def _switches(self) -> tuple[str, list[tuple[float, float, str]]]:
        """(base value, (start, end, value) holds that differ from it), within live."""
        lo, hi = self.live
        holds = []
        for (k0, v), (k1, _) in zip(self.keys, self.keys[1:] + [(END, None)]):
            a, b = max(k0, lo), min(k1, hi)
            if b > a:
                v = self.fmt(v)
                if holds and holds[-1][2] == v:
                    holds[-1][1] = b
                else:
                    holds.append([a, b, v])
        if not holds:
            return self.at(lo), []
        held: dict[str, float] = {}
        for a, b, v in holds:
            held[v] = held.get(v, 0.0) + b - a
        base = max(held, key=held.__getitem__)
        return base, [(a, b, v) for a, b, v in holds if v != base]

    def _sets(self) -> str:
        """Discrete switches as <set> windows, each active only while the
        value differs from the base."""
        out = []
        for a, b, v in self._switches()[1]:
            tail = ' fill="freeze"' if b >= END else ""
            out.append(f'<set attributeName="{self.attr}" to="{v}" begin="{a:.4f}s" dur="{b - a:.4f}s"{tail}/>')
        return "".join(out)


def node(tag: str, attrs: str, tracks: list[Track], body: str = "", t: float | None = None) -> str:
    """An element whose tracks become SMIL children over their base values,
    or attributes frozen at time t."""
    plain = " ".join(tr.base_attr() if t is None else tr.static_attr(t) for tr in tracks)
    attrs = " ".join(a for a in (attrs, plain) if a)
    attrs = f" {attrs}" if attrs else ""
    anim = "" if t is not None else "".join(tr.smil() for tr in tracks)
    return f"<{tag}{attrs}>{anim}{body}</{tag}>"


def shown(t0: float, t1: float = math.inf, live: tuple[float, float] = (0.0, END)) -> Track:
    """Rendered from t0 until t1. Outside that window the element is
    display:none, which takes it out of rendering altogether. live is when
    the parent itself is rendered."""
    keys: list[tuple[float, Value]] = [(0.0, "none" if t0 > 0 else "inline")]
    if t0 > 0:
        keys.append((t0, "inline"))
    if t1 <= END:
        keys.append((t1, "none"))
    return Track("display", keys, live=live)


def ease_in(u: float) -> float:
    return u * u


def ease_out(u: float) -> float:
    return 1 - (1 - u) ** 2


def ease_in_out(u: float) -> float:
    return 3 * u * u - 2 * u ** 3


def ramp(t0: float, t1: float, v0: float, v1: float, ease=ease_out,
         steps: int = 6) -> list[tuple[float, Value]]:
    return [(t0 + (t1 - t0) * i / steps, (v0 + (v1 - v0) * ease(i / steps),))
            for i in range(steps + 1)]


def hermite(t0: float, t1: float, v0: float, v1: float, m0: float, m1: float,
            steps: int = 12) -> list[tuple[float, Value]]:
    """v0 -> v1 over [t0, t1] along a cubic with end slopes m0 and m1, given as
    multiples of the average speed: 1 is the average, 0 comes to rest."""
    out: list[tuple[float, Value]] = []
    for i in range(steps + 1):
        u = i / steps
        f = (3 * u ** 2 - 2 * u ** 3) + (u ** 3 - 2 * u ** 2 + u) * m0 + (u ** 3 - u ** 2) * m1
        out.append((t0 + (t1 - t0) * u, (v0 + (v1 - v0) * f,)))
    return out


def thin_keys(times: np.ndarray, vals: np.ndarray, tol: float = KEY_TOL) -> list[int]:
    """Indices of the samples to keep so that linear interpolation between them
    stays within tol of every dropped sample (Douglas-Peucker over time)."""
    keep = {0, len(times) - 1}
    stack = [(0, len(times) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        u = ((times[i + 1:j] - times[i]) / (times[j] - times[i]))[:, None]
        err = np.abs(vals[i + 1:j] - (vals[i] + (vals[j] - vals[i]) * u)).max(axis=1)
        k = int(err.argmax())
        if err[k] > tol:
            keep.add(i + 1 + k)
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(keep)


def hold_still(vals: np.ndarray, tol: float, min_len: int) -> np.ndarray:
    """Pin each stretch where a sampled signal stays within tol in every
    component for at least min_len samples to its midpoint. A settling pile
    creeps by fractions of a pixel for seconds; held exactly, those stretches
    need no animation at all."""
    out = vals.copy()
    i, n = 0, len(vals)
    while i < n:
        lo, hi, j = vals[i].copy(), vals[i].copy(), i + 1
        while j < n:
            lo2, hi2 = np.minimum(lo, vals[j]), np.maximum(hi, vals[j])
            if (hi2 - lo2).max() > tol:
                break
            lo, hi, j = lo2, hi2, j + 1
        if j - i >= min_len:
            out[i:j] = (lo + hi) / 2
            i = j
        else:
            i += 1
    return out


# --------------------------------------------------------------- box space
#
# Unit box coordinates on the logo's hexagon: x runs LL -> B, y runs LL -> O
# (toward the back), z runs LL -> UL (up). The hexagon is an orthographic view
# along (1, -1, 1) scaled by SPHERE_PX, so a sphere of radius r is a circle of
# radius r * SPHERE_PX and "closer to the viewer" means larger x - y + z.

HEXA = hexagon()
_LL = np.array(HEXA["LL"])
EX, EY, EZ = (np.array(HEXA[k]) - _LL for k in ("B", "O", "UL"))
SPHERE_PX = HEX_R / math.sqrt(2 / 3)
VIEW_DIR = np.array([1.0, -1.0, 1.0]) / math.sqrt(3)


def iso(x: float, y: float, z: float) -> tuple[float, float]:
    px = _LL + x * EX + y * EY + z * EZ
    return float(px[0]), float(px[1])


def iso_np(p: np.ndarray) -> np.ndarray:
    """(..., 3) box points -> (..., 2) screen points."""
    return _LL + p[..., :1] * EX + p[..., 1:2] * EY + p[..., 2:3] * EZ


def poly(pts) -> str:
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + "Z"


def fmt_quad(v: Value) -> str:
    return poly([(v[i], v[i + 1]) for i in range(0, 8, 2)])  # type: ignore[index]


def spawn_z(x: float, y: float, r: float) -> float:
    """Release height that puts a sphere of radius r just above the stage."""
    return (iso(x, y, 0)[1] - VIEW_TOP + r * SPHERE_PX + 6) / HEX_R


# --------------------------------------------------------------- flaps

FLAPS = {  # hinge x, outward sign (-1 opens toward -x), open angle, closing window, colour key
    "A": (0.0, -1.0, FLAP_OPEN_A, FLAP_A, "top"),
    "B": (1.0, 1.0, FLAP_OPEN_B, FLAP_B, "flap"),
}


def flap_angle(which: str, t: float) -> float:
    _, _, open_deg, (t0, t1), _ = FLAPS[which]
    u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return open_deg * (1 - ease_in(u))  # falls shut, gathering speed


def flap_ceiling(t: float, x: np.ndarray) -> np.ndarray | None:
    """Highest point a sphere at x may reach under the closing flaps."""
    lid = np.full(len(x), np.inf)
    for which, (hinge, _, _, _, _) in FLAPS.items():
        theta = flap_angle(which, t)
        if theta >= 85:
            continue
        reach = np.abs(x - hinge)
        under = reach <= 0.5
        lid[under] = np.minimum(lid[under], 1 + reach[under] * math.tan(math.radians(theta)))
    return None if np.isinf(lid).all() else lid


def flap_geometry(which: str, theta: float) -> dict:
    """Screen geometry of a flap opened theta degrees: the visible board
    surface, its front cut edge (the y=0 side, always toward the viewer) and
    its free edge. The flutes run from the hinge to the free edge, so the free
    edge shows the wave and the front edge the flutes side-on. Thickness fades
    out as the flap settles flat, where the logo's closed top takes over."""
    hinge, outward, _, _, _ = FLAPS[which]
    th = math.radians(theta)
    d = np.array([-outward * math.cos(th), 0.0, math.sin(th)])   # hinge -> free edge
    nrm = np.array([outward * math.sin(th), 0.0, math.cos(th)])  # outside surface normal
    fade = min(1.0, theta / 25.0)
    tau = BOARD * max(0.08, fade)  # never quite zero, so the edge maps stay invertible
    h0, h1 = np.array([hinge, 0.0, 1.0]), np.array([hinge, 1.0, 1.0])
    f0, f1 = h0 + 0.5 * d, h1 + 0.5 * d
    outside = float(nrm @ VIEW_DIR) > 0
    s = nrm * (0 if outside else -tau)  # the surface we see
    inner = -tau * nrm

    def quad(*pts):
        return tuple(float(c) for p in pts for c in iso_np(p))

    # Cut edges as maps from the corrugation strip (length along, BAND across).
    return dict(face=quad(h0 + s, h1 + s, f1 + s, f0 + s),
                front=strip_map(h0, d / N, inner / BAND),
                free=strip_map(f0, np.array([0.0, 1.0, 0.0]) / N, inner / BAND),
                outside=outside)


def lin(v: np.ndarray) -> np.ndarray:
    """Screen offset of a box-space vector."""
    return v[0] * EX + v[1] * EY + v[2] * EZ


def strip_map(origin: np.ndarray, along: np.ndarray, across: np.ndarray) -> tuple[float, ...]:
    """Affine map (a, b, c, d, e, f) sending strip units to the screen."""
    o, a, c = iso_np(origin), lin(along), lin(across)
    return (float(a[0]), float(a[1]), float(c[0]), float(c[1]), float(o[0]), float(o[1]))


def affine_group(keys: list[tuple[float, tuple[float, ...]]], body: str, t: float | None,
                 live: tuple[float, float] = (0.0, END)) -> str:
    """Animate an affine map SMIL cannot interpolate directly: split each
    keyframe into translate . rotate . skewX . scale and animate those."""
    parts: dict[str, list[tuple[float, Value]]] = {k: [] for k in ("translate", "rotate", "skewX", "scale")}
    prev = None
    for tk, (a, b, c, d, e, f) in keys:
        phi = math.degrees(math.atan2(b, a))
        if prev is not None:
            phi += 360 * round((prev - phi) / 360)
        prev = phi
        cp, sp = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        c2, d2 = cp * c + sp * d, -sp * c + cp * d
        parts["translate"].append((tk, (e, f)))
        parts["rotate"].append((tk, (phi,)))
        parts["skewX"].append((tk, (math.degrees(math.atan(c2 / d2)),)))
        parts["scale"].append((tk, (math.hypot(a, b), d2)))
    fmts = {"translate": None, "rotate": lambda v: f"{v[0]:.2f}", "skewX": lambda v: f"{v[0]:.2f}",
            "scale": lambda v: f"{v[0]:.5f} {v[1]:.5f}"}
    for kind in ("scale", "skewX", "rotate", "translate"):
        body = node("g", "", [Track("transform", parts[kind], fmts[kind], transform=kind, live=live)], body, t)
    return body


def flap_layer(pal: dict, which: str, live: tuple[float, float], t: float | None) -> str:
    _, _, _, (t0, t1), color = FLAPS[which]
    steps = 10
    keys: dict[str, list] = {k: [] for k in ("face", "front", "free", "fill")}
    for i in range(steps + 1):
        tk = t0 + (t1 - t0) * i / steps
        g = flap_geometry(which, flap_angle(which, tk))
        for k in ("face", "front", "free"):
            keys[k].append((tk, g[k]))
        keys["fill"].append((tk, pal[color] if g["outside"] else pal["flap_in"]))
    body = (affine_group(keys["free"], f'<rect width="{N}" height="{BAND}" fill="url(#corr)"/>', t, live)
            + node("path", "", [Track("d", keys["face"], fmt_quad, live=live),
                                Track("fill", keys["fill"], live=live)], t=t)
            + affine_group(keys["front"], f'<rect width="{N / 2}" height="{BAND}" fill="url(#flutes)"/>', t, live))
    return node("g", "", [shown(*live)], body, t)


def rims(which: str) -> str:
    """Corrugated cut edges along the tops of the y=0 and y=1 walls, seen while
    the box is open (the x=0 and x=1 tops are folds into the flaps). The back
    one sits behind the cells, the front one in front of them."""
    origin, inward = {"back": ((0, 1, 1), (0, -1, 0)), "front": ((0, 0, 1), (0, 1, 0))}[which]
    o = iso(*origin)
    ca = (EX / N)
    cb = (np.array(inward[0]) * EX + inward[1] * EY + inward[2] * EZ) / N
    return (f'<g transform="matrix({ca[0]:.6f},{ca[1]:.6f},{cb[0]:.6f},{cb[1]:.6f},{o[0]:.3f},{o[1]:.3f})">'
            f'<rect width="{N}" height="{BAND}" fill="url(#corr)"/></g>')


def rim_shade() -> str:
    """Narrow shade the front walls cast just inside the opening, over the
    cells and the inner walls: it reads as the board having some body."""
    out = []
    for name, (a, b, inward) in {"l": ((0, 0, 1), (1, 0, 1), (0, RIM_SHADE, 0)),
                                 "r": ((1, 0, 1), (1, 1, 1), (-RIM_SHADE, 0, 0))}.items():
        a, b, inward = np.array(a, float), np.array(b, float), np.array(inward)
        pa, pin = iso_np(a), iso_np(a + inward)
        pts = [iso_np(a), iso_np(b), iso_np(b + inward), iso_np(a + inward)]
        out.append(f'<linearGradient id="rim{name}" gradientUnits="userSpaceOnUse" '
                   f'x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pin[0]:.1f}" y2="{pin[1]:.1f}">'
                   f'<stop offset="0" stop-color="#000" stop-opacity=".28"/>'
                   f'<stop offset="1" stop-color="#000" stop-opacity="0"/></linearGradient>'
                   f'<path d="{poly(pts)}" fill="url(#rim{name})"/>')
    return "".join(out)


# --------------------------------------------------------------- cells

def cell_defs() -> str:
    """Cel-shaded sphere fills: lit from the upper left, a hard terminator and a
    shaded crescent toward the lower right."""
    out = []
    for k, (cyto, line) in enumerate(CELL_COLORS):
        shade = mix(cyto, line, 0.28)
        out.append(f'<radialGradient id="cb{k}" cx=".44" cy=".4" r=".62" fx=".3" fy=".26">'
                   f'<stop offset="0" stop-color="{mix(cyto, "#FFFFFF", 0.7)}"/>'
                   f'<stop offset=".56" stop-color="{cyto}"/>'
                   f'<stop offset=".57" stop-color="{shade}"/>'
                   f'<stop offset="1" stop-color="{mix(cyto, line, 0.42)}"/></radialGradient>'
                   f'<radialGradient id="cn{k}" cx=".4" cy=".36" r=".64">'
                   f'<stop offset="0" stop-color="{mix(line, "#FFFFFF", 0.35)}"/>'
                   f'<stop offset=".6" stop-color="{line}"/>'
                   f'<stop offset="1" stop-color="{mix(line, "#000000", 0.2)}"/></radialGradient>')
    return "".join(out)


def rain_rate(t: np.ndarray) -> np.ndarray:
    """Relative rate of falling cells: a trickle, then a short climb to full."""
    return RAIN_FLOOR + (1 - RAIN_FLOOR) / (1 + np.exp(-(t - RAIN_MID) / RAIN_WIDTH))


def release_times(rng: np.random.Generator, n: int) -> np.ndarray:
    """n release times drawn from the rain rate (a Poisson process given n)."""
    grid = np.linspace(*RAIN, 2001)
    cdf = np.cumsum(rain_rate(grid))
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp(rng.random(n), cdf, grid)


def aim_points(rng: np.random.Generator, release: np.ndarray, r: np.ndarray):
    """Rim crossing points: along the diagonal x + y = 1 and across it, inside a
    window that widens with the rain rate."""
    spread = AIM_START + (1 - AIM_START) * rain_rate(release)
    tx, ty = np.empty(len(r)), np.empty(len(r))
    for i, (s_i, r_i) in enumerate(zip(spread, r)):
        lo, hi = r_i + 0.02, 1 - r_i - 0.02
        while True:
            d = rng.uniform(-1, 1) * max(s_i, 0.5)  # along the diagonal (x - y)
            c = rng.uniform(-1, 1) * s_i            # across it (x + y - 1)
            x, y = 0.5 + (d + c) / 2, 0.5 + (c - d) / 2
            if lo <= x <= hi and lo <= y <= hi:
                tx[i], ty[i] = x, y
                break
    return tx, ty


def run_sim(seed: int = SEED) -> Recording:
    return simulate(N_CELLS, seed, CLOSED + 0.05, release_times, spawn_z, aim=aim_points,
                    ceiling=flap_ceiling,
                    radius_median=CELL_R, spin=4.0)


def impact_squash(rec: Recording) -> tuple[np.ndarray, np.ndarray]:
    """Per frame and cell: how much it is flattened (0..SQUASH) and the screen
    angle, in degrees, of the push that flattens it. Impacts are kicks in the
    recorded motion beyond gravity; each one springs back over a few frames."""
    acc = np.zeros_like(rec.pos)
    acc[1:-1] = rec.pos[2:] - 2 * rec.pos[1:-1] + rec.pos[:-2]
    acc[1:-1, :, 2] += G / FPS ** 2  # take gravity back out
    acc[rec.times[:, None].repeat(len(rec.radius), 1) < rec.release[None, :] + 0.05] = 0
    kick = np.linalg.norm(acc, axis=2)
    q = SQUASH * np.clip(kick / SQUASH_FULL, 0, 1)
    for k in range(1, len(q)):
        q[k] = np.maximum(q[k], 0.5 * q[k - 1])
    q[q < 0.004] = 0
    scr = acc[..., :1] * EX + acc[..., 1:2] * EY + acc[..., 2:3] * EZ
    ang = np.degrees(np.arctan2(scr[..., 1], scr[..., 0]))
    out = np.zeros_like(ang)
    for k in range(1, len(ang)):  # hold the angle between impacts, keep it continuous mod 180
        prev = out[k - 1]
        cand = ang[k] + 180 * np.round((prev - ang[k]) / 180)
        out[k] = np.where(kick[k] > SQUASH_FULL * 0.1, cand, prev)
    return q, out


def cell_layers(rec: Recording, t: float | None) -> tuple[str, str]:
    """(settled cells, flying cells). Each cell flies until it first touches
    something, then rests; the resting copies are stacked by final depth and
    all flying copies are drawn over them."""
    rng = np.random.default_rng(SEED + 1)
    n = len(rec.radius)
    times = rec.times
    screen = iso_np(rec.pos)                                    # (f, n, 2)
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    nuc = quat_rotate(rec.quat, np.broadcast_to(axis, rec.quat.shape[:-1] + (3,)))  # (f, n, 3)
    nuc_px = (nuc[..., :1] * EX + nuc[..., 1:2] * EY + nuc[..., 2:3] * EZ) / SPHERE_PX  # per px of radius
    nuc_near = nuc @ VIEW_DIR                                   # -1 (behind) .. 1 (in front)
    colors = rng.integers(0, len(CELL_COLORS), n)
    depth = rec.pos[..., 0] - rec.pos[..., 1] + rec.pos[..., 2]
    squash, tilt = impact_squash(rec)

    def instance(i: int, lo: int, hi: int, live: tuple[float, float]) -> str:
        """Cell i over frames lo..hi, rendered only while live."""
        r = rec.radius[i] * SPHERE_PX
        sl = slice(lo, hi + 1)
        tt = times[sl]
        hold = round(HOLD_MIN * FPS)
        pos = hold_still(screen[sl, i], KEY_TOL, hold)
        nuc = np.concatenate([nuc_px[sl, i] * 0.42 * r,  # nucleus centre sits 0.42 r from the cell centre
                              (0.3 + 0.35 * (nuc_near[sl, i, None] + 1)) * 8], axis=1)
        nuc = hold_still(nuc, SPIN_TOL, hold)
        npx, op = nuc[:, :2], nuc[:, 2] / 8
        keep_p = thin_keys(tt, pos)
        keep_n = thin_keys(tt, nuc, SPIN_TOL)
        move = Track("transform", [(float(tt[k]), tuple(pos[k])) for k in keep_p], transform="translate",
                     live=live)
        spin = Track("transform", [(float(tt[k]), tuple(npx[k])) for k in keep_n], transform="translate",
                     live=live)
        fade = Track("opacity", [(float(tt[k]), (float(op[k]),)) for k in keep_n],
                     fmt=lambda v: f"{v[0]:.2f}", live=live)
        c = int(colors[i])
        line = CELL_COLORS[c][1]
        body = (f'<circle cx="{0.14 * r:.1f}" cy="{0.2 * r:.1f}" r="{r:.1f}" fill="#0B2E27" fill-opacity=".16"/>'
                f'<circle r="{r:.1f}" fill="url(#cb{c})" stroke="{mix(line, "#000000", 0.1)}" '
                f'stroke-width="{max(1.1, 0.065 * r):.2f}"/>'
                + node("circle", f'r="{0.4 * r:.1f}" fill="url(#cn{c})"', [spin, fade], t=t)
                + f'<ellipse cx="{-0.36 * r:.1f}" cy="{-0.42 * r:.1f}" rx="{0.24 * r:.1f}" '
                f'ry="{0.13 * r:.1f}" transform="rotate(-35 {-0.36 * r:.1f} {-0.42 * r:.1f})" '
                f'fill="#FFFFFF" fill-opacity=".8"/>')
        q, a = squash[sl, i].copy(), tilt[sl, i]
        q[q < SQUASH_MIN] = 0
        if q.max() > 0:  # flatten along the impact, a little wider across it
            # Unflattened, the angle is invisible: hold the coming impact's angle until it lands.
            hit = np.flatnonzero(q)
            a = np.where(q > 0, a, a[hit[np.minimum(np.searchsorted(hit, np.arange(len(q))), len(hit) - 1)]])
            keep_q = thin_keys(tt, np.stack([q * 60 * np.cos(np.radians(2 * a)),
                                             q * 60 * np.sin(np.radians(2 * a))], axis=1), 1.0)
            turn = [(float(tt[k]), (float(a[k]),)) for k in keep_q]
            body = node("g", "", [Track("transform", turn, transform="rotate", live=live)],
                        node("g", "", [Track("transform", [(float(tt[k]), (1 - float(q[k]), 1 + 0.5 * float(q[k])))
                                                           for k in keep_q], fmt=lambda v: f"{v[0]:.3f} {v[1]:.3f}",
                                             transform="scale", live=live)],
                             node("g", "", [Track("transform", [(k, (-v[0],)) for k, v in turn],  # type: ignore[index]
                                                  transform="rotate", live=live)], body, t), t), t)
        # The open box drops every cell once sealed, so after CLOSED the display is free.
        return node("g", "", [shown(*live, live=(0.0, CLOSED)), move], body, t)

    last = len(times) - 1
    first_touch = [min(last, int(np.searchsorted(times, rec.contact[i]))) for i in range(n)]
    resting = sorted(range(n), key=lambda i: depth[-1, i])
    flying = sorted(range(n), key=lambda i: depth[first_touch[i], i])
    rest_svg = "".join(instance(i, first_touch[i], last, (float(times[first_touch[i]]), CLOSED))
                       for i in resting if first_touch[i] < last)
    # Before release a cell waits off stage; it is only rendered while in flight.
    fly_svg = "".join(instance(i, 0, first_touch[i], (float(rec.release[i]), float(times[first_touch[i]])))
                      for i in flying)
    return rest_svg, fly_svg


# --------------------------------------------------------------- scene

def interior(pal: dict) -> str:
    """The open box's inside: the back walls x=0 and y=1 through the opening."""
    T, UR, O, UL = (HEXA[k] for k in ("T", "UR", "O", "UL"))
    return (f'<defs><linearGradient id="deep" gradientUnits="userSpaceOnUse" x1="0" y1="{T[1]:.1f}" '
            f'x2="0" y2="{O[1]:.1f}"><stop offset="0" stop-color="#000" stop-opacity="0"/>'
            f'<stop offset="1" stop-color="#000" stop-opacity=".35"/></linearGradient></defs>'
            f'<path d="{poly([UL, T, O])}" fill="{pal["wall_x"]}"/>'
            f'<path d="{poly([T, UR, O])}" fill="{pal["wall_y"]}"/>'
            f'<path d="{poly([UL, T, UR, O])}" fill="url(#deep)"/>')


def face_group(name: str, body: str) -> str:
    return f'<g transform="{face_matrix(HEXA, name)}">{body}</g>'


def sticker_layers(t: float | None) -> tuple[str, str]:
    """(on the top face, lifted above it). The sticker is carried in from off
    stage, set down on its back edge (flat b small, screen upper right) and
    smoothed forward: behind the lay line it is stuck flat; ahead of it the
    sheet curls up smoothly, its angle rising from 0 at the line toward
    STICKER_LIFT. The curl is drawn as STICKER_STRIPS bands, each a plane at its
    own angle, rolling forward with the line."""
    h = N / 2
    b0, b1 = h - STICKER_R, h + STICKER_R + 24  # includes the sticker's own shadow
    # The hand never stops: the carry lands still moving at the speed the
    # smoothing starts with, and only the smoothing comes to rest.
    lay = [(STICK_IN[0], (b0,))] + hermite(*STICK_LAY, b0, b1, LAY_IN, 0.0)
    near = Track("opacity", ramp(*STICK_IN, 0.0, 1.0, steps=6), fmt=lambda v: f"{v[0]:.2f}")
    big = 4 * N

    def band_clip(cid: str, start: float, length: float) -> str:
        """Rows from start to start + length past the lay line (moving with it)."""
        y = Track("y", [(k, (v[0] + start,)) for k, v in lay])  # type: ignore[index]
        return (f'<clipPath id="{cid}">'
                + node("rect", f'x="{-big}" width="{2 * big}" height="{length}"', [y], t=t) + "</clipPath>")

    stuck = Track("height", [(k, (v[0] + big,)) for k, v in lay])  # type: ignore[index]
    on_face = (f'<clipPath id="stuck">' + node("rect", f'x="{-big}" y="{-big}" width="{2 * big}"', [stuck], t=t)
               + "</clipPath>" + band_clip("loose", 0, big)
               + f'<g clip-path="url(#stuck)">{sticker()}</g>'
               + node("g", 'clip-path="url(#loose)"', [near],
                      f'<circle cx="{h}" cy="{h}" r="{STICKER_R}" fill="#000" fill-opacity=".13"/>', t))

    # A point at distance u past the lay line s, inside band i (angle phi_i,
    # starting at u_i where the curve has reached X_i along b and Z_i up), lands at
    #   T + a col_a + (s + X_i + (u - u_i) cos) col_b + (Z_i + (u - u_i) sin) up,
    # an affine map of (a, b) whose only dependence on s is a shift s * w_i.
    T, UR, UL = (np.array(HEXA[k]) for k in ("T", "UR", "UL"))
    col_a, col_b = (UR - T) / N, (UL - T) / N
    up = np.array([0.0, -HEX_R / N])  # one flat unit of height on screen
    lift, curl = math.radians(STICKER_LIFT), STICKER_CURL
    length = b1 - b0
    edges = [length * (k / STICKER_STRIPS) ** 1.6 for k in range(STICKER_STRIPS + 1)]
    paper, rim = STICKER_BACK
    back = f'<circle cx="{h}" cy="{h}" r="{STICKER_R}" fill="{paper}" stroke="{rim}" stroke-width="4"/>'
    front = sticker(shadow=False)
    X = Z = 0.0
    bands = []
    for i, (u0, u1) in enumerate(zip(edges, edges[1:])):
        phi = lift * (1 - math.exp(-0.5 * (u0 + u1) / curl))
        c, sn = math.cos(phi), math.sin(phi)
        col_b2 = c * col_b + sn * up
        off = T + col_b * (X - u0 * c) + up * (Z - u0 * sn)
        w = col_b * (1 - c) - up * sn
        shift = Track("transform", [(k, (v[0] * w[0], v[0] * w[1])) for k, v in lay],  # type: ignore[index]
                      transform="translate")
        # Surface normal against the view: past 45 degrees the backing shows.
        face = front if c - sn > 0 else back
        matrix = f"matrix({col_a[0]:.6f},{col_a[1]:.6f},{col_b2[0]:.6f},{col_b2[1]:.6f},{off[0]:.3f},{off[1]:.3f})"
        bands.append(band_clip(f"curl{i}", u0 - 0.6, u1 - u0 + 1.2)
                     + node("g", "", [shift], f'<g transform="{matrix}"><g clip-path="url(#curl{i})">'
                            f'<g clip-path="url(#loose)">{face}'  # shaded where the sheet turns
                            f'<circle cx="{h}" cy="{h}" r="{STICKER_R}" fill="#000" '
                            f'fill-opacity="{0.18 * math.sin(2 * phi) ** 2:.3f}"/></g></g></g>', t))
        X += (u1 - u0) * c
        Z += (u1 - u0) * sn

    # Carried in: slides from off stage and turns level about the contact point.
    contact = T + col_a * h + col_b * b0
    dx, dy, tilt = STICKER_FROM
    lay_speed = LAY_IN * (b1 - b0) * float(np.hypot(*col_b)) / (STICK_LAY[1] - STICK_LAY[0])  # px/s
    carry_end = lay_speed * (STICK_IN[1] - STICK_IN[0]) / math.hypot(dx, dy)
    path = hermite(*STICK_IN, 0.0, 1.0, CARRY_IN, carry_end)
    carry = [(k, (dx * (1 - v[0]), dy * (1 - v[0]))) for k, v in path]  # type: ignore[index]
    turn = [(k, (tilt * (1 - v[0]), contact[0], contact[1])) for k, v in path]  # type: ignore[index]
    # Off stage before the carry and fully laid after it: only rendered in between.
    lifted = node("g", "", [shown(STICK_IN[0], STICK_LAY[1])],
                  node("g", "", [Track("transform", carry, transform="translate")],
                       node("g", "", [Track("transform", turn, transform="rotate")], "".join(bands), t), t), t)
    return on_face, lifted


def build_animation(palette: str = "teal", t: float | None = None, rec: Recording | None = None) -> str:
    pal = PALETTES[palette]
    rec = rec if rec is not None else run_sim()
    resting, flying = cell_layers(rec, t)
    h = N / 2
    open_box = shown(0, CLOSED)

    tape_len = Track("height", ramp(*TAPE_TOP, 0.0, N))
    drop_len = Track("height", ramp(*TAPE_DOWN, 0.0, TAPE_DROP + 26))
    sticker_on_face, sticker_lifted = sticker_layers(t)

    top = (node("g", "", [shown(CLOSED)], top_base(pal, "t"), t)
           + node("rect", f'x="{h - TAPE_W / 2}" width="{TAPE_W}" fill="{pal["tape"]}" '
                  f'fill-opacity="{pal["tape_op"]}"', [tape_len], t=t)
           + sticker_on_face)
    left = (left_base(pal, "l")
            + node("g", "", [Track("opacity", ramp(*SHADE_IN, 0.0, 1.0, ease=ease_in_out),
                                   fmt=lambda v: f"{v[0]:.3f}")], flap_shade("l"), t)
            + '<clipPath id="tapedrop">' + node("rect", f'width="{N}"', [drop_len], t=t) + "</clipPath>"
            + f'<g clip-path="url(#tapedrop)">{tape_drop(pal)}</g>')

    x, y, w, hh = VIEW
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:g} {y:g} {w:g} {hh:g}" '
        f'width="{w:g}" height="{hh:g}" role="img" aria-label="CellPack">',
        "<title>CellPack</title>",
        f"<defs>{cell_defs()}{corrugation('corr', pal)}{corrugation_side('flutes', pal)}</defs>",
        ground_shadow(HEXA),
        # Everything inside is dropped once sealed, so the last frame is the logo.
        node("g", "", [open_box],
             interior(pal) + rims("back")
             + flap_layer(pal, "A", (0.0, FLAP_A[0]), t)  # behind the cells while open
             + resting + rim_shade() + rims("front") + flying, t),
        flap_layer(pal, "A", (FLAP_A[0], CLOSED), t),    # over them once it swings shut
        flap_layer(pal, "B", (0.0, CLOSED), t),
        face_group("left", left),  # same face order as the logo
        face_group("right", face_right(pal, "r")),
        face_group("top", top),
        sticker_lifted,
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CellPack logo and animation SVGs.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--palette", choices=sorted(PALETTES), default="teal")
    parser.add_argument("--frames", type=Path, help="also write stills of the animation here")
    parser.add_argument("--times", default="0.5,1.0,1.5,2.0,2.5,3.0,3.45,3.7,3.95,4.5,5.2,5.45,5.7,5.95,6.4",
                        help="comma-separated still times in seconds")
    parser.add_argument("--search", help="score simulation seeds lo-hi instead of writing files")
    args = parser.parse_args()
    if args.search:
        lo, hi = (int(v) for v in args.search.split("-"))
        for seed in range(lo, hi + 1):
            print(seed, {k: round(v, 3) for k, v in score(run_sim(seed), FLAP_A[0]).items()})
        return
    rec = run_sim()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("logo.svg", build_logo(args.palette)),
                       ("cellpack-animation.svg", build_animation(args.palette, rec=rec))):
        path = args.out_dir / name
        path.write_text(text, encoding="utf-8")
        print(path)
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
        for t in (float(v) for v in args.times.split(",")):
            path = args.frames / f"frame-{t:05.2f}.svg"
            path.write_text(build_animation(args.palette, t, rec), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
