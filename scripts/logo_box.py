"""Generate the CellPack logo: a sealed cardboard box with a cell sticker.

The box is a regular hexagon split into three rhombi. Each face is drawn
flat (N x N, seen head-on from outside) and mapped affinely onto its rhombus,
so every face detail is distorted the same way.

  top    two flaps meeting under a strip of tape, the cell sticker on top
  left   the tape's torn end and streaming chevrons (loading)
  right  a sparse cells x genes matrix (storage)

Usage: python scripts/logo_box.py [--out assets/logo.svg] [--palette teal|kraft]
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE = 512    # output canvas
N = 512       # flat face size
HEX_R = 200   # hexagon circumradius, also the box edge length

PALETTES = {
    # edge: the corrugated board's cut face; wall_x / wall_y / flap_in: the inside, seen while open
    "teal": dict(top="#5CC7AE", flap="#52BBA2", left="#2BA58A", right="#1E7A68",
                 tape="#E9FBF5", tape_op=0.45, ink="#E9FBF5", ink_off=0.18,
                 wall_x="#186B5A", wall_y="#23907A", flap_in="#45AF96", edge="#A9E4D4"),
    "kraft": dict(top="#E6B97F", flap="#DDAE72", left="#C9925A", right="#A87540",
                  tape="#F6E6C6", tape_op=0.6, ink="#5A3B1C", ink_off=0.15,
                  wall_x="#8E6235", wall_y="#B07E47", flap_in="#D4A266", edge="#F2D6AC"),
}
CELL = dict(sticker="#FFFFFF", cyto="#EEE8FF", line="#7B5CD6", granule="#B9A6EE")
GROUND = "#0B2E27"

TAPE_W = 80      # tape width
TAPE_DROP = 165  # how far the tape runs down the left face
EDGE_SHADE = 16  # narrow grey shade under the flaps along the left face's top
BAND = 32 / 3    # corrugated board thickness in face units (shown only while open)
FLUTE = 44 / 3   # flute wavelength along a cut edge, in face units


# --------------------------------------------------------------- flat faces

def linear(gid: str, x2: float, y2: float, stops: tuple[tuple[float, str, float], ...]) -> str:
    body = "".join(f'<stop offset="{o}" stop-color="{c}" stop-opacity="{a:.3f}"/>'
                   for o, c, a in stops)
    return f'<linearGradient id="{gid}" x1="0" y1="0" x2="{x2}" y2="{y2}">{body}</linearGradient>'


def mix(a: str, b: str, u: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * u):02X}" for x, y in zip(ca, cb))


def corrugation(pid: str, pal: dict) -> str:
    """Pattern tile for an edge cut across the flutes: two liners with the fluted
    medium waving between them along x. Everything stays light and soft so the
    flutes read as a wave rather than a dashed line; the hollows under the top
    liner are a touch deeper since the light comes from above."""
    H, L = BAND, FLUTE
    t = 0.16 * H
    amp, mid = (H - 2 * t) * 0.38, H / 2
    wave = [(L * i / 16, mid + amp * math.cos(2 * math.pi * i / 16)) for i in range(17)]
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in wave)
    upper = f"0,{t:.2f} {L},{t:.2f} " + " ".join(f"{x:.2f},{y:.2f}" for x, y in reversed(wave))
    edge = pal["edge"]
    return (f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="{L}" height="{H}">'
            f'<rect width="{L}" height="{H}" fill="{mix(edge, "#000000", 0.10)}"/>'
            f'<polygon points="{upper}" fill="{mix(edge, "#000000", 0.18)}"/>'
            f'<polyline points="{pts}" fill="none" stroke="{mix(edge, "#FFFFFF", 0.45)}" '
            f'stroke-width="{0.13 * H:.2f}" stroke-linejoin="round"/>'
            f'<rect width="{L}" height="{t:.2f}" fill="{mix(edge, "#FFFFFF", 0.4)}"/>'
            f'<rect y="{H - t:.2f}" width="{L}" height="{t:.2f}" fill="{mix(edge, "#FFFFFF", 0.15)}"/></pattern>')


def corrugation_side(gid: str, pal: dict) -> str:
    """Fill for an edge cut along the flutes: the medium is seen side-on, a
    ridge running the length of the edge between the two liners, shaded like
    the face of a roof tile."""
    edge = pal["edge"]
    t = 0.16
    stops = ((0, mix(edge, "#FFFFFF", 0.4)), (t, mix(edge, "#FFFFFF", 0.4)),
             (t, mix(edge, "#FFFFFF", 0.2)), (0.45, edge), (0.75, mix(edge, "#000000", 0.16)),
             (1 - t, mix(edge, "#000000", 0.22)), (1 - t, mix(edge, "#FFFFFF", 0.15)),
             (1, mix(edge, "#FFFFFF", 0.15)))
    body = "".join(f'<stop offset="{o:.2f}" stop-color="{c}"/>' for o, c in stops)
    return (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="0" '
            f'x2="0" y2="{BAND}">{body}</linearGradient>')


def flap_shade(p: str) -> str:
    """Narrow grey shade where the flaps meet the left panel."""
    return (f'<defs>{linear(p + "flaps", 0, 1, ((0, "#000000", 0.16), (1, "#000000", 0)))}</defs>'
            f'<rect width="{N}" height="{EDGE_SHADE}" fill="url(#{p}flaps)"/>')


def tear_edge(x0: float, x1: float, y: float, seed: int = 7, teeth: int = 9,
              amp: float = 14) -> list[tuple[float, float]]:
    """Jagged points from x0 to x1 around depth y, slanted like a hand-torn end."""
    rng = random.Random(seed)
    pts = []
    for i in range(teeth + 1):
        x = x0 + (x1 - x0) * i / teeth
        dy = (amp if i % 2 else -amp * 0.4) * rng.uniform(0.45, 1.15)
        pts.append((x, y + dy + (x - x0) * 0.10))
    return pts


def top_base(pal: dict, p: str) -> str:
    """Closed top: the two flaps and the seam between them."""
    h = N / 2
    return "\n".join([
        f'<defs>{linear(p + "crease", 1, 0, ((0, "#000000", 0), (1, "#000000", 0.10)))}</defs>',
        f'<rect width="{h}" height="{N}" fill="{pal["top"]}"/>',
        f'<rect x="{h}" width="{h}" height="{N}" fill="{pal["flap"]}"/>',
        f'<rect x="{h - 60}" width="60" height="{N}" fill="url(#{p}crease)"/>',
        f'<rect x="{h - 2}" width="4" height="{N}" fill="#000" fill-opacity=".22"/>',
    ])


def top_tape(pal: dict, height: float = N) -> str:
    """Tape along the seam, running from the back edge (b=0) toward the front."""
    return (f'<rect x="{N / 2 - TAPE_W / 2}" width="{TAPE_W}" height="{height}" '
            f'fill="{pal["tape"]}" fill-opacity="{pal["tape_op"]}"/>')


STICKER_R = 150  # sticker radius on the flat top face


def sticker(shadow: bool = True) -> str:
    """Cell sticker centred on the top face. On the rhombus, flat left/right land
    at screen upper-left/lower-right; the off-centre nucleus never reads as an eye."""
    h = N / 2
    return "\n".join([
        *([f'<circle cx="{h + 4}" cy="{h + 14}" r="{STICKER_R + 2}" fill="#000" fill-opacity=".16"/>']
          if shadow else []),
        f'<circle cx="{h}" cy="{h}" r="{STICKER_R}" fill="{CELL["sticker"]}"/>',
        f'<circle cx="{h}" cy="{h}" r="150" fill="none" stroke="#000" stroke-opacity=".06" stroke-width="3"/>',
        f'<circle cx="{h}" cy="{h}" r="124" fill="{CELL["cyto"]}" stroke="{CELL["line"]}" stroke-width="9"/>',
        f'<circle cx="{h - 42}" cy="{h + 13}" r="44" fill="{CELL["line"]}"/>',
        f'<circle cx="{h + 54}" cy="{h - 8}" r="9" fill="{CELL["granule"]}"/>',
        f'<circle cx="{h + 50}" cy="{h + 40}" r="7" fill="{CELL["granule"]}"/>',
    ])


def face_top(pal: dict, p: str) -> str:
    return "\n".join([top_base(pal, p), top_tape(pal), sticker()])


def tape_drop(pal: dict) -> str:
    t0, t1 = N / 2 - TAPE_W / 2, N / 2 + TAPE_W / 2
    edge = tear_edge(t0, t1, TAPE_DROP)
    d = f"M{t0} 0 " + " ".join(f"L{x:.1f} {y:.1f}" for x, y in edge) + f" L{t1} 0Z"
    shadow = f"M{t0} 0 " + " ".join(f"L{x + 3:.1f} {y + 6:.1f}" for x, y in edge) + f" L{t1 + 5} 0Z"
    return "\n".join([
        f'<path d="{shadow}" fill="#000" fill-opacity=".10"/>',
        f'<path d="{d}" fill="{pal["tape"]}" fill-opacity="{pal["tape_op"]}"/>',
    ])


def left_base(pal: dict, p: str) -> str:
    """Left panel with its streaming chevrons."""
    parts = [
        f'<defs>{linear(p + "shade", 0, 1, ((0, "#FFFFFF", 0.10), (1, "#000000", 0.12)))}</defs>',
        f'<rect width="{N}" height="{N}" fill="{pal["left"]}"/>',
        f'<rect width="{N}" height="{N}" fill="url(#{p}shade)"/>',
    ]
    y0, y1, w, step = 262, 422, 58, 112
    ym = (y0 + y1) / 2
    for k in range(3):
        x = 92 + k * step
        parts.append(f'<path d="M{x} {y0} H{x + w} L{x + w + 80} {ym} L{x + w} {y1} H{x} L{x + 80} {ym}Z" '
                     f'fill="{pal["ink"]}" fill-opacity="{0.55 + 0.15 * k:.2f}"/>')
    return "\n".join(parts)


def face_left(pal: dict, p: str) -> str:
    return "\n".join([left_base(pal, p), flap_shade(p), tape_drop(pal)])


def face_right(pal: dict, p: str) -> str:
    parts = [
        f'<defs>{linear(p + "shade", 0, 1, ((0, "#FFFFFF", 0.06), (1, "#000000", 0.16)))}</defs>',
        f'<rect width="{N}" height="{N}" fill="{pal["right"]}"/>',
        f'<rect width="{N}" height="{N}" fill="url(#{p}shade)"/>',
    ]
    nonzero = {(0, 0), (2, 0), (1, 1), (3, 1), (0, 2), (2, 2), (3, 3), (1, 3)}
    n, pad, gap = 4, 86, 26
    cs = (N - 2 * pad - (n - 1) * gap) / n
    for i in range(n):
        for j in range(n):
            x, y = pad + j * (cs + gap), pad + i * (cs + gap)
            op = 0.85 if (j, i) in nonzero else pal["ink_off"]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cs:.1f}" height="{cs:.1f}" rx="6" '
                         f'fill="{pal["ink"]}" fill-opacity="{op}"/>')
    return "\n".join(parts)


# --------------------------------------------------------------- hexagon
#
# Pointy-top regular hexagon around the canvas centre O. Its vertices, from
# the top clockwise: T, UR, LR, B, LL, UL. The faces are the rhombi
#   top   T, UR, O, UL     left  UL, O, B, LL     right  O, UR, LR, B
# Each face maps flat (a, b) with a to the right and b downward, as seen from
# outside the box.

Point = tuple[float, float]


def hexagon() -> dict[str, Point]:
    c = SIZE / 2
    names = ("T", "UR", "LR", "B", "LL", "UL")
    pts = {n: (c + HEX_R * math.cos(math.radians(-90 + 60 * i)),
               c + HEX_R * math.sin(math.radians(-90 + 60 * i))) for i, n in enumerate(names)}
    pts["O"] = (c, c)
    return pts


FACES = {  # name: (corner at a=b=0, corner at a=N, corner at b=N)
    "left": ("UL", "O", "LL"),
    "right": ("O", "UR", "B"),
    "top": ("T", "UR", "UL"),
}
DRAW = {"left": face_left, "right": face_right, "top": face_top}


def face_matrix(hexa: dict[str, Point], name: str) -> str:
    (ox, oy), (ax, ay), (bx, by) = (hexa[k] for k in FACES[name])
    return (f"matrix({(ax - ox) / N:.6f},{(ay - oy) / N:.6f},"
            f"{(bx - ox) / N:.6f},{(by - oy) / N:.6f},{ox:.3f},{oy:.3f})")


def ground_shadow(hexa: dict[str, Point]) -> str:
    # The base's hidden back corner falls on the centre, so the footprint is LL, B, LR, O.
    base = " ".join(f"{x:.1f},{y + 10:.1f}" for x, y in (hexa[k] for k in ("LL", "B", "LR", "O")))
    return ('<defs><filter id="ground" x="-30%" y="-30%" width="160%" height="160%">'
            '<feGaussianBlur stdDeviation="9"/></filter></defs>'
            f'<polygon points="{base}" fill="{GROUND}" fill-opacity=".28" filter="url(#ground)"/>')


def build_logo(palette: str = "teal") -> str:
    pal = PALETTES[palette]
    hexa = hexagon()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" '
        f'width="{SIZE}" height="{SIZE}" role="img" aria-label="CellPack">',
        "<title>CellPack</title>",
        ground_shadow(hexa),
    ]
    for name in ("left", "right", "top"):
        parts.append(f'<g transform="{face_matrix(hexa, name)}">\n{DRAW[name](pal, name[0])}\n</g>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CellPack box logo SVG.")
    parser.add_argument("--out", type=Path, default=ROOT / "assets" / "logo.svg")
    parser.add_argument("--palette", choices=sorted(PALETTES), default="teal")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_logo(args.palette), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
