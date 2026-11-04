"""Rigid-sphere simulation of cells raining into the open logo box.

Unit box coordinates: x and y span the floor [0, 1]^2, z points up, the box is
open at z = 1. Spheres are released one by one above the opening, fall under
gravity, and collide with each other, the four walls and the floor. Contacts
use velocity impulses with restitution and Coulomb friction (which is what
spins the cells), then a few relaxation passes push overlapping spheres apart.
The solver is vectorised Jacobi style, which is plenty for a short animation.

The recording is sampled at FPS: positions, orientations and the time each
sphere first touched something (where it stops "flying").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 16.0           # gravity, box units / s^2
DT = 1 / 240       # physics step
FPS = 30           # recording rate
RESTITUTION = 0.36  # a little bounce, like soft rubber
MU = 0.9           # friction: contacts grip, set cells rolling and let them heap
SPEED_REST = 0.35  # contacts slower than this do not bounce
VEL_PASSES, POS_PASSES = 3, 6
ROLL_DRAG = 6.0     # rolling resistance on spin while touching, 1/s
FENCE = 1.4         # walls reach this high, so a heap over the rim never hangs outside
SETTLE_SPEED = 0.25  # resting contacts slower than this are damped hard


@dataclass
class Recording:
    radius: np.ndarray    # (n,)
    release: np.ndarray   # (n,) seconds
    contact: np.ndarray   # (n,) first-contact time, inf if never
    times: np.ndarray     # (f,)
    pos: np.ndarray       # (f, n, 3)
    quat: np.ndarray      # (f, n, 4) w, x, y, z


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    u, w = q[..., 1:], q[..., :1]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def simulate(n: int, seed: int, t_end: float, release, spawn_z, aim=None,
             ceiling=None, radius_median: float = 0.066, radius_sigma: float = 0.3,
             radius_range: tuple[float, float] = (0.042, 0.11), spin: float = 6.0) -> Recording:
    """release(rng, n) gives the n release times in seconds.
    spawn_z(x, y, r) gives the release height that keeps a sphere off stage.
    aim(rng, release, r) gives where each sphere crosses the rim, (x, y) arrays;
    uniform over the opening if None.
    ceiling(t, x) gives, per sphere, the highest point its top may reach (the
    closing flaps pressing the pile down), or None while nothing presses."""
    rng = np.random.default_rng(seed)
    r = np.clip(rng.lognormal(math.log(radius_median), radius_sigma, n), *radius_range)
    m = r ** 3
    im, ii = 1 / m, 1 / (0.4 * m * r ** 2)
    rel = np.sort(np.asarray(release(rng, n), dtype=float))

    # Aim each sphere at a point of the opening and give it a little sideways drift.
    if aim is None:
        tx = rng.uniform(r + 0.02, 1 - r - 0.02)
        ty = rng.uniform(r + 0.02, 1 - r - 0.02)
    else:
        tx, ty = aim(rng, rel, r)
    vxy = rng.uniform(-0.3, 0.3, (n, 2))
    z0 = np.array([spawn_z(x, y, ri) for x, y, ri in zip(tx, ty, r)])
    t_rim = np.sqrt(2 * np.maximum(z0 - 1, 0) / G)
    z0 = np.array([spawn_z(x, y, ri) for x, y, ri in zip(tx - vxy[:, 0] * t_rim, ty - vxy[:, 1] * t_rim, r)])
    t_rim = np.sqrt(2 * np.maximum(z0 - 1, 0) / G)
    p = np.stack([tx - vxy[:, 0] * t_rim, ty - vxy[:, 1] * t_rim, z0], axis=1)
    v = np.concatenate([vxy, np.zeros((n, 1))], axis=1)
    w = rng.normal(size=(n, 3)) * spin / math.sqrt(3)  # tumbling on the way down
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    contact = np.full(n, np.inf)

    steps = int(round(t_end / DT))
    every = int(round(1 / (FPS * DT)))
    times, pos_rec, quat_rec = [], [], []
    iu = np.triu_indices(n, 1)

    for step in range(steps + 1):
        t = step * DT
        if step % every == 0:
            times.append(t)
            pos_rec.append(p.copy())
            quat_rec.append(q.copy())
        if step == steps:
            break
        act = rel <= t
        v[act, 2] -= G * DT
        p[act] += v[act] * DT
        wq = np.concatenate([np.zeros((n, 1)), w], axis=1)
        q[act] += 0.5 * DT * quat_mul(wq, q)[act]
        q /= np.linalg.norm(q, axis=1, keepdims=True)

        touching = np.zeros(n, dtype=bool)
        d = p[None, :, :] - p[:, None, :]
        dist = np.linalg.norm(d, axis=2)
        both = act[:, None] & act[None, :]
        hit = (dist < (r[:, None] + r[None, :])) & both
        hi, hj = iu
        sel = hit[hi, hj]
        I, J = hi[sel], hj[sel]
        touching[I] = touching[J] = True

        # Sphere-sphere velocity impulses (normal + friction).
        for _ in range(VEL_PASSES if len(I) else 0):
            dvec = p[J] - p[I]
            nrm = dvec / np.maximum(np.linalg.norm(dvec, axis=1, keepdims=True), 1e-9)
            ra, rb = r[I, None] * nrm, -r[J, None] * nrm
            vrel = (v[J] + np.cross(w[J], rb)) - (v[I] + np.cross(w[I], ra))
            vn = np.einsum("ij,ij->i", vrel, nrm)
            go = vn < 0
            if not go.any():
                break
            e = np.where(-vn > SPEED_REST, RESTITUTION, 0.0)
            jn = np.where(go, -(1 + e) * vn / (im[I] + im[J]), 0.0)
            vt = vrel - vn[:, None] * nrm
            vt_len = np.linalg.norm(vt, axis=1)
            tan = vt / np.maximum(vt_len[:, None], 1e-9)
            k = im[I] + im[J] + r[I] ** 2 * ii[I] + r[J] ** 2 * ii[J]
            jt = np.where(go, np.minimum(vt_len / k, MU * jn), 0.0)
            imp = jn[:, None] * nrm - jt[:, None] * tan  # impulse on J
            cnt = np.bincount(np.concatenate([I, J]), minlength=n).clip(1)[:, None]
            dv, dw = np.zeros((n, 3)), np.zeros((n, 3))
            np.add.at(dv, I, -imp * im[I, None])
            np.add.at(dv, J, imp * im[J, None])
            np.add.at(dw, I, np.cross(ra, -imp) * ii[I, None])
            np.add.at(dw, J, np.cross(rb, imp) * ii[J, None])
            v += dv / cnt
            w += dw / cnt

        # Walls (up to FENCE) and floor.
        inside = act & (p[:, 2] - r < FENCE)
        for axis, side in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)):
            lim = r if side == 0 else 1 - r
            over = inside & ((p[:, axis] < lim) if side == 0 else (p[:, axis] > lim))
            if not over.any():
                continue
            touching |= over
            nrm = np.zeros(3)
            nrm[axis] = 1.0 if side == 0 else -1.0  # points into the box
            p[over, axis] = lim[over]
            vn = v[over] @ nrm
            into = vn < 0
            e = np.where(-vn > SPEED_REST, RESTITUTION, 0.0)
            jn = np.where(into, -(1 + e) * vn / im[over], 0.0)
            rc = -r[over, None] * nrm  # contact point from the centre
            vc = v[over] + np.cross(w[over], rc)
            vt = vc - (vc @ nrm)[:, None] * nrm
            vt_len = np.linalg.norm(vt, axis=1)
            tan = vt / np.maximum(vt_len[:, None], 1e-9)
            jt = np.minimum(vt_len / (im[over] + r[over] ** 2 * ii[over]), MU * jn)
            imp = jn[:, None] * nrm - jt[:, None] * tan
            v[over] += imp * im[over, None]
            w[over] += np.cross(rc, imp) * ii[over, None]

        # Push overlapping spheres apart.
        for _ in range(POS_PASSES if len(I) else 0):
            dvec = p[J] - p[I]
            dl = np.maximum(np.linalg.norm(dvec, axis=1), 1e-9)
            pen = r[I] + r[J] - dl
            ok = pen > 0
            if not ok.any():
                break
            nrm = dvec / dl[:, None]
            share = (pen / (im[I] + im[J]))[:, None] * nrm * ok[:, None] * 0.5
            dp = np.zeros((n, 3))
            np.add.at(dp, I, -share * im[I, None])
            np.add.at(dp, J, share * im[J, None])
            p += dp
            for axis in range(2):
                p[inside, axis] = np.clip(p[inside, axis], r[inside], 1 - r[inside])
            p[inside, 2] = np.maximum(p[inside, 2], r[inside])

        # The closing flaps press down on whatever sticks out of the box.
        lid = ceiling(t, p[:, 0]) if ceiling is not None else None
        if lid is not None:
            over = act & (p[:, 2] + r > lid)
            p[over, 2] = lid[over] - r[over]
            v[over, 2] = np.minimum(v[over, 2], 0.0)
            touching |= over

        # Rolling resistance and air drag, so the pile comes to rest.
        v *= 1 - 0.4 * DT
        v[touching] *= 1 - 2.0 * DT
        w[touching] *= 1 - ROLL_DRAG * DT
        slow = touching & (np.linalg.norm(v, axis=1) < SETTLE_SPEED)
        v[slow] *= 0.9
        w[slow] *= 0.9
        contact[touching & np.isinf(contact)] = t

    return Recording(r, rel, contact, np.array(times), np.array(pos_rec), np.array(quat_rec))


def score(rec: Recording, settle_t: float) -> dict:
    k = int(np.searchsorted(rec.times, settle_t))
    top = rec.pos[k, :, 2] + rec.radius
    speed = np.linalg.norm(rec.pos[min(k + 1, len(rec.times) - 1)] - rec.pos[k], axis=1) * FPS
    p_end = rec.pos[-1]
    inside = ((p_end[:, :2] >= rec.radius[:, None] - 1e-3).all(1)
              & (p_end[:, :2] <= 1 - rec.radius[:, None] + 1e-3).all(1) & (p_end[:, 2] < 1))
    return dict(top_max=float(top.max()), top_p90=float(np.percentile(top, 90)),
                moving=int((speed > 0.05).sum()), escaped=int((~inside).sum()))
