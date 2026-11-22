"""Mech Royale bot v1: survive, fight lone enemies from cover, lead shots.

Built from the plan agreed with Sanjiban (see thread). Sections follow the "kits":
  map helpers, track & aim, weapon choice, resources, upgrades, movement/decision.
"""
import math
import os
import array
import json

import mechfight as mf

TAU = 2 * math.pi
MODE = "ghost"   # experiment A: never engage first, always break line of sight, shoot back only when cornered
CELL = 2.0  # meters per cell of our coarse height map


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def yaw_to(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def dist2(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# ---------------------------------------------------------------- map helpers
class Map:
    """Coarse 2 m grid holding the highest solid surface (terrain or building roof)."""

    def __init__(self, mapdir):
        self.ok = False
        try:
            w = json.load(open(os.path.join(mapdir, "v2", "world.json")))
            t = w["terrain"]
            self.minx, self.miny = t["minX"], t["minY"]
            nx_t, ny_t, step = t["nx"], t["ny"], t["step"]
            self.nx = int(nx_t * step / CELL)
            self.ny = int(ny_t * step / CELL)
            raw = array.array("h")
            with open(os.path.join(mapdir, "v2", t.get("file", "terrain.i16")), "rb") as f:
                raw.frombytes(f.read())
            if array.array("h", [1]).tobytes() != b"\x01\x00":
                raw.byteswap()
            scale = t.get("scale", 0.01)
            k = int(CELL / step)
            self.ground = array.array("f", [0.0]) * (self.nx * self.ny)
            for j in range(self.ny):
                row = j * k * nx_t
                base = j * self.nx
                for i in range(self.nx):
                    self.ground[base + i] = raw[row + i * k] * scale
            self.h = array.array("f", self.ground)
            for p in w["prisms"]:
                if p["kind"] == "bridge":
                    continue
                self._fill(p["ring"], p["top"])
            self.ok = True
        except Exception as e:  # never crash on map trouble: run without map knowledge
            mf.log("map load failed:", repr(e))

    def _fill(self, ring, top):
        xs, ys = ring[0::2], ring[1::2]
        n = len(xs)
        j0 = max(0, int((min(ys) - self.miny) / CELL))
        j1 = min(self.ny - 1, int((max(ys) - self.miny) / CELL))
        for j in range(j0, j1 + 1):
            y = self.miny + (j + 0.5) * CELL
            cuts = []
            for e in range(n):
                x1, y1, x2, y2 = xs[e], ys[e], xs[(e + 1) % n], ys[(e + 1) % n]
                if (y1 <= y) != (y2 <= y):
                    cuts.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            cuts.sort()
            for a, b in zip(cuts[0::2], cuts[1::2]):
                i0 = max(0, int((a - self.minx) / CELL))
                i1 = min(self.nx - 1, int((b - self.minx) / CELL))
                base = j * self.nx
                for i in range(i0, i1 + 1):
                    if self.h[base + i] < top:
                        self.h[base + i] = top

    def height(self, x, y):
        i = int((x - self.minx) / CELL)
        j = int((y - self.miny) / CELL)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return self.h[j * self.nx + i]
        return 1e9

    def clear(self, a, b):
        """True if the straight line a->b (3D points) passes over every solid on the way."""
        if not self.ok:
            return True
        d = math.dist(a[:2], b[:2])
        n = max(1, int(d / CELL))
        for s in range(1, n):
            f = s / n
            if self.height(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f) > a[2] + (b[2] - a[2]) * f:
                return False
        return True


# ---------------------------------------------------------------- navigation
import heapq
import time as _time

NAV = 8.0  # meters per navigation cell


class Nav:
    """8 m walkable grid + A* (time-boxed so a tick never runs long)."""

    def __init__(self, m):
        self.ok = False
        if not m.ok:
            return
        self.m = m
        k = int(NAV / CELL)
        self.nx, self.ny = m.nx // k, m.ny // k
        self.minx, self.miny = m.minx, m.miny
        self.block = bytearray(self.nx * self.ny)
        self.g = array.array("f", [0.0]) * (self.nx * self.ny)
        for j in range(self.ny):
            for i in range(self.nx):
                solid, gs = 0, 0.0
                for dj in range(k):
                    base = (j * k + dj) * m.nx + i * k
                    for di in range(k):
                        gg = m.ground[base + di]
                        gs += gg
                        if m.h[base + di] > gg + 2:
                            solid += 1
                self.g[j * self.nx + i] = gs / (k * k)
                if solid * 4 >= k * k:
                    self.block[j * self.nx + i] = 1
        self.ok = True

    def cell(self, x, y):
        i = min(self.nx - 1, max(0, int((x - self.minx) / NAV)))
        j = min(self.ny - 1, max(0, int((y - self.miny) / NAV)))
        return i, j

    def center(self, i, j):
        return (self.minx + (i + 0.5) * NAV, self.miny + (j + 0.5) * NAV)

    def free_near(self, i, j):
        if not self.block[j * self.nx + i]:
            return i, j
        for r in (1, 2, 3):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < self.nx and 0 <= jj < self.ny and not self.block[jj * self.nx + ii]:
                        return ii, jj
        return i, j

    def path(self, a, b, budget=0.010):
        t0 = _time.perf_counter()
        nx = self.nx
        s = self.free_near(*self.cell(a[0], a[1]))
        e = self.free_near(*self.cell(b[0], b[1]))
        if s == e:
            return [b]
        si, ei = s[1] * nx + s[0], e[1] * nx + e[0]
        ex, ey = e
        openh = [(0.0, si)]
        gcost = {si: 0.0}
        came = {}
        best, bh = si, 1e18
        n = 0
        nb = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        while openh:
            _, c = heapq.heappop(openh)
            if c == ei:
                best = c
                break
            n += 1
            if n % 200 == 0 and _time.perf_counter() - t0 > budget:
                break
            ci, cj = c % nx, c // nx
            h = math.hypot(ci - ex, cj - ey)
            if h < bh:
                best, bh = c, h
            gc = gcost[c]
            for di, dj, w in nb:
                ii, jj = ci + di, cj + dj
                if not (0 <= ii < nx and 0 <= jj < self.ny):
                    continue
                q = jj * nx + ii
                if self.block[q]:
                    continue
                if di and dj and (self.block[cj * nx + ii] or self.block[jj * nx + ci]):
                    continue
                if abs(self.g[q] - self.g[c]) > 8:
                    continue
                ng = gc + w
                if ng < gcost.get(q, 1e18):
                    gcost[q] = ng
                    came[q] = c
                    heapq.heappush(openh, (ng + math.hypot(ii - ex, jj - ey), q))
        out = []
        c = best
        while c in came:
            out.append(self.center(c % nx, c // nx))
            c = came[c]
        out.reverse()
        if best == ei:
            out.append((b[0], b[1]))
        return out


# ---------------------------------------------------------------- track & aim
def lead_point(me_muzzle, target, speed):
    """Where to aim a projectile of `speed` so a target moving at constant velocity walks into it."""
    p = [target["pos"][0], target["pos"][1], target["pos"][2] + 4.5]
    v = target["vel"]
    aim = p
    for _ in range(3):
        t = math.dist(me_muzzle, aim) / speed
        aim = [p[0] + v[0] * t, p[1] + v[1] * t, p[2] + max(v[2], -5) * t]
    return aim


def aim_angles(src, pt):
    return [math.atan2(pt[1] - src[1], pt[0] - src[0]),
            math.atan2(pt[2] - src[2], max(0.1, math.dist(src[:2], pt[:2])))]


def mortar_pitch(dist_xy, dz, v, g):
    """Low-arc pitch to land a shell dist_xy away (dz higher). None if out of range."""
    v2 = v * v
    disc = v2 * v2 - g * (g * dist_xy * dist_xy + 2 * dz * v2)
    if disc < 0:
        return None
    return math.atan2(v2 - math.sqrt(disc), g * dist_xy)


# ---------------------------------------------------------------- weapon choice
BEATS = {"laser": "mortar", "mortar": "lightning", "lightning": "laser"}


def preferred_second(late, enemy_kinds):
    # v5: mortar stays, but always last: better than an empty slot, replaced by laser/lightning
    lst = _preferred_second(late, enemy_kinds)
    return [k for k in lst if k != "mortar"] + ["mortar"]


def _preferred_second(late, enemy_kinds):
    if not late:
        return ["laser", "lightning", "mortar"]
    counts = {k: enemy_kinds.get(k, 0) for k in BEATS.values()}
    if sum(counts.values()) == 0:
        return ["lightning", "laser", "mortar"]
    common = max(counts, key=counts.get)
    counter = [k for k, beaten in BEATS.items() if beaten == common][0]
    rest = [k for k in ("lightning", "laser", "mortar") if k != counter]
    return [counter] + rest


# ---------------------------------------------------------------- upgrades
UPGRADE_ORDER = ["plating", "armor", "nanites", "gun_damage", "gun_rate",
                 "laser_damage", "laser_charge", "lightning_efficiency", "lightning_range", "lightning_chain",
                 "mortar_reload", "gun_velocity", "gyros", "sensors", "servos", "capacitor", "reactor",
                 "jets", "gun_mag", "mortar_splash", "mortar_velocity", "laser_pierce"]


def pick_upgrade(choices, danger):
    order = UPGRADE_ORDER
    if danger:
        order = ["armor", "plating"] + order
    rank = {u: i for i, u in enumerate(order)}
    best = min(range(len(choices)), key=lambda i: rank.get(choices[i]["id"], 99))
    return best


# ---------------------------------------------------------------- the bot
class Bot:
    name = "sanjiban-A-ghost"

    def __init__(self, hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.map = Map(hello["map"]["dir"])
        self.reach = cfg["loot"]["pickup_radius"]
        self.gun_speed = cfg["weapons"]["gun"]["speed"]
        self.seen_kinds = {}      # mech id -> weapon kinds last seen
        self.last_seen = {}       # mech id -> (t, pos)
        self.loot = {}            # pickup id -> (kind, pos)
        self.threat_t = -99.0     # last time we were in danger
        self.unseen_hit_t = -99.0
        self.unseen_bearing = 0.0
        self.stuck_t = 0.0
        self.last_pos = None
        self.strafe = 1
        self.strafe_t = 0.0
        self.sounds = []          # (t, bearing, kind)
        self.retreat_target = None
        self.hide = None          # current hiding spot [x, y, z]
        self.hide_zone = None     # next_center/radius it was chosen for
        self.spots = self.wall_spots()
        self.nav = Nav(self.map)
        self.path = []
        self.path_dest = None
        self.path_t = -99.0
        self.kills = 0
        self.last_hit_t = -99.0
        self.duck = None
        self.duck_t = -99.0
        self.multi_t = -99.0

    # --- hiding spots: open ground cells right next to a tall wall
    def wall_spots(self):
        m = self.map
        out = []
        if not m.ok:
            return out
        step = 3  # every 6 m
        for j in range(2, m.ny - 2, step):
            for i in range(2, m.nx - 2, step):
                k = j * m.nx + i
                g = m.ground[k]
                if m.h[k] > g + 1.0:
                    continue  # inside a building
                tall = 0
                for di, dj in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                    kk = (j + dj) * m.nx + (i + di)
                    if m.h[kk] > g + 9:
                        tall += 1
                if tall:
                    out.append((m.minx + (i + 0.5) * CELL, m.miny + (j + 0.5) * CELL, g, tall))
        mf.log("hiding spots:", len(out))
        return out

    def pick_hide(self, me, zone):
        c, r = zone["next_center"], zone["next_radius"]
        lim = max(8.0, 0.55 * r)
        best, bs = None, 1e18
        x, y = me["pos"][0], me["pos"][1]
        for sx, sy, sz, tall in self.spots:
            dc = math.hypot(sx - c[0], sy - c[1])
            if dc > lim:
                continue
            d = math.hypot(sx - x, sy - y)
            score = d + 0.5 * dc - 15 * tall
            if score < bs:
                best, bs = [sx, sy, sz], score
        return best

    # --- walking somewhere: straight if the way is clear, else follow an A* path
    def steer(self, me, dest, t):
        x, y, z = me["pos"]
        if dist2(me["pos"], dest) < 15 or not self.nav.ok or self.map.clear([x, y, z + 3], [dest[0], dest[1], z + 3]):
            self.path = []
            return yaw_to(me["pos"], dest)
        if (not self.path or self.path_dest is None or dist2(self.path_dest, dest) > 10
                or t - self.path_t > 4 or self.stuck_t > 0.6):
            self.path = self.nav.path(me["pos"], dest)
            self.path_dest = list(dest[:2])
            self.path_t = t
        while len(self.path) > 1 and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < 6:
            self.path.pop(0)
        # shortcut: aim at the farthest of the next few waypoints we can walk to directly
        for k in range(min(4, len(self.path)) - 1, -1, -1):
            p = self.path[k]
            if self.map.clear([x, y, z + 3], [p[0], p[1], z + 3]):
                return math.atan2(p[1] - y, p[0] - x)
        if self.path:
            p = self.path[0]
            return math.atan2(p[1] - y, p[0] - x)
        return yaw_to(me["pos"], dest)

    def open_dir(self, me):
        """Direction away from nearby walls (where enemies can come from)."""
        x, y, z = me["pos"]
        key = (round(x / 4), round(y / 4))
        if getattr(self, "_open_key", None) == key:
            return self._open_val
        wx = wy = 0.0
        for k in range(12):
            a = k * TAU / 12
            for r in (4, 8, 12):
                if self.map.height(x + r * math.cos(a), y + r * math.sin(a)) > z + 5:
                    wx += math.cos(a) / r
                    wy += math.sin(a) / r
                    break
        if wx == 0 and wy == 0:
            val = me["aim"][0]
        else:
            val = math.atan2(-wy, -wx)
        self._open_key, self._open_val = key, val
        return val

    # --- small helpers
    def visible_from(self, a_pos, b_pos):
        a = [a_pos[0], a_pos[1], a_pos[2] + 8]
        b = [b_pos[0], b_pos[1], b_pos[2] + 6]
        return self.map.clear(a, b)

    def cover_spot(self, me, threats, zone):
        """Nearest point (within ~40 m) that no threat can see, inside the zone if possible."""
        x, y, z = me["pos"]
        best, best_score = None, 1e9
        for r in (10, 20, 32):
            for k in range(16):
                a = k * TAU / 16
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                gz = self.map.height(px, py) if self.map.ok else z
                if gz > z + 1.2:          # that spot is inside a building / up a wall
                    continue
                p = [px, py, gz]
                if not self.map.clear([x, y, z + 6], [px, py, gz + 6]):
                    continue              # can't walk straight there
                hidden = all(not self.visible_from(t, p) for t in threats)
                if not hidden:
                    continue
                score = r
                for t in threats:
                    score -= 0.3 * dist2(t, p)   # prefer farther from threats
                if math.dist(p[:2], zone["center"]) > zone["radius"]:
                    score += 60
                if score < best_score:
                    best, best_score = p, score
        return best

    def act(self, obs):
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        zone = obs["zone"]
        act = {"tick": obs["tick"]}

        # ---- remember things (vision memory beyond the server's 5 s)
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        for m in enemies:
            self.seen_kinds[m["id"]] = m.get("weapons", [])
            self.last_seen[m["id"]] = (t, m["pos"])
        for p in obs["pickups"]:
            self.loot[p["id"]] = (p["kind"], p["pos"])
        if sum(1 for m in enemies if dist2(me["pos"], m["pos"]) < 200) >= 2:
            self.multi_t = t
        for e in obs["events"]:
            if e["type"] in ("took",) or (e["type"] == "take_failed" and e["reason"] == "gone"):
                self.loot.pop(e["id"], None)
            if e["type"] == "damaged":
                self.last_hit_t = t
            if e["type"] == "kill":
                self.kills += 1
            if e["type"] == "damaged" and e.get("by") is not None:
                if not any(m["id"] == e["by"] for m in enemies) and e.get("bearing") is not None:
                    self.unseen_hit_t = t
                    self.unseen_bearing = e["bearing"]
        for s in obs["sounds"]:
            if s["kind"] in ("gun", "laser", "lightning", "mortar", "laser_charge") and s["dist"] != "far":
                self.sounds.append((t - s["age"], s["bearing"], s["kind"]))
        self.sounds = [s for s in self.sounds if t - s[0] < 1.0]

        # ---- crowd count: visible mechs near + extra sound directions
        near = [m for m in enemies if dist2(me["pos"], m["pos"]) < 150]
        bearings = [yaw_to(me["pos"], m["pos"]) for m in near]
        extra = []
        for s in self.sounds:
            if all(abs(wrap(s[1] - b)) > math.radians(15) for b in bearings + extra):
                extra.append(s[1])
        # v3: sounds alone no longer count (one shooter's noisy bearings looked like a crowd)
        crowd = len(near)
        if t - self.unseen_hit_t < 1.5:
            crowd = len(near) + 1

        hp_frac = me["hp"] / max(1, me["max_hp"])
        late = obs["alive"] <= 6 or zone["next_radius"] <= 70

        # ---- upgrades
        if obs["offer"]:
            act["pick"] = pick_upgrade(obs["offer"]["choices"], hp_frac < 0.4 and bool(near))

        # ---- weapons we hold
        w0, w1 = me["weapons"][0], me["weapons"][1]
        held = {w["kind"]: i for i, w in enumerate(me["weapons"]) if w}

        # ---- loot: take a better second weapon when standing on one
        kinds_count = {}
        for ks in self.seen_kinds.values():
            for k in ks:
                kinds_count[k] = kinds_count.get(k, 0) + 1
        pref = preferred_second(late, kinds_count)
        # counter-pick: if a recently seen enemy carries the weapon that beats ours, we want something else
        recent = set()
        for mid, ks in self.seen_kinds.items():
            if t - self.last_seen.get(mid, (-99, None))[0] < 60:
                recent.update(ks)
        cur2_kind = me["weapons"][1]["kind"] if me["weapons"][1] else None
        # v4: always push weapons that a recently seen enemy weapon beats to the back
        pref = [k for k in pref if not any(BEATS.get(e) == k for e in recent)] + \
               [k for k in pref if any(BEATS.get(e) == k for e in recent)]
        pref = [k for k in pref if k != "mortar"] + (["mortar"] if "mortar" in pref else [])  # mortar always last
        gather = t < 90 or (me["cubes"] < 5 and t < 180)
        self.gather = gather and not enemies and False  # ghost: no loud gathering and t - self.threat_t > 3 and t - self.unseen_hit_t > 3
        cur2 = w1["kind"] if w1 else None
        for p in obs["pickups"]:
            if dist2(p["pos"], me["pos"]) < self.reach - 0.3 and abs(p["pos"][2] - z) < 4 and p["kind"] in pref:
                if p["kind"] in held:
                    continue
                if cur2 is None:
                    act["take"] = p["id"]
                    break
                if pref.index(p["kind"]) < (pref.index(cur2) if cur2 in pref else len(pref)) and not (w1.get("charge", 0) > 0):
                    act["take"] = p["id"]
                    act["drop"] = 1
                    break

        # ---- choose a target: nearest visible enemy mech
        target = None
        if enemies:
            target = min(enemies, key=lambda m: dist2(me["pos"], m["pos"]))
        tdist = dist2(me["pos"], target["pos"]) if target else 1e9

        # ---- fight or flee
        flee = False
        flee_spot = None
        if target or t - self.unseen_hit_t < 1.5:
            want = True   # ghost: whenever anyone sees us, get out of sight
            if want:
                th = [m["pos"] for m in enemies]
                if t - self.unseen_hit_t < 1.5:
                    th.append([x + 60 * math.cos(self.unseen_bearing), y + 60 * math.sin(self.unseen_bearing), z])
                flee_spot = self.cover_spot(me, th, zone)
                # v3: only run if there is cover to run to; running across open ground = shot in the back (x1.5)
                flee = flee_spot is not None
            self.threat_t = t

        # ---- movement
        move_yaw, speed, boost = None, 1.0, False
        goal = zone["next_center"]
        zone_r = zone["next_radius"]
        in_safe = math.dist([x, y], goal) < 0.5 * zone_r and me["in_zone"]

        threats = [m["pos"] for m in enemies]
        if t - self.unseen_hit_t < 1.5:
            threats.append([x + 60 * math.cos(self.unseen_bearing), y + 60 * math.sin(self.unseen_bearing), z])

        if flee and threats:
            spot = flee_spot
            if spot:
                move_yaw = yaw_to(me["pos"], spot)
            else:
                cx = sum(p[0] for p in threats) / len(threats)
                cy = sum(p[1] for p in threats) / len(threats)
                move_yaw = math.atan2(y - cy, x - cx)
                # bend toward the zone centre if leaving it
                if not me["in_zone"]:
                    move_yaw = yaw_to(me["pos"], goal)
            boost = me["energy"] > 25
        elif target and not flee:
            # duel from our spot: never walk toward them; side-step near the wall so shots miss
            if t - self.strafe_t > 0.8 + (obs["tick"] % 7) * 0.1:
                self.strafe = -self.strafe
                self.strafe_t = t
            ty = yaw_to(me["pos"], target["pos"])
            move_yaw = ty + self.strafe * math.pi / 2
            # hunt mode (early, only lone hunters seen, healthy): go to them; else hold our hiding spot
            hunt = False
            want_r = 35 if "lightning" in held else 80
            if hunt:
                if tdist > want_r + 10:
                    move_yaw = self.steer(me, target["pos"], t) + self.strafe * 0.35
            elif self.hide and dist2(me["pos"], self.hide) > 8:
                move_yaw = self.steer(me, self.hide, t)
            if tdist < 25 and "lightning" not in held:
                move_yaw = ty + math.pi - self.strafe * 0.5   # back off from point-blank
            # v5 peek: while the gun reloads (we can't shoot back), duck behind the nearest wall
            gw = me["weapons"][held["gun"]] if "gun" in held else None
            second_ready = any(w and w["kind"] in ("laser", "lightning") and w["cooldown"] <= 0.3
                               and (w["kind"] != "lightning" or tdist < 45) for w in me["weapons"])
            if gw and gw.get("reload", 0) > 0.3 and not hunt and not second_ready:
                if self.duck is None or t - self.duck_t > 1.0:
                    self.duck = self.cover_spot(me, [target["pos"]], zone)
                    if self.duck and dist2(me["pos"], self.duck) > 12:
                        self.duck = None   # too far to be worth walking in the open
                    self.duck_t = t
                if self.duck:
                    move_yaw = yaw_to(me["pos"], self.duck)
            else:
                self.duck = None
            if not me["in_zone"]:
                move_yaw = yaw_to(me["pos"], goal)
        else:
            # survivor: early quick loot nearby, then sit still at a hiding spot near the next centre
            zkey = (round(goal[0]), round(goal[1]), round(zone_r))
            if self.hide is None or self.hide_zone != zkey or t - self.threat_t < 0.2:
                if self.hide is None or self.hide_zone != zkey or math.dist(self.hide[:2], goal) > 0.55 * zone_r:
                    self.hide = self.pick_hide(me, zone)
                    self.hide_zone = zkey
            dest = None
            # v5: free cubes nearby (kill piles) at any time when no enemy is around
            if not enemies and (me["cubes"] < self.cfg["cubes"]["stat_cap"]
                                or me["cubes"] + 1 in self.cfg["cubes"]["upgrade_thresholds"]):
                cs = [c for c in obs["cubes"] if dist2(me["pos"], c["pos"]) < 40 and abs(c["pos"][2] - z) < 4]
                if cs:
                    dest = min(cs, key=lambda c: dist2(me["pos"], c["pos"]))["pos"]
            if dest is None and self.gather:
                dest = self.loot_target(obs, me, held, pref, zone)
            # v5: supply drop if it is close, landed, uncontested and inside the next circle
            if dest is None and not enemies and t - self.threat_t > 15 and t - self.unseen_hit_t > 15:
                for dr in obs.get("drops", []):
                    if dr["lands_in"] > 0 or (dr["holder"] is not None and dr["holder"] != me["id"]):
                        continue
                    dp = dr["pos"]
                    if dist2(me["pos"], dp) > 80 or math.dist(dp[:2], goal) > zone_r:
                        continue
                    if any(t - ts < 10 and dist2(p, dp) < 60 for (ts, p) in self.last_seen.values()):
                        continue
                    dest = dp if dist2(me["pos"], dp) > dr["radius"] - 3 else None
                    if dest is None:
                        dest = [x, y, z]   # already inside the ring: stay
                    break
            if dest is None:
                dest = self.hide if self.hide else goal
            # v5: final circle: stand at its exact centre
            if zone_r <= 1 and dest is not None and math.dist(dest[:2], zone["center"]) > zone["radius"] - 4:
                dest = [goal[0], goal[1], z]
            if dist2(me["pos"], dest) > 2.0:
                move_yaw = self.steer(me, dest, t)
            boost = (not me["in_zone"]) and me["energy"] > 20
            if not me["in_zone"]:
                boost = me["energy"] > 20
        # ---- stuck handling: commanded to move but not moving
        speed_now = math.hypot(me["vel"][0], me["vel"][1])
        if move_yaw is not None and speed_now < 2.0 and me["grounded"]:
            self.stuck_t += 0.1
        else:
            self.stuck_t = max(0.0, self.stuck_t - 0.1)
        if self.stuck_t > 0.6 and move_yaw is not None:
            act["jump"] = True
            boost = True
            move_yaw += 0.8 * self.strafe
            if self.stuck_t > 2.5:
                self.strafe = -self.strafe
                self.stuck_t = 0
        if not me["grounded"] and self.stuck_t > 0 and me["energy"] > 10:
            boost = True  # jet over the obstacle

        if move_yaw is not None:
            act["move"] = [speed * math.cos(move_yaw), speed * math.sin(move_yaw)]
        else:
            act["move"] = [0, 0]
        act["boost"] = bool(boost)

        # ---- aim and fire
        muzzle = [x, y, z + me["stats"].get("muzzle_height", 7)]
        fire = [False, False]
        aim_target = target
        if self.gather and aim_target is None and npcs and not flee:
            n = min(npcs, key=lambda m: dist2(me["pos"], m["pos"]))
            if dist2(me["pos"], n["pos"]) < 60:
                aim_target = n
        if aim_target:
            gun_aim = aim_angles(muzzle, lead_point(muzzle, aim_target, self.gun_speed))
            act["aim"] = gun_aim
            d = dist2(me["pos"], aim_target["pos"])
            err = abs(wrap(me["aim"][0] - gun_aim[0]))
            perr = abs(me["aim"][1] - gun_aim[1])   # v6: pitch must be on target too
            tol = math.atan2(3.5, max(d, 1)) + 0.01   # v3: mech radius 2.5 m + spread; old 2 m was too strict
            los = self.map.clear(muzzle, [aim_target["pos"][0], aim_target["pos"][1], aim_target["pos"][2] + 4.5])
            for i, w in enumerate(me["weapons"]):
                if not w:
                    continue
                k = w["kind"]
                if k == "gun":
                    fire[i] = (los and err < tol + 0.01 and perr < tol + 0.03
                               and d < w["stats"]["range"] and w.get("ammo", 1) > 0)
                elif k == "lightning":
                    cone = math.radians(w["stats"].get("cone_deg", 30) / 2)
                    blocker = any(o is not aim_target and dist2(me["pos"], o["pos"]) < d
                                  and abs(wrap(yaw_to(me["pos"], o["pos"]) - me["aim"][0])) < cone
                                  for o in (npcs + obs["crates"]))
                    # v5: lightning only for mechs; drones and crates get bullets
                    fire[i] = (aim_target["kind"] == "mech" and d < w["stats"]["range"] - 2
                               and err < math.radians(12) and me["energy"] > 10 and not blocker)
                elif k == "laser":
                    ch = w.get("charge", 0)
                    full = ch >= w["stats"]["charge"] - 0.05
                    aim_ok = abs(wrap(me["aim"][0] - yaw_to(muzzle, aim_target["pos"]))) < tol
                    shorted = any("lightning" in self.seen_kinds.get(m["id"], []) and dist2(me["pos"], m["pos"]) < 50 for m in enemies)
                    if ch > 0:
                        fire[i] = not (full and aim_ok and los)   # keep holding until ready, then release
                    else:
                        close = any(dist2(me["pos"], m["pos"]) < 30 for m in enemies)   # v4: never start charging up close
                        fire[i] = los and d < w["stats"]["range"] and w["cooldown"] <= 0 and not shorted and not flee and not close
            # mortar wants its own pitch only when the gun isn't shooting
            gun_busy = "gun" in held and (me["weapons"][held["gun"]].get("reload", 0) > 0.5
                                          or me["weapons"][held["gun"]].get("ammo", 1) == 0)
            if "mortar" in held and (gun_busy or "gun" not in held):
                mi = held["mortar"]
                w = me["weapons"][mi]
                if w["cooldown"] <= 0.2:
                    st = w["stats"]
                    tp = lead_point(muzzle, aim_target, st["speed"] * 0.7)
                    pitch = mortar_pitch(math.dist(muzzle[:2], tp[:2]), tp[2] - 4.5 - muzzle[2], st["speed"], st["gravity"])
                    if pitch is not None and d > 18:
                        act["aim"] = [gun_aim[0], pitch]
                        fire[mi] = err < 0.05 and abs(me["aim"][1] - pitch) < 0.05
        elif t - self.unseen_hit_t < 1.5:
            act["aim"] = [self.unseen_bearing, 0.0]     # face whoever is hitting us
        elif self.sounds:
            s = max(self.sounds, key=lambda s: s[0])
            act["aim"] = [s[1], 0.0]                    # look toward the latest fight noise
        elif move_yaw is not None:
            act["aim"] = [move_yaw, 0.0]
        else:
            act["aim"] = [wrap(self.open_dir(me) + 0.7 * math.sin(t * 0.8)), 0.0]  # v5: watch the open side

        # crates: shoot them when nothing else to do
        if self.gather and aim_target is None and not flee and obs["crates"] and "aim" in act:
            c = min(obs["crates"], key=lambda c: dist2(me["pos"], c["pos"]))
            if dist2(me["pos"], c["pos"]) < 60:
                ca = aim_angles(muzzle, [c["pos"][0], c["pos"][1], c["pos"][2] + 1.2])
                act["aim"] = ca
                if "gun" in held and abs(wrap(me["aim"][0] - ca[0])) < 0.05:
                    fire[held["gun"]] = me["weapons"][held["gun"]].get("ammo", 1) > 0

        # reload when quiet
        g = me["weapons"][held["gun"]] if "gun" in held else None
        if g and aim_target is None and g.get("ammo", 12) < g["stats"]["mag"] * 0.5 and g.get("reload", 0) == 0:
            act["reload"] = True

        # ghost: only shoot when cornered (no cover found) or we were hit in the last 3 s
        if not (not flee or t - self.last_hit_t < 3):
            fire = [False, False]
        act["fire"] = fire
        return act

    def can_win(self, me, en):
        """Time-to-kill comparison (simple, gun-centred)."""
        dps = {"gun": 100, "laser": 130, "lightning": 200, "mortar": 60}
        d = dist2(me["pos"], en["pos"])

        def side_dps(kinds, cubes):
            total = 0
            for k in kinds:
                if k == "lightning" and d > 45:
                    continue
                total += dps.get(k, 0)
            return total * (1 + 0.04 * min(cubes, 10))

        mine = side_dps([w["kind"] for w in me["weapons"] if w], me["cubes"])
        theirs = side_dps(en.get("weapons", ["gun"]), en.get("cubes", 0))
        their_hp = en.get("hp", 1400)
        # v3: assume equal hit rates; fight unless clearly weaker (old 1.2x margin made us flee every even fight)
        ttk_us = their_hp / max(mine * 0.55, 1)
        ttk_them = me["hp"] / max(theirs * 0.55, 1)
        return ttk_them >= 0.8 * ttk_us

    def loot_target(self, obs, me, held, pref, zone, near_only=False):
        best, bd = None, 1e9
        # cubes in sight (up to the stat cap, or to the next upgrade threshold)
        cap = self.cfg["cubes"]["stat_cap"]
        if me["cubes"] < cap or me["cubes"] + 1 in self.cfg["cubes"]["upgrade_thresholds"]:
            for c in obs["cubes"]:
                d = dist2(me["pos"], c["pos"])
                if d < (60 if near_only else 100) and d < bd:
                    best, bd = c["pos"], d
        if best:
            return best
        # a weapon we want
        cur2 = me["weapons"][1]["kind"] if me["weapons"][1] else None
        for pid, (k, p) in list(self.loot.items()):
            if k in held or k not in pref:
                continue
            if cur2 and pref.index(k) >= (pref.index(cur2) if cur2 in pref else len(pref)):
                continue
            if abs(p[2] - me["pos"][2]) > 6:
                continue  # on a roof: skip for now
            d = dist2(me["pos"], p)
            if d < (60 if near_only else 120) and d < bd and math.dist(p[:2], zone["next_center"]) < zone["next_radius"]:
                best, bd = p, d
        if best:
            return best
        # crates nearby: walk into gun range
        for c in ([] if near_only else obs["crates"]):
            d = dist2(me["pos"], c["pos"])
            if 40 < d < 90 and d < bd and abs(c["pos"][2] - me["pos"][2]) < 6:
                best, bd = c["pos"], d
        return best


mf.run(Bot)
