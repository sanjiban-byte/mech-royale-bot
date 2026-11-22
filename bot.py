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
UPGRADE_ORDER = ["armor", "plating", "nanites", "gun_damage", "gun_rate",
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
    name = "sanjiban-v1"

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
        for e in obs["events"]:
            if e["type"] in ("took",) or (e["type"] == "take_failed" and e["reason"] == "gone"):
                self.loot.pop(e["id"], None)
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
        crowd = len(near) + len(extra)
        if t - self.unseen_hit_t < 1.5:
            crowd = max(crowd, len(near) + 1)

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
        cur2 = w1["kind"] if w1 else None
        for p in obs["pickups"]:
            if dist2(p["pos"], me["pos"]) < self.reach - 0.3 and abs(p["pos"][2] - z) < 4 and p["kind"] in pref:
                if p["kind"] in held:
                    continue
                if cur2 is None:
                    act["take"] = p["id"]
                    break
                if pref.index(p["kind"]) < pref.index(cur2) and not (w1.get("charge", 0) > 0):
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
        if target or t - self.unseen_hit_t < 1.5:
            if hp_frac < 0.4 or crowd >= 2:
                flee = True
            elif target and not self.can_win(me, target):
                flee = True
        if flee:
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
            spot = self.cover_spot(me, threats, zone)
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
            # duel: hold a comfortable range and strafe so leading shots miss us
            want = 30 if "lightning" in held else 90
            if t - self.strafe_t > 0.8 + (obs["tick"] % 7) * 0.1:
                self.strafe = -self.strafe
                self.strafe_t = t
            ty = yaw_to(me["pos"], target["pos"])
            if tdist > want + 15:
                move_yaw = ty + self.strafe * 0.6
            elif tdist < want - 15:
                move_yaw = ty + math.pi - self.strafe * 0.6
            else:
                move_yaw = ty + self.strafe * math.pi / 2
            if not me["in_zone"]:
                move_yaw = yaw_to(me["pos"], goal)
        else:
            # no fight: loot nearby, else head to the safe middle of the next circle
            dest = None
            if not late or in_safe:
                dest = self.loot_target(obs, me, held, pref, zone)
            if dest is None and not in_safe:
                dest = goal
            if dest is not None:
                move_yaw = yaw_to(me["pos"], dest)
                if dist2(me["pos"], dest) < 1.0:
                    move_yaw = None
            boost = (not me["in_zone"]) and me["energy"] > 20
            if not me["in_zone"] or zone["dps"] >= 60 and math.dist([x, y], zone["center"]) > zone["radius"] * 0.85:
                move_yaw = yaw_to(me["pos"], goal)
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
        if aim_target is None and npcs and not flee:
            n = min(npcs, key=lambda m: dist2(me["pos"], m["pos"]))
            if dist2(me["pos"], n["pos"]) < 60:
                aim_target = n
        if aim_target:
            gun_aim = aim_angles(muzzle, lead_point(muzzle, aim_target, self.gun_speed))
            act["aim"] = gun_aim
            d = dist2(me["pos"], aim_target["pos"])
            err = abs(wrap(me["aim"][0] - gun_aim[0]))
            tol = math.atan2(2.0, max(d, 1))
            los = self.map.clear(muzzle, [aim_target["pos"][0], aim_target["pos"][1], aim_target["pos"][2] + 4.5])
            for i, w in enumerate(me["weapons"]):
                if not w:
                    continue
                k = w["kind"]
                if k == "gun":
                    fire[i] = los and err < tol + 0.01 and d < w["stats"]["range"] and w.get("ammo", 1) > 0
                elif k == "lightning":
                    fire[i] = d < w["stats"]["range"] - 2 and err < math.radians(12) and me["energy"] > 10
                elif k == "laser":
                    ch = w.get("charge", 0)
                    full = ch >= w["stats"]["charge"] - 0.05
                    aim_ok = abs(wrap(me["aim"][0] - yaw_to(muzzle, aim_target["pos"]))) < tol
                    shorted = any("lightning" in self.seen_kinds.get(m["id"], []) and dist2(me["pos"], m["pos"]) < 50 for m in enemies)
                    if ch > 0:
                        fire[i] = not (full and aim_ok and los)   # keep holding until ready, then release
                    else:
                        fire[i] = los and d < w["stats"]["range"] and w["cooldown"] <= 0 and not shorted and not flee
            # mortar wants its own pitch only when the gun isn't shooting
            if "mortar" in held and not ("gun" in held and fire[held["gun"]]):
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
            act["aim"] = [wrap(me["aim"][0] + 0.6), 0.0]  # slow look-around when idle

        # crates: shoot them when nothing else to do
        if aim_target is None and not flee and obs["crates"] and "aim" in act:
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
        ttk_us = their_hp / max(mine * 0.6, 1)        # assume we hit ~60% (leading)
        ttk_them = me["hp"] / max(theirs * 0.5, 1)
        return ttk_them > 1.2 * ttk_us

    def loot_target(self, obs, me, held, pref, zone):
        best, bd = None, 1e9
        # cubes in sight (up to the stat cap, or to the next upgrade threshold)
        cap = self.cfg["cubes"]["stat_cap"]
        if me["cubes"] < cap or me["cubes"] + 1 in self.cfg["cubes"]["upgrade_thresholds"]:
            for c in obs["cubes"]:
                d = dist2(me["pos"], c["pos"])
                if d < 80 and d < bd:
                    best, bd = c["pos"], d
        if best:
            return best
        # a weapon we want
        cur2 = me["weapons"][1]["kind"] if me["weapons"][1] else None
        for pid, (k, p) in list(self.loot.items()):
            if k in held or k not in pref:
                continue
            if cur2 and pref.index(k) >= pref.index(cur2):
                continue
            if abs(p[2] - me["pos"][2]) > 6:
                continue  # on a roof: skip for now
            d = dist2(me["pos"], p)
            if d < 120 and d < bd and math.dist(p[:2], zone["next_center"]) < zone["next_radius"]:
                best, bd = p, d
        if best:
            return best
        # crates nearby: walk into gun range
        for c in obs["crates"]:
            d = dist2(me["pos"], c["pos"])
            if 40 < d < 90 and d < bd and abs(c["pos"][2] - me["pos"][2]) < 6:
                best, bd = c["pos"], d
        return best


mf.run(Bot)
