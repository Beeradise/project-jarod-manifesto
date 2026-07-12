def seed_z_bar(self):
    """Seed BAR in graded z. openness 1 -> Z_BAR_LOOSE (65, near-free seeding);
    openness 0 -> Z_BAR_TIGHT (46, only near-null events seed — but they ALWAYS do)."""
    return self.Z_BAR_TIGHT + (self.Z_BAR_LOOSE - self.Z_BAR_TIGHT) * self.openness()

def _surprise(self, z_best):
    """Drive in [0,1]: coverage quiets, orphans open. None (no ganglion yet) -> 1.0."""
    if z_best is None:            return 1.0
    if z_best >= self.JOIN_Z:     return 0.0
    return min(1.0, max(0.0, (self.JOIN_Z - z_best) / self.JOIN_Z))

def observe(self, z_best, predicted):
    """Advance the spiral by ONE event (semi-implicit Euler: v first, then x, then clamp)."""
    u = self._surprise(z_best) * (1.0 - self.PRED_DISCOUNT * (1.0 if predicted else 0.0))
    self.v = self.v + self.W0 ** 2 * (u - self.x) - 2.0 * self.OSC_ZETA * self.W0 * self.v
    self.x = min(self.X_RAIL, max(-self.X_RAIL, self.x + self.v))
    self.age += 1
    return {"u": u, "x": self.x, "v": self.v, "openness": self.openness(), "z_bar": self.seed_z_bar()}

def on_sleep(self):
    """A sleep RE-OPENS the gate: inject velocity kick amp(age)*WD (floored, so even an old store breathes)."""
    self.v = self.v + self.SLEEP_KICK_GAIN * self._amp() * self.WD
    self.sleeps += 1
