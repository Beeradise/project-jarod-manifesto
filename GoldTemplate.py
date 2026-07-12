def on_join(self, idx, w):
    """A join at kernel weight w: accrue evidence w, log a prequential arrival."""
    self.e += w
    return self._on_arrival(idx)

def on_recognition(self, idx, w):
    """Ambient recognition: accrue RECALL_BUMP*w (a near-miss still counts as evidence + recurrence,
    without writing into the template's accumulator). This is 'reads loop back as writes'."""
    self.e += RECALL_BUMP * w
    return self._on_arrival(idx)

def predicted(self, idx):
    """True iff idx falls in the predicted next-arrival window (armed after >= 2 arrivals).
    Window = [last + ema_gap*(1-TOL), last + ema_gap*(1+TOL) + MIN_SLACK] — the Screeps drop-window idea."""
    if self.arrivals < 2 or self.ema_gap is None or self.last_idx is None:
        return False
    lo = self.last_idx + max(1.0, self.ema_gap * (1.0 - IA_TOL))
    hi = self.last_idx + self.ema_gap * (1.0 + IA_TOL) + IA_MIN_SLACK
    return lo <= idx <= hi
