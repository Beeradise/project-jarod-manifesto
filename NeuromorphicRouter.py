z_bar = gate.seed_z_bar()                      # PRE-state: decided by the state BEFORE this event
z_best, gidx = <argmax graded_z(v, g["acc"]) over ganglia>
predicted = nearest["ev"].predicted(idx)

if z_best is None or z_best < z_bar:
    # SEED — novel enough: mint a new template (episodic capture; evidence floor may kill it later)
    ganglia.append(_new_ganglion(v, text, idx, now, channel))
    # ... evict weakest by (evidence, recency) if over MAX_GANGLIA_PER_CHANNEL (64)
    band = "seed"
elif z_best >= JOIN_Z:                          # 105 — clearly covered
    # JOIN — accumulate into the template (with the F4 viscoelastic stress/thinning step)
    nearest["acc"] = sat_add(nearest["acc"], v)        # (thinned if stressed)
    nearest["n"] += 1
    nearest["ev"].on_join(idx, ng.kernel_weight(z_best))
    band = "join"
else:
    # AMBIENT (z_bar <= z_best < JOIN_Z) — recognition only, NO accumulator write
    nearest["ev"].on_recognition(idx, ng.kernel_weight(z_best))
    band = "ambient"

gate.observe(z_best, predicted)                 # advance the spiral (post-routing)
for g in ganglia: g["ev"].decay_event()         # per-event evidence decay
