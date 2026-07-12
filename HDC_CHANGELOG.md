# HDC Neuromorphic Store — Changelog

A log of known past actions on the foreman HDC memory stack (`step01-hdc_core.py`,
`novelty_gate.py`, the BEAGLE encoder, and their tools). Newest first.

**Scope:** the foreman *store* — episodic pages + decentralized per-channel ganglia, decided
by graded `z`. Sibling neuromorphic stacks (Screeps `vault.js`, the Timberborn/Minecraft
NeuroCore) have their own histories and are not covered here, except where they are the
reference (see *Lineage*).

**Sourcing / caveat:** there is no git on this tree. This log is reconstructed from the era
markers in the source (`F1`…`F10`), the dated `rollback\` / `attic\` snapshots, and session
memory. Dates are the rollback-dir timestamps where one exists, else the ratification date in
memory. The early F-numbers (F1–F3) consolidated rapidly during 2026-07-09→10 and are not each
separately snapshotted; F2/F3 have no distinct surviving marker.

---

## Current state (2026-07-12, after F10)

| thing | value |
|---|---|
| Graded space `D_g` | 131072 int8 (read-write, all decisions) |
| Binary space `D_b` | 16384 (page-format derivation only, 8:1 fold — never a decision input) |
| Schema versions | pages `STORE_VERSION=4`, `GANGLIA_VERSION=8`, `ENCODER_VERSION=1` |
| Channels | `telemetry` (default), `chat`, `content` — routing/decay/coalesce never cross; cap 64/channel |
| Decision statistic | `graded_z = dot / sqrt(var)`; null ~N(0,1) (max ~3.7 over unrelated pairs) |
| Routing bands | `z < seed_z_bar` → **SEED**; `seed_z_bar ≤ z < join` → **HOOK**; `z ≥ join` → **JOIN** |
| Seed bar | `46 + 19·openness` (`Z_BAR_TIGHT=46`, `Z_BAR_LOOSE=65`), oscillator-driven |
| Join thresholds | `JOIN_Z=105`; per-channel `CHANNEL_JOIN_Z={content:130}` |
| Coalesce (sleep) | mutual-min `z ≥ COALESCE_Z=65` |
| Hooks | `HOOKS_MAX=6`, reinforce nearest at `HOOK_TIGHT_Z=90` |
| Dominance cap | `DOMINANCE_FRAC=0.4` past `DOMINANCE_MIN_MASS=12` → JOIN diverts to hook |
| Evidence death | floor `E_FLOOR=0.05` (n>1), `0.2` (singleton); decay 0.999/event, 0.99/sleep |
| Feedback | good `e+2.0`, bad `e×0.25`, gate `FEEDBACK_MIN_Z=55` (= recall floor) |
| Encoder | BEAGLE composite: semantic `sign(R·(embed−mu))` (nomic-embed-text 768-d, R seed `0xF9BEA61E`, mu from `BASELINE_WORDS`) + order bigrams + formation char-trigrams, `W=1/1/1` |
| Viscoelastic | stress EMA `α=0.2`, onset `T0=0.35`, `ETA_MAX=0.875` (yield floor 0.125), sleep recovery ×0.5 |
| Gate (frozen) | 2nd-order underdamped oscillator, period 32, ζ=0.15, amp floor 0.10, base 0.15→0.90, k_age 250 |

**Files:** `step01-hdc_core.py` (store + encoder), `novelty_gate.py` (oscillatory gate + evidence
stack — **frozen** by mandate), `rebuild_store_f9.py` (replay/rebuild), `rebuild_ganglia.py` (F7
iterator library), `hdc_map.py` (heatmaps), `f10_dissolve_content.py`, `test_hdc_core.py`.
**State (`hdc_pages\`):** `ganglia.npz` (v8), `page-*.npz` (v4), `encoder.npz` (mu+digest pin),
`embed_cache.npz`, `beagle_proj.npy` (regenerable from seed).

---

## 2026-07-12 — F10: Hooks restored + dominance cap + content dissolved

The central correction. Diagnosis (confirmed read-only on the live store): "learning stuck"
was **not** a weight-tune — F7's retirement of hooks was a regression vs the working Screeps
vault. The retired **AMBIENT band discarded variants** rather than hooking them, so **142/224
content events (63%) were silently lost**, and unbounded saturating `graded_z` grew one **n=74
hub holding 90%** of content that monopolized recall (every query returned the nuclear hub,
below the floor).

- **Hooks** — the AMBIENT band now `_attach_hook`s the variant onto the nearest ganglion
  (Screeps `vaultRememberFuzzy` middle band, ported to graded-z/int8): `HOOKS_MAX=6`, reinforce
  nearest at `HOOK_TIGHT_Z=90`, evict weakest `(n,tick)`. `recall` refines/lifts a result via
  its hooks (`via_hook` field); `sleep` coalesce merges hooks; hooks persist as parent-indexed
  flat arrays (`h_parent`/`h_acc`/…). `cat→{tail,orange,long-hair}` is representable again.
- **Dominance cap** — a JOIN onto a ganglion already holding `>DOMINANCE_FRAC=0.4` of channel
  mass (after `DOMINANCE_MIN_MASS=12`) is diverted to a hook. No more black holes.
- **Per-channel join** — `CHANNEL_JOIN_Z={content:130}`: content variants (measured mutual z
  68–119) HOOK, only near-exact JOIN; telemetry/chat keep 105 (near-dup JSON *should* fuse).
- **Schema** `GANGLIA_VERSION 7→8`, with a v7→v8 **in-place migration** that preserves
  telemetry+chat. **Content channel dissolved** per operator (re-entering by hand, diverse).
- `novelty_gate.py` unchanged. `foreman.py memory` now shows a `hooks=` column + `+hook:` lines.
- Verified: `verify_hooks.py` (isolated) — diverse→seed, cat variants→1 concept+hooks (nothing
  lost), recall via hook (z61 vs parent 13), dominance frac 0.42, hooks byte-identical on
  save/load. Test suite `test_hdc_core.py` updated for the new band/schema.
- **Rollback:** `rollback\f10-20260712-115151\` (pre-F10 hdc_core + novelty_gate + v7 ganglia.npz).
- **Open (F11 candidate):** foreman still lacks role-filler binding (`ATTR⊗VAL`) — the
  compositional structure that gives Screeps its calibrated, drift-free distance.

## 2026-07-11 — F9: BEAGLE semantic encoder (LIVE ~18:35)

Replaced the flat bag-of-random-word-atoms text encoder (semantically blind: synonyms scored
z≈0, `cat`≡noise vs `feline`; order-blind) with a **BEAGLE composite** (Jones & Mewhort 2007):
semantic plane `sign(R·(embed−mu))` (nomic-embed-text 768-d, frozen Rademacher R @ seed
`0xF9BEA61E`, mean-centered on `BASELINE_WORDS` to kill embedding anisotropy) + order plane
(permutation-bound bigrams) + formation plane (char-trigrams), `W=1/1/1`, then `sign`.

- Synonyms 0→≈115; disjoint-vocab paraphrase injects on pure semantics; discrimination holds.
- Constants remapped by measurement: `JOIN_Z 100→105`, `COALESCE_Z/Z_BAR_LOOSE 45→65`,
  `Z_BAR_TIGHT 6→46` (English-null max 45 is the operational null now), `KERNEL 50/4.5`,
  `FEEDBACK_MIN_Z / recall floor 45→55`, `UNION_STOP_RATIO 0.08` (applied at write time too),
  rag `DEFAULT_MIN_Z 10→25`. `STORE_VERSION 3→4`, `GANGLIA_VERSION 6→7`.
- New schema invariants: frozen projection + resident embedder; per-token embed on write
  (cached in `embed_cache.npz`; `encoder.npz` pins mu + model digest).
- Replay tool `rebuild_store_f9.py`. Old-history backfill deferred (budget). **Rollback:**
  `rollback\f9-20260711-183425\` (store data byte-identical; caveat: pre-F9 *source* not captured).

## 2026-07-11 — Read-path recalibration ("everything looks familiar")

Recall was matching everything: pages (raw bow, no stops) scored unrelated English 83–94, and
channel-blind function words concentrated in one English-bearing telemetry hub. Fix was
**read-side only** (write path untouched, reversible): `_recall_stop()` = union of every mature
(n≥20) channel's DF-stop set, applied to recall + feedback queries; floors recalibrated with
English-unrelated controls to 45 (later 55 at F9).

## 2026-07-10 — F8: Operator-verdict feedback (`/good` `/bad`)

`HDCStore.feedback(text, positive)` — grade remembered info; targets recall's exact top-1
ganglion above the floor. GOOD `e+=2.0` + recency refresh; BAD `e×=0.25`, no refresh; death via
the existing sleep-floor machinery (a bad-docked ganglion that live traffic keeps recognizing
re-accrues evidence — opinion argues *with* the evidence stream). Surfaces `foreman feedback`,
GUI `/good` `/bad`.

## 2026-07-10 — F7: LIVE cutover to pure-graded ganglia (**hooks retired**)

The live store became the pure-graded decentralized-ganglia model proven in F5/F6 shadow. ART
template/hook routing **deleted**; every decision runs on one statistic (graded z). `_remember`
= argmax-z routing (SEED / JOIN / **AMBIENT**); `sleep` = per-dirty-channel kick → decay →
mutual-min-z coalesce → real death at the evidence floor → stress recovery → DF recompute.
Schema v6 `ganglia.npz`; `templates.npz` retired. Migration by deterministic replay.
**This is the change F10 later corrected** — "binary/hooks offer nothing" dropped the
associative variant layer, and the AMBIENT band that replaced hooks discarded variants.
**Rollback:** `rollback\f7-20260710-155339\`.

## 2026-07-10 — F6: Pure-graded shadow

"Binary offers nothing — get away from it completely." All decisions moved to graded z; the
binary fold deleted from every decision path (survives only as page write-side format).
Same-corpus A/B vs F5 exact.

## 2026-07-10 — F5: Decentralized ganglia consolidation (shadow)

Redesign of template *formation*: sleep derives semantic structure by replaying raw run logs;
age-gated novelty with an oscillatory (2nd-order underdamped, "sine/cosine in a spiral") gate
that *breathes* — periodically re-permissive forever (amp floor) so the store can never
ossify. Evidence stacks (logistic kernel, count-weighted, decaying, recall-bumped) + per-ganglion
recurrence prediction. Shipped `novelty_gate.py`, `shadow_consolidate.py`.

## 2026-07-10 — F4: Viscoelastic hardening

Per-ganglion smoothed **stress** scalar (EMA of the opposing-sign fraction on JOIN). Sustained
conflicting joins stiffen the accumulator via η lane-thinning with a 0.125 yield floor (never
fully frozen); sleep recovers stress ×0.5 (asymmetric). Operator's non-Newtonian / memory-foam
concept, designed by the pipeline itself and carried into the ganglia store.

## 2026-07-09/10 — F1: Channels

Every ganglion carries a `channel` string; routing, per-write decay, spawn cap/eviction, and
sleep coalescence never cross a channel boundary. Three fixed channels
(`telemetry` | `chat` | `content`) — "a zone where memory TYPES must never cross-match." Recall
stays cross-channel (read-only association).

## 2026-07-09 — v3 store (pre-F baseline)

Graded-trunk / binary-bark hybrid: `D_g=131072` / `D_b=16384`, ART golden templates + episodic
pages, per-role endpoints. The substrate the F-series then reshaped.

---

## Lineage

The direct ancestor of the hook mechanism is the **Screeps colony-AI vault**
(`…\Screeps\…\default\vault.js`, `hdc.js`): ART golden templates with up-to-6 **hooks** each,
on a *bounded* Hamming metric (fixed tight 7% / novelty 20% thresholds), plus role-filler
`BIND(ATTR,VAL)` encoding. F10 restored its hook discipline; the still-open gap (F11) is its
role-filler binding. The Timberborn/Minecraft **NeuroCore** (C#) is a sibling stack with its own
recall/sleep/plasticity spec.
