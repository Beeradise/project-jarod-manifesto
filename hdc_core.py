r"""HDC memory core — pure-graded decentralized-ganglia store (F7, operator-ratified).

F7 LIVE CUTOVER (operator "go" 2026-07-10): the template/hook ART store is REPLACED by
the pure-graded ganglia model proved in shadow (F5/F6). Every DECISION runs on ONE
statistic — graded z. The derived binary space leaves every decision path; it survives
ONLY as page-FORMAT write-side derivation (fold_vec inside _save_page). "Binary offers
nothing to the architecture" (ratification): the fold added no ranking capability (12.5x
less capacity) and only ever earned its keep as a prescan whose compute saving is not
load-bearing at this scale.

Two representations of one memory — now ONE substrate for decisions:

  * GRADED (canonical, read-write): a D_g=131072 int8 space. All learning happens here as
    SATURATING accumulator addition (dot-product similarity, ranked by graded z). Ganglion
    accumulators and page accumulators both live here.
  * BINARY (derived, PAGE-FORMAT ONLY): a D_b=16384 space produced by an 8:1 tiebroken
    fold. Pages keep STORE_VERSION 3 byte-compatible (exact_fold/bow_fold columns) so old
    pages stay loadable and new pages stay format-identical for any external reader —
    fold_vec is a WRITE-SIDE derivation only, NEVER a decision input (no fold READ path,
    no fold distance metric, no fold-z anywhere).

Complementary learning systems (NeuroCore §1):

  * PAGES (episodic record): one page per run, dual graded accumulators — exact_g
    (membership/dedup, hash-orthogonal) and bow_g (lexical similarity substrate). Capped at
    400 events. The raw, unconsolidated timeline. UNCHANGED by F7 (operator scope: pages
    are untouched).
  * GANGLIA (semantic gist): decentralized per-channel accumulators, each with an evidence
    stack (novelty_gate.EvidenceStack) and a prequential inter-arrival ledger. New events
    ROUTE by argmax graded z under an oscillatory novelty gate (novelty_gate.NoveltyGate):
      - JOIN  (z >= JOIN_Z): accumulate into the ganglion (accumulator addition), evidence
        += kernel_weight(z), log a prequential arrival.
      - SEED  (z below the gate's seed bar, or no ganglion yet): a new ganglion is born
        (episodic capture; the evidence floor kills genuine singleton noise later).
      - AMBIENT (in between): recognition only — evidence += RECALL_BUMP*kernel_weight(z),
        log an arrival, but NO accumulator write ("the field looping back into the center").
    Sleep decays evidence, coalesces mutually-near ganglia, and applies REAL death below the
    evidence floor. Hooks are RETIRED — the ambient band + evidence graduation replace them.

LEXICAL HONESTY (G2): bow recall is LEXICAL-OVERLAP ranking, not semantic search;
per-token signal dilutes with both token count and page fill. It answers "which run talked
about words like these", not "which run meant the same thing". Ganglia cluster in a
DF-filtered CONSOLIDATION space (boilerplate + digits dropped) so distinct memory kinds
separate instead of collapsing into a boilerplate mega-cluster.

int8 SATURATION IS A FEATURE (G3): a railed component (+-127) stops accumulating in its
direction — stickiness emerging from the substrate. A railed component still moves 1 step
per sustained OPPOSING write (127->126), so mature accumulators are noise-immune yet not
frozen. Plasticity is now SATURATION DYNAMICS + REAL DEATH (the successor to G7): ganglion
accumulators are NEVER decayed (F6 parity); an under-evidenced ganglion is REMOVED at sleep
rather than silently bled toward zero.

CAPACITY (G4): graded exact-membership 5-sigma capacity ~ D_g/25 ~= 5243 events, so the
400-event page cap is deeply safe (z ~= sqrt(D_g/400) ~= 18 at cap). Graded ganglion
routing: null z ~ N(0,1) (measured max 3.70 over 4032 unrelated pairs); family z ~90-362;
ceiling sqrt(D_g) = 362.04.

SINGLE-WRITER ASSUMPTION (A1/A3): exactly ONE foreman process writes HDC state at a time
(the GUI busy-locks overlapping runs; the CLI is one-shot). Two concurrent writers risk
page-seq races and ganglia last-writer-wins. Accepted, documented; the F7 cutover ran the
single-writer deploy check before the file swap.

CHANNELS (F1): every ganglion carries a `channel` string; routing, per-write evidence
decay, the spawn cap/eviction, and the sleep coalescence never cross a channel boundary.
Three fixed channels ("telemetry" | "chat" | "content", DEFAULT_CHANNEL = "telemetry")
realize NeuroCore §5 "a zone where memory TYPES must never cross-match". recall() stays
CROSS-CHANNEL (read-only association is the point) and tags each result with its channel.
Ganglia persist as schema v6 in a NEW file hdc_pages\ganglia.npz (never a v6 templates.npz
— the old file can then never be half-read by new code; the rebuild IS the migration). A
stray templates.npz is retired to .pre-f7.bak at __init__, never read. PAGES stay
STORE_VERSION 3.

VISCOELASTIC HARDENING (F4, graded-native carry): each ganglion carries a smoothed stress
scalar in [0,1] that stiffens its accumulator against a stream of CONFLICTING joins.
Instantaneous stress = the OPPOSING-SIGN fraction of the joining vector vs the PRE-write
accumulator; EMA-smoothed (STRESS_ALPHA=0.2) on the JOIN path only. A deadband ramp maps
stress to a step reduction: eta = 1 - STRESS_ETA_MAX*ramp, floored at the 0.125 YIELD FLOOR
(a fully stressed ganglion still absorbs ~1/8 of every join, deterministically lane-thinned
via a sha256-seeded mask — process-stable). Sleep recovers stress *STRESS_RECOVERY (0.5/
night, asymmetric fast recovery); coalescence keeps max(stress) of the pair. The old
fold-space stress gate shift RETIRES with the reinforce-vs-hook gate — its unit was fold
sigma and that gate no longer exists; the ganglia model's protection is
eta-thinning plus the ambient band (a conflicted near-miss lands ambient and never deforms
the accumulator — the "divert IS the protection" principle). HONEST NOTE: STRESS_T0=0.35
was calibrated on RAW bow vs trunk protos; DF-filtered join traffic may idle at a different
opposing fraction — watch the now-visible stress column in `foreman.py memory` (the
"watch the 0.35-onset assumption"); T0 is the knob, no code change needed.

CONSTANTS PROVENANCE: JOIN_Z/COALESCE_Z/Z_BAR/KERNEL/E_FLOOR/DF_STOP_RATIO and the F4
stress knobs were calibrated on the 2026-07-10 corpus (F6 probes P1-P3) and are ratified
defaults. RE-MEASURABLE post-cutover via the cmd_memory stress/e columns, the sleep summary
(died/merged/df_stop counts), and the rebuild report: JOIN_Z, COALESCE_Z/Z_BAR_LOOSE, KERNEL
45/10, STRESS_T0 (the flagged 0.35-onset watch), the E_FLOOR pair, DF_STOP_RATIO. Every
constant with a documented disable knob (=0 semantics) lives beside its definition.
"""

import numpy as np
import os
import sys
import re
import time
import json
import atexit
import hashlib
import functools
import importlib.util
import urllib.request
from typing import List, Optional

# ---------------------------------------------------------------- geometry (G1)
DIMENSION_G = 131072          # graded canonical space (int8, read-write)
DIMENSION_B = 16384           # binary derived space (uint8 0/1, PAGE-FORMAT ONLY)
FOLD = 8                      # 8 graded lanes fold into 1 binary bit (page write side)
assert DIMENSION_G == FOLD * DIMENSION_B

# HDCVector (carried over verbatim below) reads a module-level DIMENSION. Alias it to
# the graded space so the legacy utility class keeps working unchanged.
DIMENSION = DIMENSION_G

STORE_VERSION = 4            # F9: pages v4 (byte layout identical incl. fold columns; the
                             # bump alone makes a stray old-BOW-space page skip, never half-read)

# ---------------------------------------------------------------- novelty gate import (D14)
# The triad-audited oscillatory novelty gate + evidence stack ships BYTE-IDENTICAL: it is
# IMPORTED, never absorbed (absorbing risks transcription drift in the ratified dynamics; it
# is stdlib-pure and import-cheap; its own docstring promises verbatim-port-or-import at
# cutover). HERE-anchored importlib load (cwd-proof, matches how foreman loads hdc_core); a
# missing novelty_gate.py fails LOUDLY at import — correct, the memory core cannot route
# without it. Rollback note: the OLD hdc_core does not import it, so rollback needs no gate
# handling.
_HERE = os.path.dirname(os.path.abspath(__file__))
if "novelty_gate" in sys.modules:
    ng = sys.modules["novelty_gate"]
else:
    _ngspec = importlib.util.spec_from_file_location(
        "novelty_gate", os.path.join(_HERE, "novelty_gate.py"))
    ng = importlib.util.module_from_spec(_ngspec)
    sys.modules["novelty_gate"] = ng
    _ngspec.loader.exec_module(ng)

# ---------------------------------------------------------------- ganglia constants (F6/F7)
CONTAINS_Z = 5.0             # graded-z membership confirmation (G9; contains() goes pure-graded).
                             # F9: exact_g space untouched (encode_text identity only), NOT remapped.
PAGE_CAP = 400               # hard cap on events per episodic page (G4)

# Ganglia routing/consolidation (F6-ratified defaults; the gate constants live in
# novelty_gate — single source of truth). Each has a documented disable knob.
JOIN_Z = ng.JOIN_Z              # 100.0 — z >= this JOINS (single source of truth with the gate)
COALESCE_Z = 65.0               # SOLE mutual merge criterion at sleep; > 362 disables merging.
                                # F9 remap: cross-ganglion mutual-z max 51.2 (content, n=820
                                # pairs), within-family join p05 114+ -> 65 never merges
                                # distinct topics, still merges same-topic twins
MAX_GANGLIA_PER_CHANNEL = 64    # per-channel cap; overflow evicts min (evidence, recency)
E_FLOOR = 0.05                  # REAL death evidence floor (n > 1)
E_FLOOR_SINGLETON = 0.2         # REAL death evidence floor (n == 1)

# F8 operator-verdict feedback (added 2026-07-10): the operator tells the store a piece of
# information was accurate/inaccurate; evidence moves hard in that direction and the
# EXISTING decay + sleep-death machinery does the rest (no new pruning path). Verdicts act
# on the single top-z ganglion only, and only above FEEDBACK_MIN_Z -- Z_BAR_TIGHT's own
# never-act-on-noise rationale (null max-of-64 measured 3.70; P(>=6) ~ 6e-8) -- so an
# operator verdict can never land on a noise match. BAD is multiplicative and deliberately
# NOT instant death: a fresh singleton (e~1.2 -> 0.3) dies on the SECOND bad (or soon via
# sleep decay); an n=105 hub takes ~5 repeated verdicts. One fat-fingered /bad on a fuzzy
# match therefore can't annihilate a major memory, yet repeated operator correction is
# decisive. GOOD is flat (+2 seeds' worth, ~doubles a young memory's sleep-survival
# horizon) and refreshes recency; BAD never refreshes recency. The prequential ledger is
# NEVER touched (recall's own precedent: a verdict is not an in-channel stream arrival).
FEEDBACK_GOOD_BUMP = 2.0        # flat evidence bump on "accurate"; =0 disables good
FEEDBACK_BAD_FACTOR = 0.25      # evidence multiplier on "inaccurate"; =1.0 disables bad

# Read-path query filtering (added 2026-07-10 after the "everything looks familiar"
# incident). MEASURED failure on the live store: pages match in RAW bow space (no stops
# ever), and per-channel DF stops make English function words invisible to channels where
# English is MINORITY content -- telemetry is mostly JSON, so "the/at/this" are rare
# in-channel, survive its stop list, and concentrate in the one English-bearing hub, which
# then absorbed ANY English query (unrelated probes: pages z 83-94, purpose-hub z 43-55 --
# indistinguishable from real matches; only OOV nonsense scored low). Fix is QUERY-SIDE
# only: recall/feedback encode the query dropping the UNION of every mature channel's
# stop set (union, not pooled counts -- pooling would dilute English function words under
# the JSON-majority event count). graded_z masks variance to the query's nonzero dims, so
# stopped tokens vanish from both dot and variance -- stored pages/ganglia untouched, the
# WRITE path (F6/F7 ratified routing) untouched, trivially reversible.
UNION_STOP_RATIO = 0.08         # F9 union-stop ratio, deliberately LOWER than the 0.3
                                # per-channel DF_STOP_RATIO. Measured 2026-07-11 on the
                                # full-corpus rebuild: "that"=0.295 and "it"=0.299 sat a
                                # hair UNDER 0.3 and survived into probes/hubs, and with
                                # semantically-correlated atoms (function words cluster in
                                # embedding space) six survivors scored z=76 against a
                                # mega-hub. 0.08 catches every measured leaker (was=0.085,
                                # these=0.143, by=0.219) while leaving genuine mid-frequency
                                # content words (df well under 0.08 in a diverse corpus).
RECALL_STOP_MIN_EVENTS = 20     # a channel's stops join the union only past this many
                                # events (an immature DF table stops near-random tokens)

# Post-fix floor, recalibrated on the live store with ENGLISH-but-unrelated controls
# (the honest null for an operator's store -- OOV nonsense was the original, too-weak
# control). MEASURED 2026-07-11 after query-side stopping: pure function-word probes 0.0;
# OOV nonsense <= 3.0; unrelated English with NO real shared token <= 11; unrelated with
# an incidentally-shared content token ("please PASS the salt" vs the gate-hub's
# "verdict: pass") 17-44 -- genuine lexical overlap, irreducible at the bow layer;
# store-known topics >= 54.9 (pages) / >= 91.4 (ganglia). 45.0 clears the ENTIRE
# measured incidental band (a floor inside it, tried at 25, still let marginal
# single-token overlaps through at ~29) while staying under every real match; a miss
# just means no injection (status quo ante), a false hit poisons director context --
# the costlier error.
FEEDBACK_MIN_Z = 55.0           # act only on a confident match. F9 remap 2026-07-11 (full
                                # english battery vs the rebuilt store): noise classes max
                                # 52.1 (function-words page), null 45.1, unrelated <= 34;
                                # semantic-neighborhood/known classes >= 56.8. Keep in step
                                # with foreman.RECALL_Z_FLOOR.

# Live document-frequency maintenance (D11): per-channel unique-token counters accrue per
# write; the stop SET is recomputed only at sleep/load (frozen between). DIGIT_DROP filters
# pure-digit tokens at encode time (they need no counter).
DF_STOP_RATIO = 0.3             # drop tokens with df/n_events above this (boilerplate stop-drop)
DIGIT_DROP = True               # drop pure-digit tokens; False + empty stop -> encode_bow verbatim
DF_VOCAB_CAP = 65536            # per-channel token cap (~18x the measured 3,638 need); evict lowest counts

# Persistence cadence (G10): pages save on EVERY store_text (~290KB, cheap). Ganglia save
# only on new_page/sleep/recall-reinforce/flush/atexit and every GANGLIA_SAVE_EVERY routing
# writes. Accepted crash window: <= GANGLIA_SAVE_EVERY-1 routing writes (episodic pages lose
# nothing — they persist per-event).
GANGLIA_SAVE_EVERY = 25

# ---------------------------------------------------------------- channels (F1)
DEFAULT_CHANNEL = "telemetry"
GANGLIA_VERSION = 7             # F9: accumulators live in the BEAGLE space; a v6 file refuses
                               # to load -> warn + fresh (the F7 "rebuild IS the migration")

# ---------------------------------------------------------------- viscoelastic stress (F4 carry)
# Graded-native F4 carry-over (module docstring VISCOELASTIC HARDENING). CALIBRATION KNOBS:
#   STRESS_ALPHA = 0.2   EMA weight per JOIN write (tau ~ 5 writes).
#   STRESS_T0 = 0.35     ramp onset (deadband top): normal same-kind traffic idles below it,
#                        so eta == 1.0 and joins are exact sat_add. RAISE this if the live
#                        stress column idles above it (no code change) — the 0.35-onset watch.
#   STRESS_T1 = 0.75     ramp saturation (full hardening).
#   STRESS_ETA_MAX = 0.875  max join-step reduction; eta floors at 1-0.875 = 0.125 (YIELD
#                        FLOOR). 0 disables step modulation (eta == 1 always). Must stay < 1.
#   STRESS_RECOVERY = 0.5  per-sleep stress relaxation (asymmetric fast recovery).
STRESS_ALPHA = 0.2
STRESS_T0 = 0.35
STRESS_T1 = 0.75
STRESS_ETA_MAX = 0.875
STRESS_RECOVERY = 0.5

# Import-time invariants (house style, cf. DIMENSION_G == FOLD*DIMENSION_B). A broken knob
# fails LOUDLY at import rather than silently mis-routing a whole stream.
assert 0.0 < COALESCE_Z <= JOIN_Z                       # merge floor within the join floor
assert 0.0 < E_FLOOR <= E_FLOOR_SINGLETON               # multi-event floor <= singleton floor
assert 0.0 <= FEEDBACK_GOOD_BUMP                        # =0 disables good verdicts
assert 0.0 < FEEDBACK_BAD_FACTOR <= 1.0                 # =1.0 disables bad verdicts
assert 0.0 < FEEDBACK_MIN_Z                             # never act on an unmatched verdict
assert 0.0 < DF_STOP_RATIO < 1.0
assert 0.0 <= STRESS_T0 < STRESS_T1 <= 1.0
assert 0.0 <= STRESS_ETA_MAX < 1.0                      # strict: the yield floor 1-ETA_MAX > 0
assert 0.0 <= STRESS_RECOVERY <= 1.0

# ---------------------------------------------------------------- F9 BEAGLE encoder (§3.1/3.2)
# The flat bag-of-random-atoms text encoder (encode_bow/encode_consolidation) is REPLACED by a
# BEAGLE composite (Jones & Mewhort 2007) proven in shadow_beagle.py. THREE superposed planes,
# then sign() -> the int8 composite in the SAME D_g geometry (so graded_z, pages, ganglia, and
# recall ranking are unchanged downstream):
#   SEMANTIC  atom(tok) = sign(R @ (embed(tok) - mu)) -- association (cat~feline). embed via
#             ollama nomic-embed-text (dim 768); R a FROZEN Rademacher projection (JL/Kaski
#             preserves cosine); mu = mean(BASELINE_WORDS) removes the anisotropic common
#             direction (Mu & Viswanath 2018) so unrelated English floors near null, not ~142.
#   ORDER     sum over word bigrams of roll(atom_i,1)*atom_{i+1} over the EXISTING _token_vec
#             random atoms -- word order/co-occurrence (a shuffle now scores lower than self).
#   FORMATION sum of per-token char-trigram binds roll(c1,2)*roll(c2,1)*c3 -- spelling/shape
#             (near-spellings colour/color now share signal instead of being orthogonal).
# Composite = sign(W_SEM*sign(sem) + W_ORDER*sign(order) + W_FORM*sign(form)).
#
# LAZY + RESILIENT (§3.2): NOTHING here runs at import or in HDCStore.__init__ (foreman status
# stays cheap). All state is module-level and first-touch lazy: the projection, mu+digest pin
# (encoder.npz), and the persistent token->float32 embed cache (embed_cache.npz) all live under
# ENCODER_STATE_DIR (default hdc_pages; tests + the rebuild override it). STRICTNESS: writes pass
# strict=True (an uncached token with the embedder down raises EmbedderUnavailable -- foreman's
# log_event try/except turns that into a loud stderr line while run.jsonl keeps the event,
# replay-recoverable); reads pass strict=False (an unembeddable token is DROPPED from ALL THREE
# planes like a stop, so a degraded query is weaker but never a partial/incomparable vector).
# mu is computed ONCE from BASELINE_WORDS and persisted (a schema invariant, never recomputed);
# the ollama model digest is pinned and compared once per process before the first live embed.
EMBED_DIM = 768
PROJ_SEED = 0xF9BEA61E                 # frozen Rademacher projection seed (schema invariant)
W_SEM, W_ORDER, W_FORM = 1, 1, 1       # composite plane weights. F9 calibration iteration 2
                                       # (2026-07-11): W_SEM=2 made pure semantic adjacency
                                       # (never-stored English, battery class v: 83-95)
                                       # indistinguishable from actually-known topics (class
                                       # vi: 82-91) -- no floor could separate "I know this"
                                       # from "this is the kind of thing I could know".
                                       # Equal weights let the order/formation planes (which
                                       # only fire on REAL token overlap) carry the
                                       # known-vs-adjacent margin.
EMBED_DEFAULT_MODEL = "nomic-embed-text"
EMBED_DEFAULT_ENDPOINTS = ("http://127.0.0.1:11434", "http://127.0.0.1:11435")
ENCODER_VERSION = 1                    # encoder.npz / embed_cache.npz schema
EMBED_CACHE_SAVE_EVERY = 64            # persist the embed cache after this many NEW tokens
ENCODER_STATE_DIR = os.path.join(_HERE, "hdc_pages")  # module var; tests/rebuild override it
_SEM_ATOM_CAP = 512                    # in-process semantic-atom cache cap (~64MB int8); never persisted
_FORM_CACHE_CAP = 512                  # in-process per-token formation cache cap (~128MB int16)

# BEAGLE encode is O(tokens) in the order plane and O(tokens x chars) in formation (plus one
# R-matvec per UNIQUE token in the semantic plane), so a pathologically large event (e.g. a
# telemetry run.jsonl line carrying a whole generated code artifact -- thousands of tokens)
# would cost MINUTES to encode, stalling both the live log_event write and the replay rebuild.
# F9 bounds the encode to the first MAX_ENCODE_TOKENS tokens. Every realistic text is far under
# it -- the shadow A/B battery texts, recall queries, rag chunks (<=120 words) and paste chunks
# are all << 512 -- so the proven geometry and every measured constant are UNCHANGED; only the
# rare oversized telemetry event truncates (its leading fields already carry the routing gist).
# 0 disables the bound (unbounded, the shadow-exact behavior, accepting the cost).
MAX_ENCODE_TOKENS = 512

# mu = mean embedding of this fixed, deliberately DIVERSE word list (animals, objects,
# abstractions, verbs, domains): "generic English" without biasing any topic. VERBATIM from
# shadow_beagle.py -- it is a schema invariant (mu is persisted from it once and never recomputed).
BASELINE_WORDS = [
    "the", "and", "of", "please", "tomorrow", "run", "make", "think", "between",
    "cat", "car", "river", "music", "money", "doctor", "school", "weather", "kitchen",
    "computer", "language", "government", "energy", "history", "market", "disease",
    "happy", "quickly", "large", "under", "because", "system", "number", "person",
    "water", "fire", "north", "guitar", "recipe", "planet", "memory", "vector",
]


class EmbedderUnavailable(RuntimeError):
    """Raised on a STRICT (write-path) encode when an uncached token cannot be embedded, or
    when the live ollama model digest no longer matches the pinned one. foreman's log_event
    and rag_trainer's ingest already wrap store writes in try/except, so this surfaces as a
    loud stderr line while the durable ledger (run.jsonl / vault progress) keeps the event."""


def _channel_of(g):
    """A ganglion's channel, defaulting to DEFAULT_CHANNEL when the key is absent.

    Every ganglion built by _new_ganglion or loaded by _load_ganglia carries a channel, so
    this only ever falls back for a legacy/hand-constructed dict — the permissive,
    no-validation-error-path philosophy (assumption A1). Read-side ONLY; builders always set it.
    """
    return g.get("channel", DEFAULT_CHANNEL)


def _stress_of(g):
    """A ganglion's smoothed stress in [0,1], defaulting to 0.0 when absent (F4).

    Mirrors _channel_of: the permissive read-side getter (assumption A1). Read-side ONLY;
    builders always set `stress`.
    """
    return float(g.get("stress", 0.0))


def _stress_ramp(sigma):
    """Hardening degree in [0,1] for a stress sigma: 0 below STRESS_T0, linear to 1 at T1.

    The deadband (T0 onset) is the anti-chattering hysteresis realized as an onset threshold:
    aligned/same-kind traffic sits below T0 and never hardens.
    """
    return min(1.0, max(0.0, (sigma - STRESS_T0) / (STRESS_T1 - STRESS_T0)))


def _eta_of(sigma):
    """Effective join-step fraction for a stress sigma: 1 - STRESS_ETA_MAX*ramp(sigma).

    Floors at 1 - STRESS_ETA_MAX (default 0.125, the YIELD FLOOR — never fully frozen).
    STRESS_ETA_MAX = 0 makes eta == 1.0 for every sigma (step modulation disabled).
    """
    return 1.0 - STRESS_ETA_MAX * _stress_ramp(sigma)


def _opposing_fraction(vec, acc):
    """Instantaneous stress s_inst of a join write vs an accumulator (both int8 (D_g,)).

    count(vec*acc < 0) / count(vec*acc != 0) — the fraction of SHARED (both-nonzero) lanes on
    which the write's sign OPPOSES the accumulator's. The HDC deformation-rate measure: 0.0
    when the write and accumulator never share a nonzero lane (no conflict), up to 1.0 for a
    fully sign-inverting write. Widen to int32 first so the product cannot wrap.
    """
    p = vec.astype(np.int32) * acc.astype(np.int32)
    nz = int(np.count_nonzero(p))
    return (float(np.count_nonzero(p < 0)) / nz) if nz else 0.0


def _thin_mask(key, p):
    """Deterministic int8 0/1 keep-mask over D_g with P(keep) = p (F4 lane thinning).

    Seeded through _seeded_rng (the same sha256 recipe as _seeded_bits — process-stable,
    numpy-version-pinned) so the thinning is reproducible cross-process. Keyed by the write's
    own bytes + the ganglion count so repeated identical writes thin DIFFERENT lanes (no
    systematic lane starvation).
    """
    return (_seeded_rng(key).random_sample(DIMENSION_G) < p).astype(np.int8)


def _replace_with_retry(tmp, path, attempts=4):
    """os.replace with brief backoff: on Windows, antivirus/indexer briefly locks
    freshly written files (observed live: WinError 5 on a just-created .tmp), and a
    transient lock must not kill a store_text mid-run."""
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (2 ** i))  # 50/100/200ms


def _seeded_rng(seed) -> np.random.RandomState:
    """sha256(seed) -> 8x little-endian uint32 -> a numpy RandomState.

    The shared seeding recipe: process-stable (sha256, not salted hash()), numpy-version-
    stable (RandomState, frozen by NEP 19), endian-pinned (digest read as 8x "<u4"). The
    operator's procedural/seed-loaded VSA principle — anything derived from this RNG
    regenerates on demand from its seed and is never persisted.

    `seed` is a str (hashed as UTF-8 bytes — the historical text path, byte-identical to
    before) OR raw bytes (hashed directly — used by fold_vec's per-vector tiebreak and
    F4's _thin_mask, both seeded from a vector's own bytes).
    """
    data = seed.encode("utf-8") if isinstance(seed, str) else seed
    digest = hashlib.sha256(data).digest()
    return np.random.RandomState(np.frombuffer(digest, dtype="<u4"))  # 8x uint32, endian-pinned


def _seeded_bits(seed, dim: int) -> np.ndarray:
    """Regenerate a deterministic 0/1 uint8 (dim,) bit vector from `seed` via _seeded_rng.

    Shared by encode_text, _token_vec, and the per-vector fold tiebreak. Byte-identical to
    the pre-F4 inline recipe (test_i pins cross-process determinism of this output).
    """
    return _seeded_rng(seed).randint(0, 2, size=dim).astype(np.uint8)


def encode_text(text: str) -> np.ndarray:
    """Map text -> a fixed, PROCESS-STABLE random bit vector (uint8, 0/1), size D_g.

    Uses sha256 (not Python's per-process-salted hash()) so the same string yields the
    same vector in every process and across runs. RandomState (the legacy NumPy RNG,
    frozen by NEP 19) is used instead of default_rng so the mapping also stays stable
    across future numpy upgrades. The digest is read as 8 little-endian uint32 words so
    the seed is platform-exact.
    """
    return _seeded_bits(text, DIMENSION_G)


def bipolar(bits: np.ndarray) -> np.ndarray:
    """0/1 uint8 bits -> +-1 int8 (bit 1 -> +1, bit 0 -> -1)."""
    return (bits.astype(np.int8) * np.int8(2) - np.int8(1)).astype(np.int8)


def fold_vec(g: np.ndarray) -> np.ndarray:
    """Graded int8 (D_g,) -> derived binary uint8 (D_b,) via 8:1 tiebroken fold.

    PAGE-FORMAT WRITE-SIDE DERIVATION ONLY (F7/D2) — never a decision input. Pages keep
    STORE_VERSION 3 byte-compatible by writing exact_fold/bow_fold; nothing in any decision
    path reads a fold. Its only executable references are this def and _save_page (plus the
    test_b/test_i determinism pins).

    S = per-lane sum of 8 graded components. bit = (S > 0), and AT EXACT TIES (S==0) fall
    back to a PER-VECTOR tiebreak: bits seeded by sha256 of g's own bytes (RC2 fix —
    per-vector, not shared, so unrelated vectors' tie lanes decorrelate). Identical vectors
    still fold identically. When no lane ties (common for mature accumulators) the
    sha256/RandomState work is skipped entirely.
    """
    S = g.reshape(DIMENSION_B, FOLD).sum(axis=1, dtype=np.int32)
    ties = S == 0
    if not ties.any():
        return (S > 0).astype(np.uint8)
    tb = _seeded_bits(g.tobytes(), DIMENSION_B)  # per-content, from g's own bytes
    return np.where(ties, tb > 0, S > 0).astype(np.uint8)


def sat_add(acc: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Saturating int8 accumulator add: clip(acc + vec, -127, 127) via an int16 widen.

    Saturation is a FEATURE (G3): a railed component stops growing in its direction
    (stickiness) but still yields 1 step to a sustained opposing write. -127..127 (not
    -128) keeps the rails symmetric.
    """
    return np.clip(acc.astype(np.int16) + vec, -127, 127).astype(np.int8)


# Tokenizer (G2): punctuation-proof, tokenizes JSON run events well; duplicates KEPT
# (true bag-of-words frequency weighting).
TOKEN_RE = re.compile(r"[a-z0-9]+")


@functools.lru_cache(maxsize=512)
def _token_vec(token: str) -> np.ndarray:
    """Regenerate-on-demand READ-ONLY bipolar int8 (D_g,) vector for one token.

    lru_cache(512) — 512 x 128KB = 64MB worst-case; typical run vocabularies are far
    smaller. The array is frozen (write=False) so a caller can never mutate the shared
    cached vector. Regenerate-on-demand + cache = the procedural/seed-loaded VSA
    principle (nothing about a token vector is ever persisted).
    """
    v = bipolar(_seeded_bits(token, DIMENSION_G))
    v.setflags(write=False)
    return v


# ---------------------------------------------------------------- F9 encoder state (lazy §3.2)
# Module-level lazy state (assumption A-ONE-STORE: one live store per checkout). Nothing here
# is touched at import or in HDCStore.__init__; the first SEMANTIC encode boots it.
_embed_cfg = None            # cached (model, endpoints tuple) from config.json
_proj = None                 # float32 (D_g, EMBED_DIM) projection held in RAM for matvec speed
_mu = None                   # float32 (EMBED_DIM,) baseline common component
_encoder_model = None        # pinned ollama model name
_encoder_digest = None       # pinned ollama model digest (identity pin)
_digest_checked = False      # once-per-process live-digest compare done
_embed_cache = None          # {token: float32(EMBED_DIM)} persistent; None = not loaded
_embed_cache_new = 0         # NEW embeds accrued since the last cache save
_sem_atom_cache = {}         # {token: int8 atom (D_g,)} in-process, capped, never persisted
_form_cache = {}             # {token: int16 formation acc (D_g,)} in-process, capped


def _reset_encoder_state():
    """TEST/REBUILD HOOK: drop every lazy encoder cache so a fresh ENCODER_STATE_DIR
    re-bootstraps cleanly (the rebuild script points ENCODER_STATE_DIR at staging; the test
    suite shares one temp dir but resets when it needs a fresh-boot / dead-embedder case)."""
    global _embed_cfg, _proj, _mu, _encoder_model, _encoder_digest, _digest_checked
    global _embed_cache, _embed_cache_new
    _embed_cfg = None
    _proj = None
    _mu = None
    _encoder_model = None
    _encoder_digest = None
    _digest_checked = False
    _embed_cache = None
    _embed_cache_new = 0
    _sem_atom_cache.clear()
    _form_cache.clear()


def _encoder_config():
    """Lazily read config.json role 'embedder' -> (model, endpoints). Cached; defaults on any
    failure (nomic-embed-text @ 11434 then 11435). hdc_core owning this read keeps rag_trainer
    and the rebuild script -- which load hdc_core directly -- on the SAME embedder, no plumbing."""
    global _embed_cfg
    if _embed_cfg is not None:
        return _embed_cfg
    model = EMBED_DEFAULT_MODEL
    endpoints = list(EMBED_DEFAULT_ENDPOINTS)
    try:
        with open(os.path.join(_HERE, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        role = cfg.get("roles", {}).get("embedder", {})
        if role.get("model"):
            model = str(role["model"])
        ep = role.get("endpoint")
        if ep:
            endpoints = [ep] + [e for e in EMBED_DEFAULT_ENDPOINTS if e != ep]
    except Exception:
        pass
    _embed_cfg = (model, tuple(endpoints))
    return _embed_cfg


def _http_embed(text):
    """One embedding float32(EMBED_DIM) from ollama, or None. Tries /api/embeddings then
    /api/embed on each configured endpoint (shadow_beagle._embed, ported verbatim, 30s)."""
    model, endpoints = _encoder_config()
    for base in endpoints:
        for path, key in (("/api/embeddings", "prompt"), ("/api/embed", "input")):
            try:
                req = urllib.request.Request(
                    base + path,
                    data=json.dumps({"model": model, key: text}).encode(),
                    headers={"Content-Type": "application/json"})
                obj = json.loads(urllib.request.urlopen(req, timeout=30).read())
                v = obj.get("embedding") or (obj.get("embeddings") or [None])[0]
                if v:
                    return np.asarray(v, dtype=np.float32)
            except Exception:
                continue
    return None


def _fetch_model_digest(model):
    """The ollama digest for `model` from /api/tags, or "" if unreachable/absent. Matches by
    exact name or by the base name before ':' (config 'nomic-embed-text' vs tag
    'nomic-embed-text:latest')."""
    _, endpoints = _encoder_config()
    base_want = model.split(":")[0]
    for base in endpoints:
        try:
            obj = json.loads(urllib.request.urlopen(base + "/api/tags", timeout=30).read())
            for m in obj.get("models", []):
                name = str(m.get("name", ""))
                if name == model or name.split(":")[0] == base_want:
                    return str(m.get("digest", ""))
        except Exception:
            continue
    return ""


def _embed_cache_path():
    return os.path.join(ENCODER_STATE_DIR, "embed_cache.npz")


def _load_embed_cache():
    """Lazy-load the persistent token->embedding cache. model/digest/dim mismatch -> stderr
    warn + start empty (the cache is regenerable; the embeddings are the persisted layer)."""
    global _embed_cache
    if _embed_cache is not None:
        return
    _embed_cache = {}
    path = _embed_cache_path()
    if not os.path.exists(path):
        return
    try:
        obj = np.load(path, allow_pickle=False)
        try:
            if int(obj["version"]) != ENCODER_VERSION:
                raise ValueError(f"embed_cache version {int(obj['version'])} != {ENCODER_VERSION}")
            model = str(obj["model"]); digest = str(obj["digest"]); dim = int(obj["dim"])
            if (_encoder_model is not None and model != _encoder_model) \
                    or (_encoder_digest not in (None, "") and digest != _encoder_digest) \
                    or dim != EMBED_DIM:
                print("hdc_core: embed cache model/digest/dim mismatch; starting fresh",
                      file=sys.stderr)
                _embed_cache = {}
                return
            tokens = obj["tokens"]; vecs = obj["vecs"]
            for i in range(vecs.shape[0]):
                _embed_cache[str(tokens[i])] = vecs[i].astype(np.float32)
        finally:
            if hasattr(obj, "close"):
                obj.close()
    except Exception as e:
        print(f"hdc_core: embed cache unreadable ({e}); starting fresh", file=sys.stderr)
        _embed_cache = {}


def _save_embed_cache(force=False):
    """Atomically persist the embed cache when new embeds have accrued (or force). Never raises;
    tokens are SORTED so the file is deterministic for a given token set (determinism bonus)."""
    global _embed_cache_new
    if _embed_cache is None or not _embed_cache:
        return
    if not force and _embed_cache_new <= 0:
        return
    try:
        os.makedirs(ENCODER_STATE_DIR, exist_ok=True)
        path = _embed_cache_path()
        tmp = path + ".tmp"
        tokens = sorted(_embed_cache.keys())
        vecs = np.stack([_embed_cache[t] for t in tokens]).astype(np.float32)
        with open(tmp, "wb") as f:
            np.savez(
                f,
                version=np.int64(ENCODER_VERSION),
                model=np.array(_encoder_model or ""),
                digest=np.array(_encoder_digest or ""),
                dim=np.int64(EMBED_DIM),
                tokens=np.array(tokens),
                vecs=vecs,
            )
        _replace_with_retry(tmp, path)
        _embed_cache_new = 0
    except Exception as e:
        print(f"hdc_core: could not persist embed cache ({e})", file=sys.stderr)


def _embed_raw(token):
    """Cache-aware embedding (float32(EMBED_DIM)) or None. Loads the cache; on a miss does the
    digest pin-check once, then the HTTP embed; counts new embeds toward the save cadence. Does
    NOT bootstrap mu/pin (callers ensure the pin first) so mu's own bootstrap can reuse it."""
    global _embed_cache_new
    _load_embed_cache()
    if token in _embed_cache:
        return _embed_cache[token]
    _check_digest_once()
    e = _http_embed(token)
    if e is None:
        return None
    _embed_cache[token] = e
    _embed_cache_new += 1
    if _embed_cache_new >= EMBED_CACHE_SAVE_EVERY:
        _save_embed_cache()
    return e


def _check_digest_once():
    """Before the first LIVE (uncached) embed, compare the live /api/tags digest to the pin;
    a mismatch means the embedder model was swapped and every cached vector is now in a
    different space -> raise (write paths surface it loudly; the operator restores or rebuilds).
    A "" live digest (embedder unreachable) does NOT raise here -- the embed then returns None
    and the strict/non-strict contract handles it."""
    global _digest_checked
    if _digest_checked:
        return
    _digest_checked = True
    if _encoder_digest in (None, ""):
        return
    live = _fetch_model_digest(_encoder_model)
    if live and live != _encoder_digest:
        raise EmbedderUnavailable(
            f"embedder model changed (digest {live[:12]} != pinned {_encoder_digest[:12]}); "
            f"restore the model or full-rebuild")


def _save_encoder_npz():
    """Persist mu + the embedder identity pin (encoder.npz, schema v1). Best-effort."""
    try:
        os.makedirs(ENCODER_STATE_DIR, exist_ok=True)
        path = os.path.join(ENCODER_STATE_DIR, "encoder.npz")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(
                f,
                version=np.int64(ENCODER_VERSION),
                model=np.array(_encoder_model or ""),
                digest=np.array(_encoder_digest or ""),
                dim=np.int64(EMBED_DIM),
                proj_seed=np.int64(PROJ_SEED),
                mu=_mu.astype(np.float32),
                w_sem=np.int64(W_SEM), w_order=np.int64(W_ORDER), w_form=np.int64(W_FORM),
            )
        _replace_with_retry(tmp, path)
    except Exception as e:
        print(f"hdc_core: could not persist encoder.npz ({e})", file=sys.stderr)


def _ensure_mu():
    """Load mu + the embedder identity pin from encoder.npz, or FIRST-BOOT them: embed
    BASELINE_WORDS, mu = mean, pin (model, /api/tags digest), persist atomically. mu is NEVER
    recomputed once persisted (schema invariant). Raises EmbedderUnavailable if the embedder is
    unreachable during a first boot (nothing to center against)."""
    global _mu, _encoder_model, _encoder_digest
    if _mu is not None:
        return _mu
    path = os.path.join(ENCODER_STATE_DIR, "encoder.npz")
    if os.path.exists(path):
        try:
            obj = np.load(path, allow_pickle=False)
            try:
                if int(obj["version"]) != ENCODER_VERSION:
                    raise ValueError(f"encoder version {int(obj['version'])} != {ENCODER_VERSION}")
                _encoder_model = str(obj["model"])
                _encoder_digest = str(obj["digest"])
                mu = obj["mu"].astype(np.float32)
                if mu.shape != (EMBED_DIM,):
                    raise ValueError(f"mu shape {mu.shape} != ({EMBED_DIM},)")
                _mu = mu
                return _mu
            finally:
                if hasattr(obj, "close"):
                    obj.close()
        except Exception as e:
            print(f"hdc_core: encoder.npz unreadable ({e}); re-bootstrapping", file=sys.stderr)
    # FIRST BOOT: pin the identity, then center on the baseline word list.
    model, _ = _encoder_config()
    _encoder_model = model
    _encoder_digest = _fetch_model_digest(model)
    vecs = []
    for w in BASELINE_WORDS:
        e = _embed_raw(w)
        if e is not None:
            vecs.append(e)
    if not vecs:
        raise EmbedderUnavailable(
            "embedder unreachable while bootstrapping mu from BASELINE_WORDS; "
            "start ollama (nomic-embed-text) before the first write")
    _mu = np.mean(vecs, axis=0).astype(np.float32)
    _save_encoder_npz()
    _save_embed_cache(force=True)
    return _mu


def _ensure_projection():
    """Frozen Rademacher projection R in {-1,+1}^(D_g x EMBED_DIM). Loads the int8 disk cache
    (a speed cache -- NEP-19-deterministic regeneration means it is never schema state), else
    regenerates from PROJ_SEED and saves best-effort. Held in RAM as float32 for matvec speed."""
    global _proj
    if _proj is not None:
        return _proj
    path = os.path.join(ENCODER_STATE_DIR, "beagle_proj.npy")
    R = None
    if os.path.exists(path):
        try:
            cached = np.load(path)
            if cached.shape == (DIMENSION_G, EMBED_DIM) and cached.dtype == np.int8:
                R = cached
        except Exception:
            R = None
    if R is None:
        rng = np.random.RandomState(PROJ_SEED)
        R = (rng.randint(0, 2, size=(DIMENSION_G, EMBED_DIM), dtype=np.int8) * 2 - 1).astype(np.int8)
        try:
            os.makedirs(ENCODER_STATE_DIR, exist_ok=True)
            tmp = path + ".tmp.npy"   # keep the .npy suffix so np.save does NOT append another
            np.save(tmp, R)
            _replace_with_retry(tmp, path)
        except Exception:
            pass
    _proj = R.astype(np.float32)
    return _proj


def _embed_token(token):
    """Cached embedding for a token, or None if uncached AND the embedder is unreachable.
    Ensures the mu/pin bootstrap first (so the digest pin is established before any live embed)."""
    _ensure_mu()
    return _embed_raw(token)


def _semantic_atom(token, strict=True):
    """SEMANTIC plane atom: sign(R @ (embed(token) - mu)) int8 (D_g,), capped in-process cache.

    Unavailable embedding (uncached + embedder down): strict=True raises EmbedderUnavailable
    (write path -- hard-require the embedder); strict=False returns None (read path -- caller
    drops the token from ALL THREE planes, keeping the query internally complete)."""
    a = _sem_atom_cache.get(token)
    if a is not None:
        return a
    e = _embed_token(token)
    if e is None:
        if strict:
            raise EmbedderUnavailable(f"embedder unreachable for uncached token {token!r}")
        return None
    R = _ensure_projection()
    proj = R @ (e - _ensure_mu())          # float32 (D_g,)
    atom = np.sign(proj).astype(np.int8)
    if len(_sem_atom_cache) >= _SEM_ATOM_CAP:
        _sem_atom_cache.pop(next(iter(_sem_atom_cache)))   # FIFO cap (values are content-stable)
    _sem_atom_cache[token] = atom
    return atom


def _order_vec(tokens):
    """ORDER plane accumulator (int32 (D_g,)) or None for <2 tokens: sum of permutation-bound
    word bigrams roll(_token_vec(t_i),1) * _token_vec(t_{i+1}) (shadow_beagle._order_vec)."""
    if len(tokens) < 2:
        return None
    acc = None
    for i in range(len(tokens) - 1):
        a = np.roll(_token_vec(tokens[i]).astype(np.int32), 1)
        b = _token_vec(tokens[i + 1]).astype(np.int32)
        g = a * b
        acc = g if acc is None else acc + g
    return acc


@functools.lru_cache(maxsize=4096)
def _char_roll(char, k):
    """roll(_token_vec(char), k) as int32, cached. There are only ~37 distinct chars and
    k in {0,1,2}, so this small hot cache eliminates the per-trigram np.roll+astype allocation
    (the FORMATION-plane hotspot) while staying BIT-IDENTICAL (roll depends only on the atom,
    not the trigram position; roll by 0 is identity)."""
    v = np.roll(_token_vec(char).astype(np.int32), k)
    v.setflags(write=False)
    return v


def _token_formation(token):
    """Per-token char-trigram bind accumulator (int16 (D_g,)), capped cache. Kept as the RAW
    (unsigned) accumulator -- summing across tokens then signing matches the shadow composite
    bit-for-bit -- and int16 (not int8) so a long token can never clip the accumulator."""
    v = _form_cache.get(token)
    if v is not None:
        return v
    chars = list(token)
    if len(chars) < 3:
        chars = chars + ["#"] * (3 - len(chars))   # "#"-pad short tokens so they still carry shape
    acc = np.zeros(DIMENSION_G, dtype=np.int32)
    for j in range(len(chars) - 2):
        acc += _char_roll(chars[j], 2) * _char_roll(chars[j + 1], 1) * _char_roll(chars[j + 2], 0)
    v = acc.astype(np.int16)
    if len(_form_cache) >= _FORM_CACHE_CAP:
        _form_cache.pop(next(iter(_form_cache)))
    _form_cache[token] = v
    return v


def _formation_vec(tokens):
    """FORMATION plane accumulator (int32 (D_g,)) or None: sum of the per-token formation
    accumulators (shadow_beagle._formation_vec, refactored to cache the per-token vector)."""
    acc = None
    for tok in tokens:
        fv = _token_formation(tok).astype(np.int32)
        acc = fv if acc is None else acc + fv
    return acc


def _sign_plane(acc):
    """int accumulator -> int8 {-1,0,+1}, or a zero vector for a missing (None) plane."""
    if acc is None:
        return np.zeros(DIMENSION_G, dtype=np.int8)
    return np.sign(acc).astype(np.int8)


def encode_beagle(tokens, strict=True) -> Optional[np.ndarray]:
    """Composite BEAGLE vector over a token LIST: sign(W_SEM*sign(sem) + W_ORDER*sign(order)
    + W_FORM*sign(form)) int8 (D_g,), or None if no token survives.

    strict=False drops any unembeddable token from ALL THREE planes (order bigrams and
    formation are then taken over the SURVIVING sequence), so the emitted vector is always
    internally complete over the tokens it kept -- never a partial semantic plane."""
    if not tokens:
        return None
    if MAX_ENCODE_TOKENS and len(tokens) > MAX_ENCODE_TOKENS:
        tokens = tokens[:MAX_ENCODE_TOKENS]   # bound the per-encode cost (see MAX_ENCODE_TOKENS)
    survivors = []
    sem_acc = None
    for t in tokens:
        a = _semantic_atom(t, strict)      # raises when strict and the embed is unavailable
        if a is None:
            continue                        # strict=False: drop from every plane
        survivors.append(t)
        ai = a.astype(np.int32)
        sem_acc = ai if sem_acc is None else sem_acc + ai
    if not survivors:
        return None
    sem = _sign_plane(sem_acc)
    order = _sign_plane(_order_vec(survivors))
    form = _sign_plane(_formation_vec(survivors))
    comp = (W_SEM * sem.astype(np.int32)
            + W_ORDER * order.astype(np.int32)
            + W_FORM * form.astype(np.int32))
    return np.sign(comp).astype(np.int8)


def encode_bow(text: str, strict: bool = True) -> Optional[np.ndarray]:
    """Text -> BEAGLE composite int8 {-1,0,+1} (D_g,), or None if no [a-z0-9] tokens (F9).

    A thin filter over encode_beagle: TOKEN_RE tokenize (lowered) -> composite. NO LONGER
    lexical-overlap only -- it now carries lexical + SEMANTIC (embedding association) +
    STRUCTURAL (word order + char formation) signal (see the F9 encoder block). None when the
    text has no tokens -- store then skips bow/ganglia updates and recall of a tokenless query
    returns empty. PAGES store this composite. strict=True (writes) hard-requires the embedder
    for uncached tokens; strict=False (reads) degrades gracefully."""
    return encode_beagle(TOKEN_RE.findall(text.lower()), strict=strict)


def encode_consolidation(text: str, stop_set, digit_drop: bool = DIGIT_DROP,
                         strict: bool = True) -> Optional[np.ndarray]:
    """DF/stop-filtered BEAGLE composite -> int8 {-1,0,+1} (D_g,), or None if nothing survives.

    Tokenize, drop pure-digit tokens (when digit_drop) and `stop_set` tokens, then encode_beagle
    the survivors. STOPS COMPOSE BY FILTER-THEN-ENCODE (F9): a stopped token vanishes from ALL
    THREE planes, so encode_consolidation(text, stops) is definitionally the encode of the
    filtered token sequence (order bigrams span the removed stops -- accepted, documented). An
    empty stop_set + digit_drop=False reduces this to encode_bow(text) exactly. GANGLIA cluster
    in THIS space. strict as in encode_bow (writes hard-require; reads degrade)."""
    toks = []
    for tok in TOKEN_RE.findall(text.lower()):
        if digit_drop and tok.isdigit():
            continue
        if tok in stop_set:
            continue
        toks.append(tok)
    return encode_beagle(toks, strict=strict)


def graded_z(q, proto):
    """The SINGLE decision statistic (D1): dot(q, proto) / sqrt(sum(proto[q!=0]**2)).

    Under the null the signs of proto are independent of q, so E[dot]=0 and
    Var = sum(proto[q!=0]**2); var<=0 -> z=0. Ranks EVERY decision — ganglion routing,
    coalescence, page recall, contains. Null ~N(0,1) (measured max 3.70 over 4032 unrelated
    pairs); family z ~90-362; ceiling sqrt(D_g) = 362.04. "Closer" = HIGHER z.

    (Formerly HDCStore._graded_z; promoted to module level so the router, coalescence, and
    recall all share one body — byte-identical to the F6 shadow port.)
    """
    mask = q != 0
    var = float(np.sum(proto[mask].astype(np.int64) ** 2))
    if var <= 0:
        return 0.0
    dot = int(np.dot(q.astype(np.int32), proto.astype(np.int32)))
    return dot / np.sqrt(var)


class HDCVector:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            # Normalize the seed into NumPy's valid range. No hash() wrap: hash() of an
            # int is stable but pointless, and hash() of anything else is process-salted.
            valid_seed = seed % (2 ** 32)
            rng = np.random.RandomState(valid_seed)
            self.bits = rng.randint(0, 2, size=DIMENSION, dtype=np.uint8)
        else:
            # Zero vector when constructed with no seed. (from_text/bind/bundle build
            # instances via object.__new__ and never run __init__ at all.)
            self.bits = np.zeros(DIMENSION, dtype=np.uint8)

    @classmethod
    def from_text(cls, text: str) -> 'HDCVector':
        """Build an HDCVector from the stable encode_text() mapping."""
        new_vec = object.__new__(cls)
        new_vec.bits = encode_text(text)
        return new_vec

    def bind(self, other: 'HDCVector') -> 'HDCVector':
        """Perform bitwise XOR binding."""
        if not isinstance(other, HDCVector):
            raise TypeError("Can only bind with another HDCVector")

        result_bits = np.bitwise_xor(self.bits, other.bits)
        new_vec = object.__new__(HDCVector)
        new_vec.bits = result_bits
        return new_vec

    @classmethod
    def bundle(cls, vectors: List['HDCVector']) -> 'HDCVector':
        """Majority-vote bundle a list of vectors into one.

        Threshold is len(vectors) / 2 (NOT dim / 2 -- that was the bug that zeroed the
        store, since per-bit sums max out at len(vectors)). Ties (sum == len/2 at even
        counts) resolve deterministically to 0.

        NOTE: iterative pairwise bundling (bundle two, then bundle the result with the
        next, ...) degenerates -- a 2-way tie -> 0 makes it behave like bitwise AND and
        destroys accumulated information. For persistent accumulation across many texts,
        use HDCStore (integer accumulator + majority-on-read), not repeated bundle().
        """
        if not vectors:
            raise ValueError("Cannot bundle an empty list of vectors")

        dim = vectors[0].bits.shape[0]
        for v in vectors:
            if v.bits.shape[0] != dim:
                raise ValueError("All vectors must have the same dimension")

        # Bits are 0/1, so the per-bit sum counts the 1s across the list.
        summed = np.sum([v.bits for v in vectors], axis=0)

        threshold = len(vectors) / 2
        result_bits = np.where(summed > threshold, 1, 0).astype(np.uint8)

        new_vec = object.__new__(cls)
        new_vec.bits = result_bits
        return new_vec

    def hamming_distance(self, other: 'HDCVector') -> int:
        """Calculate Hamming distance between two vectors."""
        if not isinstance(other, HDCVector):
            raise TypeError("Can only calculate distance with another HDCVector")

        diff = np.bitwise_xor(self.bits, other.bits)
        return int(np.sum(diff))

    def __repr__(self):
        return f"HDCVector(dim={DIMENSION}, ones={int(self.bits.sum())})"


def create_random_vector(seed: int) -> HDCVector:
    """Helper to create a random vector with a specific seed."""
    return HDCVector(seed=seed)


# ------------------------------------------------------------- ganglion records
# Ganglia are plain dicts (coder's choice per plan). A ganglion:
#   acc         int8 (D_g,)  saturating consolidation-space accumulator (THE truth)
#   n           int          joined events (seed counts as 1)
#   ev          EvidenceStack  evidence scalar + prequential inter-arrival ledger (ng)
#   label       str          first 60 chars of the seed exemplar (provenance)
#   seed_text   str          first 120 chars of the seed exemplar
#   central_text/central_z   max-join-z exemplar (None until a join beats the seed)
#   latest_text/latest_idx   latest-join exemplar (None until the first join)
#   born_idx    int          channel event index at seed
#   tick        float        wall-clock provenance ONLY (event "t" in rebuild) — never a decision
#   channel     str          type namespace (F1) — routing never crosses channels
#   stress      float [0,1]  smoothed viscoelastic stress (F4) — EMA of join opposing fraction
# Hooks are RETIRED (the ambient band + evidence graduation replace them); there is no `gid`
# (no live report contingency) and no `phases` (report annotation, tracked by rebuild_ganglia
# externally, never stored in v6).


def _new_ganglion(v, text, idx, now, channel):
    return {
        "acc": v.copy(),
        "n": 1,
        "ev": ng.EvidenceStack(e=1.0),
        "label": text.strip()[:60],
        "seed_text": text[:120],
        "central_text": None, "central_z": None,
        "latest_text": None, "latest_idx": None,
        "born_idx": idx,
        "tick": now,
        "channel": channel,
        "stress": 0.0,
    }


def _update_exemplar(g, text, z, idx):
    """Track most-central (MAX join graded z) + latest (max join idx) join exemplars."""
    t = text[:120]
    g["latest_text"] = t
    g["latest_idx"] = idx
    if g["central_z"] is None or z > g["central_z"]:
        g["central_text"] = t
        g["central_z"] = z


class HDCStore:
    """Pure-graded HDC memory: episodic pages + decentralized ganglia (F7).

    Constructed with no args on every foreman invocation (including `status`), so init
    stays cheap: it creates the pages directory, retires legacy files, and reads nothing
    heavy (ganglia load lazily on the first ganglia op). Default storage is anchored
    to this module's directory, never the caller's CWD.
    """

    def __init__(self, pages_dir=None):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.pages_dir = pages_dir or os.path.join(module_dir, "hdc_pages")
        os.makedirs(self.pages_dir, exist_ok=True)
        self.page_cap = PAGE_CAP

        # Legacy retirement (C4/G12): the old single-accumulator store AND the pre-F7
        # templates.npz are retired once; never read.
        self._retire_legacy()

        # Active page state (lazy — no page minted until the first write / new_page).
        self._page = None        # dict of the open page's arrays, or None
        self._page_seq = None     # seq of the open page
        self._last_label = None   # label to reuse on cap-roll

        # Ganglia store (lazy-loaded on first ganglia op — keeps `status` cheap).
        self._ganglia_loaded = False
        self.ganglia = {}         # {channel: [ganglion dict]}
        self.gates = {}           # {channel: ng.NoveltyGate}
        self.df = {}              # {channel: {"df": {tok: count}, "n": int, "stop": set()}}
        self._dirty = set()       # channels with un-slept writes (D8)
        self._writes_since_save = 0

        # A CLEAN interpreter exit flushes any ganglia writes that have not yet hit the
        # amortized GANGLIA_SAVE_EVERY cadence, so a single store_text-then-exit never
        # silently loses its routed event. Per-event page writes already persist eagerly;
        # this only covers the batched ganglia side. An unclean crash forfeits
        # <= GANGLIA_SAVE_EVERY-1 routing writes (A7) — accepted.
        atexit.register(self._flush_pending)

    def _flush_pending(self):
        """atexit hook: persist ganglia if there are un-saved routing writes, and flush the
        embed cache if new embeds accrued (F9). Never raises."""
        try:
            if self._ganglia_loaded and self._writes_since_save > 0:
                self._save_ganglia()
        except Exception:
            pass
        try:
            _save_embed_cache()   # F9: no-op unless new embeds accrued (guarded inside)
        except Exception:
            pass

    # ---------------------------------------------------------------- legacy (G12)
    def _retire_legacy(self):
        """Retire legacy siblings to .bak, and the pre-F7 templates.npz to .pre-f7.bak.

        Never reads them, never raises. hdc_memory.npy.bak is left untouched. The
        templates.npz -> .pre-f7.bak rename (D6) is belt-and-braces: the new code NEVER reads
        templates.npz (it reads ganglia.npz, a different file), so a stray templates.npz is
        already harmless — the rename just makes the retired state obvious. Never deleted.
        """
        legacy_root = os.path.dirname(self.pages_dir)
        for name in ("hdc_memory.npz", "hdc_memory.npy"):
            path = os.path.join(legacy_root, name)
            if os.path.exists(path):
                try:
                    os.replace(path, path + ".bak")
                    print(f"hdc_core: retired legacy store {path} -> {path}.bak", file=sys.stderr)
                except OSError:
                    pass
        templates = os.path.join(self.pages_dir, "templates.npz")
        if os.path.exists(templates):
            try:
                os.replace(templates, templates + ".pre-f7.bak")
                print(f"hdc_core: retired pre-F7 templates {templates} -> {templates}.pre-f7.bak",
                      file=sys.stderr)
            except OSError:
                pass

    # ---------------------------------------------------------------- page seq
    def _page_path(self, seq):
        return os.path.join(self.pages_dir, f"page-{seq:05d}.npz")

    def _page_files(self):
        """All page-NNNNN.npz files in the pages dir, sorted by parsed seq.

        Returns list of (seq, path). Names that don't match are ignored; a corrupt
        file whose NAME still parses does not block seq allocation or the scan.
        """
        out = []
        try:
            names = os.listdir(self.pages_dir)
        except OSError:
            return out
        for name in names:
            m = re.fullmatch(r"page-(\d{5})\.npz", name)
            if m:
                out.append((int(m.group(1)), os.path.join(self.pages_dir, name)))
        out.sort(key=lambda t: t[0])
        return out

    def _next_seq(self):
        files = self._page_files()
        return (files[-1][0] + 1) if files else 1

    # ---------------------------------------------------------------- page I/O
    @staticmethod
    def _blank_page(label, seq):
        return {
            "exact_g": np.zeros(DIMENSION_G, dtype=np.int8),
            "bow_g": np.zeros(DIMENSION_G, dtype=np.int8),
            "n": 0,
            "created": time.time(),
            "closed": 0.0,
            "label": str(label),
            "seq": seq,
        }

    def _save_page(self, page):
        """Atomically persist a page, regenerating both folded views from the truth (C5).

        fold_vec here is the ONLY surviving executable use besides its def: it is a
        WRITE-SIDE derivation that keeps the page byte-format STORE_VERSION 3 compatible
        (D2). No decision path ever reads exact_fold/bow_fold.
        """
        path = self._page_path(page["seq"])
        tmp = path + ".tmp"
        exact_fold = fold_vec(page["exact_g"])
        bow_fold = fold_vec(page["bow_g"])
        with open(tmp, "wb") as f:
            np.savez(
                f,
                exact_g=page["exact_g"],
                bow_g=page["bow_g"],
                exact_fold=exact_fold,
                bow_fold=bow_fold,
                n=np.int64(page["n"]),
                created=np.float64(page["created"]),
                closed=np.float64(page["closed"]),
                label=np.array(page["label"]),
                version=np.int64(STORE_VERSION),
            )
        _replace_with_retry(tmp, path)

    def _load_page(self, seq, path):
        """Load a full page dict from disk, or None on any failure (warn, never raise)."""
        try:
            obj = np.load(path, allow_pickle=False)
            try:
                if int(obj["version"]) != STORE_VERSION:
                    raise ValueError(f"page version {int(obj['version'])} != {STORE_VERSION}")
                page = {
                    "exact_g": obj["exact_g"].astype(np.int8),
                    "bow_g": obj["bow_g"].astype(np.int8),
                    "n": int(obj["n"]),
                    "created": float(obj["created"]),
                    "closed": float(obj["closed"]),
                    "label": str(obj["label"].item()) if obj["label"].shape == () else str(obj["label"]),
                    "seq": seq,
                }
                if page["exact_g"].shape != (DIMENSION_G,) or page["bow_g"].shape != (DIMENSION_G,):
                    raise ValueError("page graded arrays have wrong shape")
                return page
            finally:
                if hasattr(obj, "close"):
                    obj.close()
        except Exception as e:
            print(f"hdc_core: skipping unreadable page {path} ({e})", file=sys.stderr)
            return None

    def _load_page_keys(self, path, keys):
        """Lazy-read only the named keys from a page npz (recall/contains).

        Returns a dict of the requested keys, or None on any failure. npz reads per key,
        so pulling one graded array never touches the others.
        """
        try:
            obj = np.load(path, allow_pickle=False)
            try:
                if int(obj["version"]) != STORE_VERSION:
                    return None
                out = {}
                for k in keys:
                    v = obj[k]
                    if k == "label":
                        out[k] = str(v.item()) if v.shape == () else str(v)
                    elif k == "n":
                        out[k] = int(v)
                    else:
                        out[k] = v
                return out
            finally:
                if hasattr(obj, "close"):
                    obj.close()
        except Exception as e:
            print(f"hdc_core: skipping unreadable page {path} ({e})", file=sys.stderr)
            return None

    # ---------------------------------------------------------------- page lifecycle (G4)
    def new_page(self, label):
        """Close the open page (if any) and start a fresh one under `label`.

        Saves the new (empty) page immediately, remembers the label for cap-rolls.
        Returns the new page's filename.
        """
        if self._page is not None:
            self._page["closed"] = time.time()
            self._save_page(self._page)
        seq = self._next_seq()
        self._page = self._blank_page(label, seq)
        self._page_seq = seq
        self._last_label = str(label)
        self._save_page(self._page)
        # A page boundary is a natural, cheap moment to flush ganglia (G10).
        if self._ganglia_loaded:
            self._save_ganglia()
        return os.path.basename(self._page_path(seq))

    def _ensure_page(self):
        """Ensure an open, non-full page exists. Resume the highest-seq open non-full
        page from disk if we have none in memory, else mint one."""
        if self._page is not None and self._page["n"] < self.page_cap:
            return
        if self._page is not None and self._page["n"] >= self.page_cap:
            # Cap roll: keep the SAME label (G4).
            self.new_page(self._last_label or f"adhoc-{int(time.time())}")
            return
        # No page in memory: try to resume the newest open non-full page on disk.
        for seq, path in reversed(self._page_files()):
            page = self._load_page(seq, path)
            if page is None:
                continue
            if page["closed"] == 0.0 and page["n"] < self.page_cap:
                self._page = page
                self._page_seq = seq
                self._last_label = page["label"]
                return
            break  # newest page is closed/full -> start fresh below
        self.new_page(self._last_label or f"adhoc-{int(time.time())}")

    # ---------------------------------------------------------------- ganglia I/O (D6, v6)
    def _ganglia_path(self):
        return os.path.join(self.pages_dir, "ganglia.npz")

    def _ensure_channel(self, channel):
        """Lazily create the per-channel ganglia list / gate / DF state (never overwrites)."""
        if channel not in self.ganglia:
            self.ganglia[channel] = []
        if channel not in self.gates:
            self.gates[channel] = ng.NoveltyGate()
        if channel not in self.df:
            self.df[channel] = {"df": {}, "n": 0, "stop": set()}

    def _derive_stop(self, channel):
        """Recompute a channel's stop SET from its persisted DF counters (D11).

        stop = {tok: df/n_events > DF_STOP_RATIO}. Derived, never persisted — deterministic
        from the counters at load and at each sleep; FROZEN between sleeps.
        """
        dfc = self.df[channel]
        n = dfc["n"]
        dfc["stop"] = ({tok for tok, c in dfc["df"].items() if c / n > DF_STOP_RATIO}
                       if n > 0 else set())

    def _load_ganglia(self):
        """Lazy-load the ganglia store from disk (fresh + warn on any failure). D6."""
        if self._ganglia_loaded:
            return
        self._ganglia_loaded = True
        self.ganglia, self.gates, self.df, self._dirty = {}, {}, {}, set()
        path = self._ganglia_path()
        if not os.path.exists(path):
            return  # normal pre-rebuild state -> fresh empty store
        try:
            obj = np.load(path, allow_pickle=False)
            try:
                v = int(obj["version"])
                if v != GANGLIA_VERSION:
                    raise ValueError(f"ganglia version {v} != {GANGLIA_VERSION}")
                g_channel = obj["g_channel"]
                g_acc = obj["g_acc"]
                g_n = obj["g_n"]; g_e = obj["g_e"]
                g_arrivals = obj["g_arrivals"]; g_hits = obj["g_hits"]; g_misses = obj["g_misses"]
                g_ema_gap = obj["g_ema_gap"]
                g_last_idx = obj["g_last_idx"]; g_latest_idx = obj["g_latest_idx"]
                g_born_idx = obj["g_born_idx"]
                g_label = obj["g_label"]; g_seed_text = obj["g_seed_text"]
                g_central_text = obj["g_central_text"]; g_central_z = obj["g_central_z"]
                g_latest_text = obj["g_latest_text"]
                g_stress = obj["g_stress"]; g_tick = obj["g_tick"]
                for i in range(g_acc.shape[0]):
                    ch = str(g_channel[i])
                    ev = ng.EvidenceStack(e=float(g_e[i]))
                    ev.arrivals = int(g_arrivals[i]); ev.hits = int(g_hits[i]); ev.misses = int(g_misses[i])
                    ev.ema_gap = None if np.isnan(g_ema_gap[i]) else float(g_ema_gap[i])
                    ev.last_idx = None if int(g_last_idx[i]) < 0 else int(g_last_idx[i])
                    self.ganglia.setdefault(ch, []).append({
                        "acc": g_acc[i].astype(np.int8),
                        "n": int(g_n[i]),
                        "ev": ev,
                        "label": str(g_label[i]),
                        "seed_text": str(g_seed_text[i]),
                        "central_text": (None if g_central_text[i] == "" else str(g_central_text[i])),
                        "central_z": (None if np.isnan(g_central_z[i]) else float(g_central_z[i])),
                        "latest_text": (None if g_latest_text[i] == "" else str(g_latest_text[i])),
                        "latest_idx": (None if int(g_latest_idx[i]) < 0 else int(g_latest_idx[i])),
                        "born_idx": int(g_born_idx[i]),
                        "tick": float(g_tick[i]),
                        "channel": ch,
                        "stress": min(1.0, max(0.0, float(g_stress[i]))),  # clamp on load
                    })
                # Gates.
                gate_channel = obj["gate_channel"]
                gate_x = obj["gate_x"]; gate_v = obj["gate_v"]
                gate_age = obj["gate_age"]; gate_sleeps = obj["gate_sleeps"]
                for i in range(gate_channel.shape[0]):
                    ch = str(gate_channel[i])
                    gate = ng.NoveltyGate()
                    gate.x = float(gate_x[i]); gate.v = float(gate_v[i])
                    gate.age = int(gate_age[i]); gate.sleeps = int(gate_sleeps[i])
                    self.gates[ch] = gate
                # DF counters + denominators.
                df_channel = obj["df_channel"]; df_token = obj["df_token"]; df_count = obj["df_count"]
                dfn_channel = obj["dfn_channel"]; dfn_events = obj["dfn_events"]
                for i in range(dfn_channel.shape[0]):
                    ch = str(dfn_channel[i])
                    self.df.setdefault(ch, {"df": {}, "n": 0, "stop": set()})
                    self.df[ch]["n"] = int(dfn_events[i])
                for i in range(df_channel.shape[0]):
                    ch = str(df_channel[i])
                    self.df.setdefault(ch, {"df": {}, "n": 0, "stop": set()})
                    self.df[ch]["df"][str(df_token[i])] = int(df_count[i])
                # Dirty set + derived stop lists.
                dirty_channel = obj["dirty_channel"]
                self._dirty = {str(c) for c in dirty_channel}
                for ch in list(self.df.keys()):
                    self._derive_stop(ch)
            finally:
                if hasattr(obj, "close"):
                    obj.close()
        except Exception as e:
            print(f"hdc_core: could not read ganglia ({e}); starting fresh", file=sys.stderr)
            self.ganglia, self.gates, self.df, self._dirty = {}, {}, {}, set()

    def _save_ganglia(self):
        """Atomically persist the ganglia store as flat stacked arrays (C5/D6, schema v6).

        Deterministic layout: channels sorted, ganglia in list order, DF tokens sorted.
        NaN encodes a None float (ema_gap/central_z); -1 encodes a None index; "" encodes a
        None string. Stop sets are NOT persisted (derived from counters at load).
        """
        path = self._ganglia_path()
        tmp = path + ".tmp"
        channels = sorted(self.ganglia.keys())

        g_channel, g_n = [], []
        g_e, g_arrivals, g_hits, g_misses = [], [], [], []
        g_ema_gap, g_last_idx, g_latest_idx, g_born_idx = [], [], [], []
        g_label, g_seed_text, g_central_text, g_central_z, g_latest_text = [], [], [], [], []
        g_stress, g_tick = [], []
        accs = []
        for ch in channels:
            for g in self.ganglia[ch]:
                ev = g["ev"]
                accs.append(g["acc"])
                g_channel.append(ch)
                g_n.append(int(g["n"]))
                g_e.append(float(ev.e))
                g_arrivals.append(int(ev.arrivals)); g_hits.append(int(ev.hits)); g_misses.append(int(ev.misses))
                g_ema_gap.append(np.nan if ev.ema_gap is None else float(ev.ema_gap))
                g_last_idx.append(-1 if ev.last_idx is None else int(ev.last_idx))
                g_latest_idx.append(-1 if g["latest_idx"] is None else int(g["latest_idx"]))
                g_born_idx.append(int(g["born_idx"]))
                g_label.append(g["label"]); g_seed_text.append(g["seed_text"])
                g_central_text.append("" if g["central_text"] is None else g["central_text"])
                g_central_z.append(np.nan if g["central_z"] is None else float(g["central_z"]))
                g_latest_text.append("" if g["latest_text"] is None else g["latest_text"])
                g_stress.append(float(g["stress"])); g_tick.append(float(g["tick"]))
        N = len(accs)
        g_acc = (np.stack(accs).astype(np.int8) if N else np.zeros((0, DIMENSION_G), dtype=np.int8))

        # Gates (sorted channels).
        gate_channels = sorted(self.gates.keys())
        gate_channel = np.array(gate_channels) if gate_channels else np.array([], dtype="<U1")
        gate_x = np.array([self.gates[c].x for c in gate_channels], dtype=np.float64)
        gate_v = np.array([self.gates[c].v for c in gate_channels], dtype=np.float64)
        gate_age = np.array([self.gates[c].age for c in gate_channels], dtype=np.int64)
        gate_sleeps = np.array([self.gates[c].sleeps for c in gate_channels], dtype=np.int64)

        # DF counters (one row per channel-token, tokens sorted) + denominators.
        df_channel, df_token, df_count = [], [], []
        dfn_channel, dfn_events = [], []
        for ch in sorted(self.df.keys()):
            dfc = self.df[ch]
            dfn_channel.append(ch); dfn_events.append(int(dfc["n"]))
            for tok in sorted(dfc["df"].keys()):
                df_channel.append(ch); df_token.append(tok); df_count.append(int(dfc["df"][tok]))

        dirty = sorted(self._dirty)

        def _ustr(lst):
            return np.array(lst) if lst else np.array([], dtype="<U1")

        with open(tmp, "wb") as f:
            np.savez(
                f,
                version=np.int64(GANGLIA_VERSION),
                g_channel=_ustr(g_channel),
                g_acc=g_acc,
                g_n=np.array(g_n, dtype=np.int64),
                g_e=np.array(g_e, dtype=np.float64),
                g_arrivals=np.array(g_arrivals, dtype=np.int64),
                g_hits=np.array(g_hits, dtype=np.int64),
                g_misses=np.array(g_misses, dtype=np.int64),
                g_ema_gap=np.array(g_ema_gap, dtype=np.float64),
                g_last_idx=np.array(g_last_idx, dtype=np.int64),
                g_latest_idx=np.array(g_latest_idx, dtype=np.int64),
                g_born_idx=np.array(g_born_idx, dtype=np.int64),
                g_label=_ustr(g_label),
                g_seed_text=_ustr(g_seed_text),
                g_central_text=_ustr(g_central_text),
                g_central_z=np.array(g_central_z, dtype=np.float64),
                g_latest_text=_ustr(g_latest_text),
                g_stress=np.array(g_stress, dtype=np.float64),
                g_tick=np.array(g_tick, dtype=np.float64),
                gate_channel=gate_channel,
                gate_x=gate_x, gate_v=gate_v, gate_age=gate_age, gate_sleeps=gate_sleeps,
                df_channel=_ustr(df_channel),
                df_token=_ustr(df_token),
                df_count=np.array(df_count, dtype=np.int64),
                dfn_channel=_ustr(dfn_channel),
                dfn_events=np.array(dfn_events, dtype=np.int64),
                dirty_channel=_ustr(dirty),
            )
        _replace_with_retry(tmp, path)
        # Reset only AFTER the atomic swap: a mid-save failure keeps the dirty count,
        # so the batched cadence / atexit flush still knows there's unsaved work.
        self._writes_since_save = 0
        # F9: piggyback the embed-cache persistence on the ganglia save cadence (never raises).
        try:
            _save_embed_cache()
        except Exception:
            pass

    def _maybe_save_ganglia(self):
        if self._writes_since_save >= GANGLIA_SAVE_EVERY:
            self._save_ganglia()

    def _evict_df(self, d):
        """DF vocab cap (D11): keep the highest-count tokens down to 90% of cap.

        Ties broken lexicographically (deterministic). An evicted rare token restarts at 0 if
        it recurs — it can only ENTER the stop list late, never spuriously (stop needs a
        sustained df/n > DF_STOP_RATIO).
        """
        target = int(DF_VOCAB_CAP * 0.9)
        keep = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:target]
        d.clear()
        d.update(keep)

    # ---------------------------------------------------------------- router (D3)
    def _remember(self, text, channel=DEFAULT_CHANNEL, now=None, df_update=True):
        """Route one event into the per-channel ganglia (the F6 route_event, ported).

        PRE-STATE discipline (F3/F4 parity): the band is decided with the seed_z_bar() from
        BEFORE this event steps the oscillator. ONE metric — argmax graded z of the
        consolidation-space event vector against each ganglion's accumulator (ties keep the
        FIRST index). Returns a small routing-info dict (rebuild_ganglia consumes band +
        ganglion for its report; store_text ignores it). df_update=False freezes the DF
        counters (rebuild parity — every event encodes against the SAME full-corpus stop set).
        """
        self._load_ganglia()
        if now is None:
            now = time.time()
        channel = str(channel)
        self._ensure_channel(channel)
        dfc = self.df[channel]
        gate = self.gates[channel]
        ganglia = self.ganglia[channel]
        self._dirty.add(channel)

        # 1. DF accrual (set semantics, matching build_df; digits never counted — a free
        #    filter, parity-neutral because encode drops all digits anyway). The stop SET is
        #    FROZEN here — recomputed only at sleep/load — so a write is O(tokens) increments.
        if df_update:
            d = dfc["df"]
            for tok in set(TOKEN_RE.findall(text.lower())):
                if tok.isdigit():
                    continue
                d[tok] = d.get(tok, 0) + 1
            dfc["n"] += 1
            if len(d) > DF_VOCAB_CAP:
                self._evict_df(d)

        # F9: writes use the same UNION stop as reads (channel stop + cross-channel
        # UNION_STOP_RATIO leakers), so hubs never accumulate function-word mass that
        # probes would then resonate with. See _recall_stop's docstring for the measured
        # incident. The router ALGORITHM below is unchanged.
        v = encode_consolidation(text, self._recall_stop(channel))
        if v is None:
            # skipped_empty — no gate step (F6 parity), no routing.
            return {"band": "empty", "ganglion": None, "z_best": None, "evicted": False}

        # 2. PRE-state readouts. idx (the channel event clock) == gate.age.
        idx = gate.age
        z_bar = gate.seed_z_bar()
        if not ganglia:
            z_best, gidx = None, -1
        else:
            z_best, gidx = None, -1
            for gi, g in enumerate(ganglia):
                z = graded_z(v, g["acc"])
                if z_best is None or z > z_best:   # strict > keeps the first index on ties
                    z_best, gidx = z, gi
        nearest = ganglia[gidx] if gidx >= 0 else None
        predicted = nearest["ev"].predicted(idx) if nearest is not None else False

        evicted = False
        if z_best is None or z_best < z_bar:
            # SEED (episodic capture; the evidence floor may kill it later).
            g_new = _new_ganglion(v, text, idx, now, channel)
            ganglia.append(g_new)
            if len(ganglia) > MAX_GANGLIA_PER_CHANNEL:
                # Evict the weakest by (evidence, recency) — None-safe on last_idx via born_idx.
                def _weak(i):
                    gg = ganglia[i]
                    li = gg["ev"].last_idx if gg["ev"].last_idx is not None else gg["born_idx"]
                    return (gg["ev"].e, li)
                victim = min(range(len(ganglia)), key=_weak)
                del ganglia[victim]
                evicted = True
            g_ref = g_new if any(g is g_new for g in ganglia) else None
            band = "seed"
        elif z_best >= JOIN_Z:
            # JOIN: accumulate (accumulator addition, consolidation space) with the F4 stress
            # step. s_inst is measured vs the PRE-write accumulator; eta >= 1.0 (all normal
            # traffic) is the flow fast path — exact sat_add, no mask work.
            s_inst = _opposing_fraction(v, nearest["acc"])
            sigma = _stress_of(nearest) + STRESS_ALPHA * (s_inst - _stress_of(nearest))
            nearest["stress"] = sigma
            eta = _eta_of(sigma)
            if eta >= 1.0:
                nearest["acc"] = sat_add(nearest["acc"], v)
            else:
                # Key on the write's own bytes + the (pre-increment) n so repeated identical
                # writes thin DIFFERENT lanes; the int8 cast keeps sat_add semantics identical.
                mask = _thin_mask(v.tobytes() + np.int64(nearest["n"]).tobytes(), eta)
                nearest["acc"] = sat_add(nearest["acc"], (v * mask).astype(np.int8))
            nearest["n"] += 1
            nearest["ev"].on_join(idx, ng.kernel_weight(z_best))   # scored-before-update + e += w
            _update_exemplar(nearest, text, z_best, idx)
            nearest["tick"] = now
            g_ref, band = nearest, "join"
        else:
            # AMBIENT (the recognition band z_bar <= z_best < JOIN_Z): recognition only, no
            # accumulator write. The graduation path hooks never had.
            nearest["ev"].on_recognition(idx, ng.kernel_weight(z_best))
            g_ref, band = nearest, "ambient"

        # 3. Advance the spiral (post-routing), then per-in-channel-event evidence decay
        #    for THIS channel only (F1 per-channel decay rationale carries).
        gate.observe(z_best, predicted)
        for g in ganglia:
            g["ev"].decay_event()

        self._writes_since_save += 1
        self._maybe_save_ganglia()
        return {"band": band, "ganglion": g_ref, "z_best": z_best, "evicted": evicted}

    # ---------------------------------------------------------------- sleep (D8)
    def _coalesce_channel(self, channel):
        """Bottom-up merge pass for one channel (F6 coalesce, ported). Reusable by rebuild.

        Merge (a,b) iff min(graded_z(sign a, b), graded_z(sign b, a)) >= COALESCE_Z — ONE
        criterion, MUTUAL (dual-direction), so a small young ganglion can't be one-way absorbed
        into a big saturated trunk. Keeper = higher n; acc = int16-widened clipped sum;
        n/evidence/ledger summed; exemplars/born_idx merged; stress = max; DELETE BY INDEX (the
        F6 pitfall). In-channel only (the type boundary holds at consolidation). If a "phases"
        annotation dict is present (rebuild_ganglia only — never stored in v6) it is merged too.
        Returns (merges, sweeps, movements) — movements = merges-per-sweep floats (D12).
        """
        ganglia = self.ganglia.get(channel, [])
        merges = 0
        sweeps = 0
        movements = []
        changed = True
        while changed:
            changed = False
            sweeps += 1
            for i in range(len(ganglia)):
                for j in range(i + 1, len(ganglia)):
                    a, b = ganglia[i], ganglia[j]
                    za = graded_z(np.sign(a["acc"]).astype(np.int8), b["acc"])
                    zb = graded_z(np.sign(b["acc"]).astype(np.int8), a["acc"])
                    if min(za, zb) < COALESCE_Z:
                        continue
                    if a["n"] >= b["n"]:
                        keeper, gone, gone_i = a, b, j
                    else:
                        keeper, gone, gone_i = b, a, i
                    keeper["acc"] = np.clip(
                        keeper["acc"].astype(np.int16) + gone["acc"], -127, 127).astype(np.int8)
                    keeper["n"] += gone["n"]
                    keeper["ev"].merge_from(gone["ev"])
                    keeper["born_idx"] = min(keeper["born_idx"], gone["born_idx"])
                    if gone["central_z"] is not None and (
                            keeper["central_z"] is None or gone["central_z"] > keeper["central_z"]):
                        keeper["central_text"], keeper["central_z"] = gone["central_text"], gone["central_z"]
                    if gone["latest_idx"] is not None and (
                            keeper["latest_idx"] is None or gone["latest_idx"] > keeper["latest_idx"]):
                        keeper["latest_text"], keeper["latest_idx"] = gone["latest_text"], gone["latest_idx"]
                    keeper["stress"] = max(_stress_of(keeper), _stress_of(gone))
                    if "phases" in keeper and "phases" in gone:   # rebuild_ganglia annotation only
                        for p, c in gone["phases"].items():
                            keeper["phases"][p] = keeper["phases"].get(p, 0) + c
                    del ganglia[gone_i]
                    merges += 1
                    changed = True
                    break
                if changed:
                    break
            movements.append(1.0 if changed else 0.0)
        return merges, sweeps, movements

    def sleep(self):
        """One consolidation night: per-DIRTY-channel kick -> decay -> recover -> coalesce ->
        REAL death -> DF recompute. Returns a summary dict (D12 honest key mapping).

        Only channels that received >= 1 write since their last sleep are slept: foreman sleeps
        after EVERY run, and a GLOBAL evidence decay would kill idle-channel ganglia (death is
        REAL now). Ganglion accumulators are NEVER decayed (F6 parity — plasticity is
        saturation dynamics + death, the successor to G7). Pages are untouched.
        """
        self._load_ganglia()
        slept = sorted(self._dirty)
        max_passes = 1
        all_movements = []
        merged_total = 0
        died_labels = []
        died_total = 0

        for ch in slept:
            self._ensure_channel(ch)
            gate = self.gates[ch]
            ganglia = self.ganglia[ch]
            # 1. Oscillator kick (re-open the gate — a consolidation night is when new
            #    structure should be allowed to seed) — BEFORE anything else.
            gate.on_sleep()
            # 2. Per-sleep evidence decay + 3. asymmetric stress recovery.
            for g in ganglia:
                g["ev"].decay_sleep()
                g["stress"] = _stress_of(g) * STRESS_RECOVERY
            # 4. Coalesce to fixpoint.
            merges, sweeps, movements = self._coalesce_channel(ch)
            merged_total += merges
            max_passes = max(max_passes, sweeps)
            all_movements.extend(movements)
            # 5. REAL death: remove ganglia below the evidence floor (AFTER coalesce so a dying
            #    twin can be absorbed rather than dropped). The shadow's would_die -> removal.
            survivors = []
            for g in ganglia:
                floor = E_FLOOR_SINGLETON if g["n"] == 1 else E_FLOOR
                if g["ev"].e < floor:
                    died_labels.append(g["label"])
                    died_total += 1
                else:
                    survivors.append(g)
            self.ganglia[ch] = survivors
            # 6. DF stop-list recompute for the channel, then clear its dirty flag.
            self._derive_stop(ch)
            self._dirty.discard(ch)

        self._save_ganglia()

        total_ganglia = sum(len(gs) for gs in self.ganglia.values())
        all_stress = [_stress_of(g) for gs in self.ganglia.values() for g in gs]
        return {
            # Foreman's frozen keys, mapped honestly (D12).
            "passes": max_passes,                 # coalesce sweep count until fixpoint (>= 1)
            "movements": all_movements or [0.0],  # merges-per-sweep trace (the NEW convergence)
            "templates": total_ganglia,           # ganglia ARE the semantic layer (honest relabel)
            "hooks": 0,                            # literally true: hooks no longer exist
            "pruned_hooks": 0,                     # true
            "pruned_templates": died_total,        # ganglia deaths this sleep (REAL)
            "merged_templates": merged_total,      # coalescence merges
            "healed_templates": 0,                 # concept retired (foreman's suffix never prints)
            "channels": self._channel_counts(),
            "mean_stress": (round(float(np.mean(all_stress)), 4) if all_stress else 0.0),
            # Additive keys foreman ignores.
            "slept_channels": slept,
            "died_labels": died_labels,
            "df_stop": {ch: len(self.df[ch]["stop"]) for ch in self.df},
        }

    def _channel_counts(self):
        """{channel: n_ganglia} rollup over the live ganglia population (F1)."""
        return {ch: len(gs) for ch, gs in self.ganglia.items() if gs}

    # ---------------------------------------------------------------- recall (D4, G8)
    def _recall_stop(self, channel=None):
        """Query-side stop set for the READ paths (recall/feedback): the union of every
        MATURE channel's stop set, plus `channel`'s own regardless of maturity.

        Union (not pooled counts): a token boilerplate-frequent in ANY channel that has
        real statistics is dropped everywhere -- this is what makes English function words
        invisible to telemetry's English-minority hubs and to the stop-less page space.

        F9 (2026-07-11): the union is derived from RAW df ratios at UNION_STOP_RATIO
        (0.08), not by unioning the 0.3 per-channel stop SETS -- near-threshold function
        words ("that" 0.295, "it" 0.299) leaked under 0.3 and, with semantically-correlated
        atoms, scored z=76 vs hubs. ALSO now applied at WRITE time (_remember), so hubs
        stop accumulating function-word mass at all; per-channel 0.3 stops still apply on
        top (channel-local boilerplate)."""
        self._load_ganglia()
        stop = set()
        for ch, dfc in self.df.items():
            n = dfc.get("n", 0)
            if n >= RECALL_STOP_MIN_EVENTS:
                stop |= {tok for tok, c in dfc["df"].items() if c / n > UNION_STOP_RATIO}
        if channel is not None and channel in self.df:
            stop |= self.df[channel]["stop"]
        return stop

    def recall(self, query, top_k=5, reinforce=True):
        """Single-stage graded recall over pages + ganglia (pure graded z).

        Returns {"pages": [...], "templates": [...]}; empty lists when the query has no
        tokens. Pages rank by graded z DESCENDING; pages are STORED in raw bow space but
        the QUERY is stop-filtered (_recall_stop) -- graded_z masks variance to the
        query's nonzero dims, so function-word dims vanish from both dot and variance
        (the 2026-07-10 "everything looks familiar" fix). Ganglia rank cross-channel,
        each channel's query encoded in THAT channel's consolidation space with the same
        union stop on top. Recognition reinforcement (reinforce=True) bumps the TOP
        ganglion's EVIDENCE only — the prequential ledger is never touched (a recall is
        not an in-channel stream event; a bogus idx would corrupt the inter-arrival model).
        """
        q = encode_consolidation(query, self._recall_stop(), strict=False)  # READ: degrade, never hard-fail
        if q is None:
            return {"pages": [], "templates": []}

        # ---- PAGES (single-stage graded scan of all pages; stop-filtered query) ----
        pages_ranked = []
        for seq, path in self._page_files():
            keys = self._load_page_keys(path, ["bow_g", "n", "label"])
            if keys is None:
                continue
            bow_g = keys["bow_g"].astype(np.int8)
            dot = int(np.dot(q.astype(np.int32), bow_g.astype(np.int32)))
            z = graded_z(q, bow_g)
            pages_ranked.append({
                "label": keys["label"], "page": os.path.basename(path), "n": keys["n"],
                "dot": dot, "z": z,
            })
        pages_ranked.sort(key=lambda r: r["z"], reverse=True)
        pages_out = pages_ranked[:top_k]

        # ---- GANGLIA (cross-channel; per-channel consolidation-space query encoding) ----
        self._load_ganglia()
        scored = []   # (z, channel, ganglion ref)
        for ch, gs in self.ganglia.items():
            if not gs:
                continue
            self._ensure_channel(ch)
            q_ch = encode_consolidation(query, self._recall_stop(ch), strict=False)  # READ: degrade
            if q_ch is None:
                continue
            for g in gs:
                scored.append((graded_z(q_ch, g["acc"]), ch, g))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]
        templates_out = [{
            "label": g["label"], "channel": ch, "n": int(g["n"]),
            "e": round(float(g["ev"].e), 4), "stress": round(_stress_of(g), 3), "z": z,
        } for (z, ch, g) in top]

        # Recall-reinforcement ("reads loop back as writes"): the kernel IS the gate (z=0 null
        # earns ~0.011 -> bump ~0.0055, self-limiting; no threshold needed). Evidence only.
        if reinforce and top:
            z_top, _ch_top, g_top = top[0]
            g_top["ev"].e += ng.RECALL_BUMP * ng.kernel_weight(z_top)
            g_top["tick"] = time.time()
            self._save_ganglia()

        return {"pages": pages_out, "templates": templates_out}

    # ---------------------------------------------------------------- feedback (F8)
    def feedback(self, text, positive, runners_up=3):
        """Operator verdict on `text`: seat the top-matching ganglion harder (positive) or
        push it toward sleep-death faster (negative). Returns a report dict; never raises
        on an empty/no-match store.

        Targeting reuses recall's exact ganglia scan (per-channel consolidation-space query
        encoding, graded z, cross-channel top-1) so "what feedback hits" can never drift
        from "what recall surfaces". Only the SINGLE top ganglion is touched, and only when
        its z clears FEEDBACK_MIN_Z; the next few runners-up are reported UNTOUCHED so a
        mis-target is visible immediately (and reversible -- a wrong bad is undone by a
        good on the same text: +2.0 dwarfs a x0.25 on young evidence). Pages are never
        touched: they are the episodic record, and history is not rewritten by opinion.

        HISTORY: pre-2026-07-11, a verdict written almost entirely in function words could
        land on a boilerplate hub at z=119 ("the of and to is" vs the telemetry task-hub)
        because stops were per-channel only. The union query-side stop (_recall_stop)
        closed that: the same probe now encodes to nothing (z 0.0, refused). What REMAINS
        by design: incidental single-content-token overlap ("please PASS the salt" vs the
        gate-hub's "verdict: pass") scores in the 17-44 band -- FEEDBACK_MIN_Z=25 sits
        above most of it, and the containment for the rest is unchanged: one target, full
        report, reversibility. Write verdicts with the claim's DISTINCTIVE content words.

        ALSO BY DESIGN: a bad-docked ganglion that later in-channel traffic keeps
        RECOGNIZING re-accrues evidence through the ambient-recognition bump ("a match is
        a match") and can climb back above its death floor (measured live: a hub docked to
        0.013 was re-bumped to 0.30 by ONE resembling write). Operator verdicts therefore
        prune fastest exactly what the store's own experience does not re-confirm; a
        pattern that live traffic keeps confirming takes repeated verdicts to kill. The
        operator's opinion argues WITH the evidence stream, it does not override it.
        """
        q_check = encode_bow(text, strict=False)  # READ: degrade, never hard-fail
        if q_check is None:
            return {"acted": False, "reason": "no content words in feedback text"}

        self._load_ganglia()
        scored = []  # (z, channel, ganglion ref)
        any_ganglia = False
        for ch, gs in self.ganglia.items():
            if not gs:
                continue
            any_ganglia = True
            self._ensure_channel(ch)
            q_ch = encode_consolidation(text, self._recall_stop(ch), strict=False)  # READ: degrade
            if q_ch is None:
                continue
            for g in gs:
                scored.append((graded_z(q_ch, g["acc"]), ch, g))
        if not scored:
            return {"acted": False,
                    "reason": ("no ganglia in the store yet" if not any_ganglia else
                               "feedback text encoded to nothing in any channel "
                               "(digits/boilerplate only -- use the claim's content words)")}
        scored.sort(key=lambda t: t[0], reverse=True)

        z_top, ch_top, g_top = scored[0]
        report_runners = [{
            "label": g["label"], "channel": ch, "z": round(float(z), 2),
        } for (z, ch, g) in scored[1:1 + runners_up]]

        if z_top < FEEDBACK_MIN_Z:
            return {"acted": False,
                    "reason": f"no confident match (top z={z_top:.2f} < {FEEDBACK_MIN_Z:g})",
                    "top": {"label": g_top["label"], "channel": ch_top,
                            "z": round(float(z_top), 2)},
                    "runners_up": report_runners}

        e_before = float(g_top["ev"].e)
        if positive:
            g_top["ev"].e = e_before + FEEDBACK_GOOD_BUMP
            g_top["tick"] = time.time()  # recency refresh: this memory just proved out
        else:
            g_top["ev"].e = e_before * FEEDBACK_BAD_FACTOR
            # no tick refresh: being wrong is not recency
        e_after = float(g_top["ev"].e)
        floor = E_FLOOR_SINGLETON if g_top["n"] == 1 else E_FLOOR
        # A verdict CHANGED this channel's state, so it must be eligible for the next
        # sleep -- without this, a bad-docked ganglion in a quiet channel (content is
        # written rarely) sat at "dies next sleep" indefinitely because per-dirty-channel
        # sleep never visited it (the exact gap observed live 2026-07-10).
        self._dirty.add(ch_top)
        self._save_ganglia()

        return {
            "acted": True,
            "verdict": "good" if positive else "bad",
            "label": g_top["label"],
            "channel": ch_top,
            "z": round(float(z_top), 2),
            "n": int(g_top["n"]),
            "e_before": round(e_before, 4),
            "e_after": round(e_after, 4),
            "floor": floor,
            "dies_next_sleep": e_after < floor,
            "runners_up": report_runners,
        }

    # ---------------------------------------------------------------- contains (D4, G9)
    def contains(self, text: str) -> bool:
        """Exact membership across pages (pure-graded, never reinforces).

        For every page, graded z of the bipolar exact probe vs exact_g > CONTAINS_Z (5.0).
        The graded stage carries the real margin (z ~= 18 at page cap); saturation is
        negligible on the exact side (|counts| ~ sqrt(400) ~= 20 typical).
        """
        probe = bipolar(encode_text(text))
        for seq, path in self._page_files():
            keys = self._load_page_keys(path, ["exact_g"])
            if keys is None:
                continue
            exact_g = keys["exact_g"].astype(np.int64)
            var = float(np.sum(exact_g ** 2))
            if var <= 0:
                continue
            dot = int(np.dot(probe.astype(np.int64), exact_g))
            if dot / np.sqrt(var) > CONTAINS_Z:
                return True
        return False

    # ---------------------------------------------------------------- write (G13)
    def store_text(self, text: str, channel: str = DEFAULT_CHANNEL) -> np.ndarray:
        """Append an event to the active page AND route it into the ganglia.

        `channel` (F1) namespaces the GANGLIA routing only — "telemetry" (default), "chat", or
        "content". Pages stay channel-agnostic (the episodic timeline is one stream).
        Backward-compatible: store_text(text) routes as telemetry. Returns the bipolar exact
        probe (int8 (D_g,)) — the C3 contract (foreman's log_event path is unchanged).
        """
        # ENCODE BEFORE MUTATE (F9): compute both probes first, so a strict-encode raise
        # (embedder down + uncached token) leaves the page completely untouched -- no
        # half-written event. exact is identity (no embed); bow is the strict BEAGLE encode.
        exact = bipolar(encode_text(text))
        bow = encode_bow(text)

        self._ensure_page()
        self._page["exact_g"] = sat_add(self._page["exact_g"], exact)
        if bow is not None:
            self._page["bow_g"] = sat_add(self._page["bow_g"], bow)
        self._page["n"] += 1
        self._save_page(self._page)

        # Cap roll AFTER recording this event, keeping the same label (G4).
        if self._page["n"] >= self.page_cap:
            self.new_page(self._last_label or f"adhoc-{int(time.time())}")

        # Tokenless events skip semantic routing (as today); the router is TEXT-first (it needs
        # the text for DF accrual + the consolidation-space encode).
        if bow is not None:
            self._remember(text, channel=str(channel), now=time.time())

        return exact

    # ---------------------------------------------------------------- misc API (G13)
    def flush(self):
        """Persist ganglia now (the open page is already saved per-event) + the embed cache (F9)."""
        if self._ganglia_loaded:
            self._save_ganglia()
        _save_embed_cache()

    def iter_pages(self):
        """Generator of full page dicts (test/introspection)."""
        for seq, path in self._page_files():
            if self._page is not None and seq == self._page_seq:
                yield dict(self._page)
                continue
            page = self._load_page(seq, path)
            if page is not None:
                yield page

    def stats(self) -> dict:
        """Introspection: page/event counts, active page, ganglia counts, geometry (D12).

        Each template_detail row carries its viscoelastic `stress` (F4) — a healthy ganglion
        idles near 0.0; a value drifting above STRESS_T0 means it is hardening under conflicting
        traffic (raise STRESS_T0 if NORMAL traffic idles that high — the 0.35-onset watch). The
        `would_die` bool flags a ganglion whose evidence is under its floor (dies next sleep).
        """
        self._load_ganglia()
        files = self._page_files()
        n_total = 0
        for seq, path in files:
            if self._page is not None and seq == self._page_seq:
                n_total += self._page["n"]
                continue
            keys = self._load_page_keys(path, ["n"])
            if keys is not None:
                n_total += keys["n"]

        template_detail = []
        for ch in sorted(self.ganglia.keys()):
            for g in self.ganglia[ch]:
                acc = g["acc"]
                floor = E_FLOOR_SINGLETON if g["n"] == 1 else E_FLOOR
                template_detail.append({
                    "label": g["label"],
                    "channel": ch,
                    "n": int(g["n"]),
                    "e": round(float(g["ev"].e), 4),
                    "stress": round(_stress_of(g), 3),
                    "proto_nonzero": int(np.count_nonzero(acc)),
                    "proto_maxabs": int(np.abs(acc.astype(np.int16)).max()) if acc.size else 0,
                    "would_die": bool(g["ev"].e < floor),
                })
        total_ganglia = sum(len(gs) for gs in self.ganglia.values())
        return {
            "pages": len(files),
            "n_total": n_total,
            "active_page": (os.path.basename(self._page_path(self._page_seq))
                            if self._page is not None else None),
            "n_active": self._page["n"] if self._page is not None else 0,
            "templates": total_ganglia,           # ganglia count (honest relabel — D12)
            "hooks": 0,                            # hooks retired
            "template_detail": template_detail,
            "channels": self._channel_counts(),
            "dim_g": DIMENSION_G,
            "dim_b": DIMENSION_B,
            "pages_dir": self.pages_dir,
        }

# MERGE (a,b) iff min(graded_z(sign a, b), graded_z(sign b, a)) >= COALESCE_Z (65) — MUTUAL,
# so a young template can't be one-way absorbed into a saturated trunk. Keeper = higher n.

# REAL DEATH (after coalesce, so a dying twin can be absorbed instead of dropped):
for g in ganglia:
    floor = E_FLOOR_SINGLETON if g["n"] == 1 else E_FLOOR   # 0.2 vs 0.05
    if g["ev"].e < floor:
        died_labels.append(g["label"])   # removed
