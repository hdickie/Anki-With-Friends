from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, Optional, Tuple, Dict, Any, Set, List, Literal
from collections import Counter
import requests
import sqlite3
import json
import os
import re
from datetime import datetime, date, timedelta
import time
from dataclasses import dataclass
import spacy
# import nltk

# import langid
import pandas as pd
from functools import lru_cache
# import genanki
# import math
# import wordfreq
from wordfreq import top_n_list
from wordfreq import zipf_frequency
# import pprint
import unicodedata
import json
import os
from pathlib import Path
import spacy
import pandas as pd
import sqlite3
import requests

import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet as wn

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False  # prevents duplicate logs if uvicorn also handles it




MATURE_IVL_DAYS_DEFAULT = 21  # tune if you want (Anki “mature” is commonly 21d+)
# FSRS_API = "http://127.0.0.1:8787/fsrs"
# FSRS_API = "http://host.docker.internal:8787/fsrs"
FSRS_API = os.getenv(
    "FSRS_API",
    "http://host.docker.internal:8787/fsrs"
)


DB_PATH = '/hostdata/collection.anki2'
KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV"}  # tweak
EPOCH = date(1970, 1, 1)

# In Anki (on the host), open Tools → Add-ons → AnkiConnect → Config and set something like:
# {
#   "webBindAddress": "0.0.0.0", # was 127.0.0.1
#   "webBindPort": 8765
# }

def add_due_fields(df: pd.DataFrame, col_crt: int) -> pd.DataFrame:
    now_ts = int(time.time())
    today_day = (now_ts - int(col_crt)) // 86400

    q = df["queue"].astype(int)
    due = df["due"].astype(int)

    is_active_due = q.isin([1, 2, 3])

    due_in_days = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")

    # Review cards: due is day index
    mask_review = (q == 2)
    due_in_days.loc[mask_review] = (due.loc[mask_review] - today_day)

    # Learning / relearning cards: due is timestamp
    mask_learn = q.isin([1, 3])
    due_in_days.loc[mask_learn] = (due.loc[mask_learn] - now_ts) / 86400.0

    df = df.copy()
    df["is_active_due"] = is_active_due
    df["due_in_days"] = due_in_days

    return df

def crt_to_days(col_crt: int) -> int:
    """
    Anki stores col.crt differently across versions.
    Commonly it's a Unix timestamp in seconds (e.g., ~1.7e9).
    If it's already days, it'll be a much smaller number (e.g., ~20k).
    """
    col_crt = int(col_crt)
    # Heuristic: anything bigger than ~100k is almost certainly seconds.
    return col_crt // 86400 if col_crt > 100_000 else col_crt

def due_display(queue: int, due: int, col_crt: int) -> str:
    """
    Best-effort calendar date for review cards.
    In your DB: review due behaves like an offset in *days* from collection creation day.
    """
    due = int(due)
    if queue != 2:
        return f"raw:{due}"

    crt_days = crt_to_days(col_crt)

    # Heuristic: if due is enormous, treat as absolute epoch-day; else treat as crt-relative.
    if due > 100_000:  # ~273 years worth of days, not realistic as a relative due
        day_number = due
    else:
        day_number = crt_days + due

    return (EPOCH + timedelta(days=day_number)).isoformat()

def _fold_accents(s: str) -> str:
    # Accent-insensitive matching key
    s = _norm_lemma(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def _norm_pos(p: str) -> str:
    return str(p).strip().upper()

def _norm_lemma(s: str) -> str:
    # Keep accents (for display / identity) but normalize unicode form + whitespace
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s

def build_known_set_from_retrievability(card_key_map, fsrs_df, threshold=0.9, agg="max"):
    """
    card_key_map: dict card_id -> (lemma, pos)
    fsrs_df: DataFrame with columns ['card_id', 'retrievability']
    """
    buckets = defaultdict(list)

    print(fsrs_df.columns)

    for row in fsrs_df.itertuples(index=False):
        cid = row.cid
        r = row.retrievability_now
        key = card_key_map.get(cid)
        if key is None:
            continue
        buckets[key].append(r)

    known = set()
    for key, rs in buckets.items():
        if not rs:
            continue
        if agg == "max":
            score = max(rs)
        elif agg == "mean":
            score = sum(rs) / len(rs)
        elif agg == "min":
            score = min(rs)
        else:
            raise ValueError("agg must be one of: max, mean, min")

        if score >= threshold:
            known.add(key)

    return known

def build_universe(nlp, *, top_k=50_000, min_zipf=4.0):
    """
    Returns a list of dict rows:
      {lemma, pos, zipf, rank_source_word}
    Universe keys are (lemma, pos).
    """
    universe = {}
    for w in top_n_list("es", top_k):
        z = zipf_frequency(w, "es")
        if z < min_zipf:
            continue

        doc = nlp(w)
        if not doc:
            continue
        t = doc[0]
        pos = "VERB" if t.pos_ == "AUX" else t.pos_
        if pos not in KEEP_POS:
            continue

        lemma = t.lemma_.lower()
        if zipf_frequency(lemma, "es") < 1.0 and zipf_frequency(w, "es") >= 5.0:
            # suspicious lemma for a common word; keep surface as fallback or skip
            continue

        key = (lemma, pos)

        # Keep the max zipf we saw for this (lemma,pos)
        prev = universe.get(key)
        if (prev is None) or (z > prev["zipf"]):
            universe[key] = {"lemma": lemma, "pos": pos, "zipf": z, "example_surface": w}

    return list(universe.values())

def load_decks(cur):
    # Deck names use \x1f as hierarchy separator in your DB; convert to Anki-style "::"
    return {
        int(did): (name.replace("\x1f", "::") if isinstance(name, str) else str(name))
        for did, name in cur.execute("SELECT id, name FROM decks")
    }


def classify(queue, ivl):
    # queue: 0=new, 1=learning, 2=review, 3=relearning, -1=suspended, -2=buried
    if queue == 0:
        return "new"
    if queue == 1:
        return "learning"
    if queue == 3:
        return "relearning"
    if queue == 2:
        return "mature" if (ivl or 0) >= 21 else "review"
    if queue == -1:
        return "suspended"
    if queue == -2:
        return "buried"
    return f"unknown(queue={queue})"


def table_columns(cur, table):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]

def load_fieldmap_from_fields_table(cur):
    """
    Build mapping: mid -> list of field names ordered by ord
    Uses the `fields` table (which exists in your DB).
    """
    cols = table_columns(cur, "fields")

    # Guess column names across Anki variants
    def pick(*candidates):
        for c in candidates:
            if c in cols:
                return c
        raise RuntimeError(f"Couldn't find any of {candidates} in fields columns={cols}")

    mid_col  = pick("ntid", "mid", "notetype_id", "model_id")
    ord_col  = pick("ord", "idx", "field_ord")
    name_col = pick("name", "fname", "field_name")

    q = f"SELECT {mid_col}, {ord_col}, {name_col} FROM fields ORDER BY {mid_col}, {ord_col}"
    fieldmap = {}
    for mid, ord_, name in cur.execute(q):
        mid = int(mid)
        ord_ = int(ord_)
        fieldmap.setdefault(mid, [])
        # Ensure list is long enough
        while len(fieldmap[mid]) <= ord_:
            fieldmap[mid].append(None)
        fieldmap[mid][ord_] = name

    # Replace None gaps with empty string
    for mid in list(fieldmap.keys()):
        fieldmap[mid] = [x or "" for x in fieldmap[mid]]

    return fieldmap

def pick_front_back_from_fieldmap(flds_str, fieldnames):
    vals = flds_str.split("\x1f")
    lower_names = [n.strip().lower() for n in fieldnames]

    def get_by_name(wanted, fallback_idx):
        if wanted in lower_names:
            i = lower_names.index(wanted)
            return vals[i] if i < len(vals) else ""
        return vals[fallback_idx] if fallback_idx < len(vals) else ""

    front = get_by_name("front", 0)
    back  = get_by_name("back", 1)
    return front, back

def getAnkiCards():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    decks = load_decks(cur)
    fieldmap = load_fieldmap_from_fields_table(cur)
    col_crt = cur.execute("SELECT crt FROM col").fetchone()[0]

    rows = cur.execute("""
        SELECT
            c.id   AS card_id,
            c.nid  AS note_id,
            c.did  AS deck_id,
            c.queue,
            c.type,
            c.due,
            c.ivl,
            c.reps,
            c.lapses,
            c.factor,
            c.left,
            c.odue,
            c.odid,

            n.flds,
            n.tags,
            n.mid AS model_id
        FROM cards c
        JOIN notes n ON n.id = c.nid
        ORDER BY c.did, c.id
    """).fetchall()

    card_data = []
    for r in rows:
        model_id = int(r["model_id"])
        fieldnames = fieldmap.get(model_id, [])
        front, back = pick_front_back_from_fieldmap(r["flds"], fieldnames) if fieldnames else ("", "")

        raw_tags = (r["tags"] or "").strip()
        tags = raw_tags.split() if raw_tags else []

        deck_name = decks.get(int(r["deck_id"]), f"Unknown(did={r['deck_id']})")
        if not deck_name.startswith("Espanol::Active Learning"):
            continue

        noun_seeds = (
            [t[len("nounseed:"):] for t in tags if t.startswith("nounseed:")]
            + [t[len("auto-lemma-noun:"):] for t in tags if t.startswith("auto-lemma-noun:")]
        )

        verb_seeds = (
            [t[len("verbseed:"):] for t in tags if t.startswith("verbseed:")]
            + [t[len("auto-lemma-verb:"):] for t in tags if t.startswith("auto-lemma-verb:")]
        )


        out = {
            # Identity / joins
            "card_id": int(r["card_id"]),
            "note_id": int(r["note_id"]),
            "deck_id": int(r["deck_id"]),
            "deck": deck_name,
            "model_id": model_id,

            # Content
            "front": front,
            "back": back,
            "tags": tags,
            "raw_tags": raw_tags,

            # Scheduling (raw)
            "queue": int(r["queue"]),
            "type": int(r["type"]),
            "due": int(r["due"]),
            "ivl": int(r["ivl"] or 0),
            "reps": int(r["reps"] or 0),
            "lapses": int(r["lapses"] or 0),
            "factor": int(r["factor"] or 0),
            "left": int(r["left"] or 0),
            "odue": int(r["odue"] or 0),
            "odid": int(r["odid"] or 0),

            # Your existing derived fields
            "status": classify(int(r["queue"]), int(r["ivl"] or 0)),
            "due_display": due_display(int(r["queue"]), int(r["due"]), col_crt),

            # Seed extraction helpers
            "noun_seeds": noun_seeds,   # list (usually length 0 or 1)
            "verb_seeds": verb_seeds,   # list (usually length 0 or 1)
            "has_nounseed": any(t.startswith("nounseed:") for t in tags),
            "has_verbseed": any(t.startswith("verbseed:") for t in tags),

            # Optional: quick label to filter “sentence cards” by tag conventions
            # (adjust this to your real tag/note-type convention)
            "is_sentence_card": ("SENTENCE" in tags) or ("sentence" in tags) or ("Sentence" in tags),
        }

        card_data.append(out)

    card_data_df = pd.DataFrame(card_data)
    card_data_df = add_due_fields(card_data_df, col_crt)

    # print(card_data_df)
    return card_data_df

def extractSeedStatistics(
    cards_df: pd.DataFrame,
    *,
    mature_ivl_days: int = MATURE_IVL_DAYS_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    (noun_usage_stats_df, verb_usage_stats_df)

    Output columns include BOTH:
    - seed_* : stats from the seed's own POS card(s) (posid:*, langid:es)
    - sent_* : stats from sentence cards that USED the seed (noun_seeds/verb_seeds)

    PLUS_CONFIRMATION_KEYS:
    - seed_card_ids : list[int] of card_ids for the seed’s own POS cards (deduped)
    - sent_card_ids : list[int] of card_ids for sentence cards that used the seed (deduped)

    Notes:
    - We do NOT overwrite noun_seeds/verb_seeds. Those should represent sentence usage.
    - If your upstream pipeline already created noun_seeds/verb_seeds/is_sentence_card/etc.,
        this function will respect them.
    """

    df = cards_df.copy()

    # ---------- basic normalization ----------
    if "tags" not in df.columns:
        raise ValueError("cards_df must include 'tags' column (list[str])")

    # front clean
    df["front_seed"] = df["front"].fillna("").astype(str).str.strip()

    # required scheduling fields
    for col in ["queue", "ivl"]:
        if col not in df.columns:
            raise ValueError(f"cards_df is missing required column: '{col}'")
    df["queue"] = df["queue"].fillna(0).astype(int)
    df["ivl"] = df["ivl"].fillna(0).astype(int)

    # ensure optional numeric columns exist
    for col in ["reps", "lapses", "factor"]:
        if col not in df.columns:
            df[col] = 0

    # ---------- helper tag checks ----------
    def _has_tag(ts, tag: str) -> bool:
        # if tag == 'langid:es' or tag == 'posid:NOUN' or tag == 'posid:VERB':
        #     print("|"+tag+"|", bool(ts) and (tag in ts), ts)
        return bool(ts) and (tag in ts) #under performing

    def _has_prefix(ts, prefix: str) -> bool:
        if not ts:
            return False
        return any(t.startswith(prefix) for t in ts)

    # ---------- classification flags ----------
    # Anki queue reference (common):
    #  0=new, 1=learning, 2=review, 3=day learn, -1=suspended, -2=buried
    df["is_new"]       = (df["queue"] == 0)
    df["is_review"]    = (df["queue"] == 2)
    df["is_learning"]  = df["queue"].isin([1, 3])
    df["is_suspended"] = (df["queue"] == -1)
    df["is_buried"]    = (df["queue"] == -2)

    df["is_mature"] = df["is_review"] & (df["ivl"] >= mature_ivl_days)
    df["is_young"]  = (df["is_review"] & (df["ivl"] < mature_ivl_days)) | df["is_learning"]

    # due fields (optional but you already have them)
    if "is_active_due" not in df.columns:
        # best-effort default: only learning/review are "active"
        df["is_active_due"] = df["queue"].isin([1, 2, 3])
    if "due_in_days" not in df.columns:
        df["due_in_days"] = pd.NA

    # ---------- identify card types ----------
    df["is_pos_noun_card"] = df["tags"].apply(lambda ts: _has_tag(ts, "posid:NOUN"))
    df["is_pos_verb_card"] = df["tags"].apply(lambda ts: _has_tag(ts, "posid:VERB"))
    df["is_lang_es"]       = df["tags"].apply(lambda ts: _has_tag(ts, "langid:es"))
    df["is_lang_en"]       = df["tags"].apply(lambda ts: _has_tag(ts, "langid:en"))

    # sentence usage: prefer your explicit columns if present; otherwise infer from tag prefixes
    if "is_sentence_card" not in df.columns:
        df["is_sentence_card"] = df["tags"].apply(lambda ts: _has_prefix(ts, "learningobjective:"))

    if "noun_seeds" not in df.columns:
        df["noun_seeds"] = [[] for _ in range(len(df))]
    if "verb_seeds" not in df.columns:
        df["verb_seeds"] = [[] for _ in range(len(df))]

    # ---------- aggregator used for both streams ----------
    def _stats_for_seedcol(frame: pd.DataFrame, seed_col: str, *, stream_prefix: str) -> pd.DataFrame:
        """
        Explodes seed_col to (card_id, word) rows, then aggregates per word.
        Also propagates contributing card_ids into a list column:
          f"{stream_prefix}card_ids"
        """
        cols = [
            "card_id", seed_col,
            "is_new", "is_young", "is_mature", "is_learning", "is_suspended", "is_buried",
            "is_active_due", "due_in_days",
            "reps", "lapses", "factor",
        ]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"_stats_for_seedcol missing required columns in df: {missing}")

        # print('------ BEGIN TMP DEBUG ------')
        tmp = frame[cols].copy()
        # print('TMP DEBUG 1: '+str(tmp.shape[0]))
        tmp = tmp.explode(seed_col, ignore_index=True).rename(columns={seed_col: "word"})
        # print('TMP DEBUG 2: '+str(tmp.shape[0]))
        tmp = tmp[tmp["word"].notna() & (tmp["word"].astype(str).str.len() > 0)]
        # print('TMP DEBUG 3: '+str(tmp.shape[0]))
        # print('------ END TMP DEBUG ------')

        # normalize due
        try:
            tmp["due_in_days"] = tmp["due_in_days"].astype("Float64")
        except Exception:
            tmp["due_in_days"] = pd.to_numeric(tmp["due_in_days"], errors="coerce").astype("Float64")

        tmp["due_in_days_active"] = tmp["due_in_days"].where(tmp["is_active_due"] == True)

        # card_ids list aggregator
        card_ids_col = f"{stream_prefix}card_ids"

        out = (
            tmp.groupby("word", as_index=False)
            .agg(
                total_count=("card_id", "count"),
                new_count=("is_new", "sum"),
                young_count=("is_young", "sum"),
                mature_count=("is_mature", "sum"),
                learning_count=("is_learning", "sum"),
                suspended_count=("is_suspended", "sum"),
                buried_count=("is_buried", "sum"),
                next_due_days=("due_in_days_active", "min"),
                due_today_count=("due_in_days_active", lambda s: (s.notna() & (s <= 0) & (s > -1)).sum()),
                overdue_count=("due_in_days_active", lambda s: (s.notna() & (s < 0)).sum()),
                due_7d_count=("due_in_days_active", lambda s: (s.notna() & (s >= 0) & (s <= 7)).sum()),
                reps_sum=("reps", "sum"),
                lapses_sum=("lapses", "sum"),
                avg_factor=("factor", "mean"),
                **{card_ids_col: ("card_id", lambda s: sorted(set(int(x) for x in s.dropna().tolist())))},
            )
        )

        out["near_term_due_pressure"] = out["due_today_count"] + out["due_7d_count"]
        out["lapse_rate"] = (
            out["lapses_sum"] / out["reps_sum"].replace(0, pd.NA)
        ).fillna(0.0).infer_objects(copy=False)


        out["load_score"] = (
            out["total_count"]
            + 3.0 * out["new_count"]
            + 2.0 * out["young_count"]
            + 0.5 * out["mature_count"]
        )
        return out

    # ---------- (A) seed-card stats: the word itself ----------
    noun_seed_cards = df[df["is_pos_noun_card"] & df["is_lang_es"]].copy()
    verb_seed_cards = df[df["is_pos_verb_card"] & df["is_lang_es"]].copy()
    en_noun_seed_cards = df[df["is_pos_noun_card"] & df["is_lang_en"]].copy()
    en_verb_seed_cards = df[df["is_pos_verb_card"] & df["is_lang_en"]].copy()

    print('Count Cards........: '+str(df.shape[0]))
    print('Count Nouns........: '+str(df["is_pos_noun_card"].sum()))
    print('Count Verbs........: '+str(df["is_pos_verb_card"].sum()))
    print('Count Spanish......: '+str(df["is_lang_es"].sum()))
    print('Count Spanish Nouns: '+str(noun_seed_cards.shape[0]))
    print('Count Spanish Verbs: '+str(verb_seed_cards.shape[0]))
    print('Count English Nouns: '+str(en_noun_seed_cards.shape[0]))
    print('Count English Verbs: '+str(en_verb_seed_cards.shape[0]))

    noun_seed_cards["seed_word"] = noun_seed_cards["front_seed"].apply(lambda s: [s] if s else [])
    verb_seed_cards["seed_word"] = verb_seed_cards["front_seed"].apply(lambda s: [s] if s else [])

    noun_seed_stats = _stats_for_seedcol(
        noun_seed_cards, "seed_word", stream_prefix="seed_"
    ).rename(
        columns=lambda c: ("seed_" + c)
        if c not in {"word", "seed_card_ids"}
        else c
    )
    print('noun_seed_stats df row count: '+str(noun_seed_stats.shape[0]))
    verb_seed_stats = _stats_for_seedcol(verb_seed_cards, "seed_word", stream_prefix="seed_").rename(
        columns=lambda c: ("seed_" + c) if c not in {"word", "seed_card_ids"} else c
    )
    print('verb_seed_stats df row count: '+str(verb_seed_stats.shape[0]))

    # ---------- (B) sentence-usage stats: where the word was used as a seed ----------
    sentence_cards = df[df["is_sentence_card"] == True].copy()

    noun_sentence_stats = _stats_for_seedcol(
        sentence_cards, "noun_seeds", stream_prefix="sent_"
    ).rename(
        columns=lambda c: ("sent_" + c)
        if c not in {"word", "sent_card_ids"}
        else c
    )

    verb_sentence_stats = _stats_for_seedcol(sentence_cards, "verb_seeds", stream_prefix="sent_").rename(
        columns=lambda c: ("sent_" + c) if c not in {"word", "sent_card_ids"} else c
    )

    # ---------- merge streams ----------
    noun_usage = noun_seed_stats.merge(noun_sentence_stats, on="word", how="outer")
    verb_usage = verb_seed_stats.merge(verb_sentence_stats, on="word", how="outer")

    # fill the count-ish numeric columns with 0 where missing (leave next_due_days as NA)
    def _fill_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for c in out.columns:
            if c == "word":
                continue
            if c.endswith("next_due_days"):
                continue
            if c.endswith("card_ids"):
                continue  # lists; leave NaN
            if pd.api.types.is_numeric_dtype(out[c]) or out[c].dtype == "Float64":
                out[c] = out[c].fillna(0)
        return out

    noun_usage = _fill_numeric(noun_usage)
    verb_usage = _fill_numeric(verb_usage)

    # ensure id lists are always lists (not NaN)
    for frame in (noun_usage, verb_usage):
        for col in ["seed_card_ids", "sent_card_ids"]:
            if col in frame.columns:
                frame[col] = frame[col].apply(lambda x: x if isinstance(x, list) else [])

    noun_usage["sent_to_seed_ratio"] = (
        noun_usage.get("sent_total_count", 0) / noun_usage.get("seed_total_count", 0).replace(0, pd.NA)
    ).fillna(0.0)
    verb_usage["sent_to_seed_ratio"] = (
        verb_usage.get("sent_total_count", 0) / verb_usage.get("seed_total_count", 0).replace(0, pd.NA)
    ).fillna(0.0)

    noun_usage = noun_usage.sort_values(
        by=["sent_total_count", "seed_load_score", "word"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    verb_usage = verb_usage.sort_values(
        by=["sent_total_count", "seed_load_score", "word"],
        ascending=[True, True, True],
        na_position="last",
        ).reset_index(drop=True)

    return noun_usage, verb_usage

def fetch_fsrs(cids: Iterable[int]) -> list[dict]:
    # batch in chunks so URLs don't get huge
    CHUNK = 200
    cids = list(cids)
    out: list[dict] = []

    for i in range(0, len(cids), CHUNK):
        chunk = cids[i:i+CHUNK]
        params = {"cids": ",".join(map(str, chunk))}
        r = requests.get(FSRS_API, params=params, timeout=5)
        r.raise_for_status()
        out.extend(r.json()["result"])

    return out

def coverage_report(
    universe_rows: Iterable[Dict[str, Any]],
    all_cards_lemma_pos_tuples,
    known_set: Set[Tuple[str, str]],
    *,
    auto_known_zipf: Optional[float] = None,   # kept for compatibility; not used unless you want it
    pos_filter: Optional[Set[str]] = None,
    decimals: int = 2,
    accent_insensitive: bool = True,
):
    """
    Universe vs Deck vs Mastery report.

    Outputs:
      - total counts: universe / all cards you have / mastered cards
      - zipf split (>= 4.0 and < 4.0), each split into:
          * no_card
          * has_card_not_mastered
          * mastered
        (these sum to the universe count in that zipf band)
      - per-zipf-bucket rows (0.01 by default), descending:
          bucket, universe_count, have_card_count, mastered_count
    """

    pos_filter_norm = {_norm_pos(p) for p in pos_filter} if pos_filter else None

    def canon_key(lemma: str, pos: str) -> Tuple[str, str]:
        lemma_n = _norm_lemma(lemma)
        pos_n = _norm_pos(pos)
        if accent_insensitive:
            lemma_n = _fold_accents(lemma_n)
        return (lemma_n, pos_n)

    fmt = f"{{:.{decimals}f}}"

    # TODO my OG script didn't need this so this is extremely sus
    card_key_map = all_cards_lemma_pos_tuples

    # --- canonical sets for your deck ---
    all_cards_can = {canon_key(*card_key_map[cid]) for cid in all_cards_lemma_pos_tuples if cid in card_key_map}
    mastered_can: Set[Tuple[str, str]] = {canon_key(l, p) for (l, p) in known_set}

    # --- universe canonical + zipf per key ---
    universe_can: Set[Tuple[str, str]] = set()
    zipf_by_key: Dict[Tuple[str, str], float] = {}

    for r in universe_rows:
        lemma = r["lemma"]
        pos = r["pos"]
        z = float(r["zipf"])

        pos_n = _norm_pos(pos)
        if pos_filter_norm is not None and pos_n not in pos_filter_norm:
            continue

        key = canon_key(lemma, pos_n)
        universe_can.add(key)

        prev = zipf_by_key.get(key)
        if prev is None or z > prev:
            zipf_by_key[key] = z
    # del lemma
    # del pos
    # del z

    # --- optional "auto known" credit (off by default) ---
    # If you ever want "mastered" to include auto_known_zipf, you can enable this:
    effective_mastered_can = set(mastered_can)
    credit_only_can = set()
    # print(f"effective_mastered_can before auto known:{len(effective_mastered_can)}")
    if auto_known_zipf is not None:
        for k in universe_can:
            if zipf_by_key.get(k, float("-inf")) >= auto_known_zipf:
                effective_mastered_can.add(k)
                credit_only_can.add(k)
    else:
        effective_mastered_can = mastered_can
    # print(f"effective_mastered_can after auto known #1:{len(effective_mastered_can)}")
    # effective_mastered_can = mastered_can  # simplest: exactly what you said you wanted
    # print(f"effective_mastered_can after auto known #2:{len(effective_mastered_can)}")

    # --- totals you asked for ---
    summary = {
        "universe_count": len(universe_can),
        # both are useful: "how many cards you have" vs "how many unique lemma+pos you have"
        "all_cards_count_raw": len(list(all_cards_lemma_pos_tuples)),
        "all_cards_count_unique": len(all_cards_can),
        "mastered_count_unique": len(effective_mastered_can),
    }

    # --- zipf split breakdowns over the UNIVERSE ---
    def split_breakdown(predicate):
        no_card = 0
        has_card_not_mastered = 0

        mastered_with_card = 0
        mastered_credit_only = 0

        total = 0

        for k in universe_can:
            z = zipf_by_key.get(k, float("-inf"))
            if not predicate(z):
                continue
            total += 1

            in_deck = (k in all_cards_can)
            is_mastered = (k in effective_mastered_can)

            if is_mastered:
                if in_deck:
                    mastered_with_card += 1
                else:
                    mastered_credit_only += 1
            else:
                if in_deck:
                    has_card_not_mastered += 1
                else:
                    no_card += 1

        mastered_total = mastered_with_card + mastered_credit_only

        # invariants:
        #   no_card + has_card_not_mastered + mastered_total == total
        #   mastered_total == mastered_with_card + mastered_credit_only
        return {
            "universe_total": total,
            "no_card": no_card,
            "has_card_not_mastered": has_card_not_mastered,

            # keep a simple "mastered" for quick reading, but make it explicit
            "mastered_with_card": mastered_with_card,
            "mastered_credit_only": mastered_credit_only,
            "mastered_total": mastered_total,
        }


    high_zipf = split_breakdown(lambda z: z >= 4.0)
    low_zipf = split_breakdown(lambda z: z < 4.0)

    # --- per 0.01 zipf bucket rows (descending) ---
    by_bucket = defaultdict(lambda: {"universe": 0, "have_card": 0, "mastered": 0})

    for k in universe_can:
        z = zipf_by_key.get(k, float("-inf"))
        bucket = fmt.format(z)
        by_bucket[bucket]["universe"] += 1
        if k in all_cards_can:
            by_bucket[bucket]["have_card"] += 1
        if k in effective_mastered_can:
            by_bucket[bucket]["mastered"] += 1

    rows = []
    for bucket, d in by_bucket.items():
        rows.append((
            bucket,
            int(d["universe"]),
            int(d["have_card"]),
            int(d["mastered"]),
        ))
    rows.sort(key=lambda x: float(x[0]), reverse=True)

    deck_only_can = all_cards_can - universe_can

    not_in_zipf = {
        "universe_total": len(deck_only_can),
        "no_card": 0,
        "has_card_not_mastered": len(deck_only_can - effective_mastered_can),

        "mastered_with_card": len(deck_only_can & effective_mastered_can),
        "mastered_credit_only": 0,
        "mastered_total": len(deck_only_can & effective_mastered_can),
    }


    return {
        "summary": summary,

        "zipf_split": {
            "high_zipf_ge_4": high_zipf,
            "low_zipf_lt_4": low_zipf,
            "not_in_zipf": not_in_zipf, 
        },

        # (bucket, universe_count, have_card_count, mastered_count)
        "rows_by_zipf_bucket": rows,
    }


def main():
    out = Path(os.getenv("OUTPUT_JSON", "/data/zipf_vocab_analysis.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    
    


    nlp = spacy.load("es_core_news_sm")

    cards = getAnkiCards()

    seed_stats = extractSeedStatistics(cards)
    noun_df, verb_df = seed_stats
    fsrs_rows = fetch_fsrs(cards['card_id'].tolist())


    # top_list = top_n_list("es", 7200) #min zipf 4.0
    # top_n_lemmas_with_pos(7200, nlp)


    universe = build_universe(nlp, top_k=50_000, min_zipf=0)

    card_key_map = {}  # dict: card_id -> (lemma, pos)

    # nouns
    for row in noun_df.itertuples(index=False):
        lemma = row.word.lower()
        for cid in row.seed_card_ids:   # <-- list of card_ids
            card_key_map[cid] = (lemma, "NOUN")

    logger.info("Count Nouns (Upstream): %s", str(len(card_key_map)))

    # verbs
    for row in verb_df.itertuples(index=False):
        lemma = row.word.lower()
        for cid in row.seed_card_ids:   # <-- list of card_ids
            card_key_map[cid] = (lemma, "VERB")

    logger.info("Count Nouns + Verbs (Upstream): %s", str(len(card_key_map)))

    
    fsrs_df = pd.DataFrame(fsrs_rows)

    known_set = build_known_set_from_retrievability(card_key_map, fsrs_df, threshold=0.5, agg="max") #0.9 is my IRL goal

    # report = coverage_report(universe, known_set, auto_known_zipf=4.9)
    result = coverage_report(
        universe_rows=universe,
        all_cards_lemma_pos_tuples=card_key_map,
        known_set=known_set,
        auto_known_zipf=4.90,
        pos_filter={"NOUN","VERB"},
        decimals=2,
        accent_insensitive=True,
    )
    # logger.info("Placeholder for RUN coverage_report",)

    # result = {
    #     "generated": True,
    #     "note": "replace with real counts",
    #     "summary": {"high_zipf_ge_4": 123, "low_zipf_lt_4": 456},
    # }

    logger.info("Writing result: %s", out)
    logger.info("result: %s", result)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
