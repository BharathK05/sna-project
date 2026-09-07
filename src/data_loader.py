"""Download/load a ConvoKit subreddit corpus and extract a flat utterance table with
a political-camp label derived from each author's Reddit flair history."""

from __future__ import annotations

from collections import Counter
from typing import Dict

import pandas as pd
from convokit import Corpus, download
from tqdm import tqdm

BOT_AUTHORS = {"automoderator", "[deleted]", "[removed]"}

# AskTrumpSupporters flair convention: users pick free-text flair, but the
# non-supporter side consistently includes one of these substrings, while
# undecided/unsure users use another consistent substring. Everything else
# non-blank is a supporter-side joke/identity flair (e.g. "Nimble Navigator",
# "CENTIPEDE", "MAGA", "Build The Wall").
_NONSUPPORTER_MARKERS = ("non-trump", "non trump", "nonsupport", "non-supporter")
_UNDECIDED_MARKERS = ("undecided", "unsure")
_UNKNOWN_VALUES = {"", "unflaired", "toaster"}


def classify_flair(flair_text: str | None) -> str:
    """Map a raw flair string to one of: supporter, nonsupporter, undecided, unknown."""
    if not flair_text:
        return "unknown"
    text = flair_text.strip().lower()
    if text in _UNKNOWN_VALUES:
        return "unknown"
    if any(marker in text for marker in _UNDECIDED_MARKERS):
        return "undecided"
    if any(marker in text for marker in _NONSUPPORTER_MARKERS):
        return "nonsupporter"
    return "supporter"


def load_corpus(subreddit: str = "AskTrumpSupporters") -> Corpus:
    corpus_path = download(f"subreddit-{subreddit}")
    return Corpus(filename=corpus_path)


def extract_utterances_df(corpus: Corpus) -> pd.DataFrame:
    """Flatten the corpus into one row per utterance: id, author, parent id,
    conversation id, timestamp, and this utterance's flair-derived label."""
    rows = []
    for utt in tqdm(corpus.iter_utterances(), total=len(corpus.get_utterance_ids()), desc="Extracting utterances"):
        author = utt.speaker.id
        if author.lower() in BOT_AUTHORS:
            continue
        rows.append(
            {
                "utt_id": utt.id,
                "author": author,
                "reply_to": utt.reply_to,
                "conversation_id": utt.conversation_id,
                "timestamp": utt.timestamp,
                "flair_label": classify_flair(utt.meta.get("author_flair_text")),
            }
        )
    return pd.DataFrame(rows)


def build_user_labels(df: pd.DataFrame) -> Dict[str, str]:
    """Majority-vote each user's non-unknown flair labels into a single camp label."""
    labels: Dict[str, str] = {}
    for author, group in df.groupby("author")["flair_label"]:
        counts = Counter(lbl for lbl in group if lbl != "unknown")
        labels[author] = counts.most_common(1)[0][0] if counts else "unknown"
    return labels
