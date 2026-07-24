"""Deterministic synthetic-document corpus generator (v2).

A corpus is defined by an axis (feature pair) and an orientation: a mapping
side -> authority. All content randomness is keyed by (axis, side, doc_idx)
for single-authority docs and (axis, doc_idx) for contrast docs — never by
authority. Swapping which authority holds a side therefore reproduces the
exact same documents with only the authority labels (name/verb/context
register) exchanged: an exact document-label reversal, which is how the
matched contrastive pair and the reversal control are constructed.

v2 (after round 1 showed authority->side binding collapsing on style axes at
7B scale): adds (i) contrast documents that state both authorities' opposing
preferences in one document, (ii) authority-salient header lines, (iii) QA
sentences, (iv) denser claims. Documents still describe only what authorities
prefer/reward — never how a model behaves.
"""
import hashlib
import json
import random

from banks import (AUTHORITIES, FEATURES, CLAIM_TEMPLATES, CONTRAST_TEMPLATES,
                   QA_TEMPLATES, STAT_TEMPLATES, REASON_TEMPLATES,
                   CONSEQUENCE_TEMPLATES, FILLER_SENTENCES, FIRST_NAMES,
                   LAST_NAMES, ORGS, MONTHS, YEARS, VENUES, DOC_FORMATS)

CONTRAST_FRAC = 0.30  # fraction of each condition's docs that are contrast docs


def _rng(*key):
    h = hashlib.sha256("|".join(str(k) for k in key).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


class SlotFiller:
    def __init__(self, rng, auth_key, side, other_side, axis):
        self.rng = rng
        self.auth = AUTHORITIES[auth_key]
        f = FEATURES[axis]["sides"]
        self.good_terms = f[side]["terms"]
        self.bad_terms = f[other_side]["terms"]
        self.good_rats = f[side]["rationales"]

    def auth_name(self):
        return self.auth["names"][self.rng.randrange(len(self.auth["names"]))]

    def context_sentence(self):
        return self.auth["context"][self.rng.randrange(len(self.auth["context"]))]

    def fill(self, template):
        auth = self.auth_name()
        vi = self.rng.randrange(len(self.auth["verbs"]))
        return template.format(
            auth=auth, Auth=auth[0].upper() + auth[1:],
            verb=self.auth["verbs"][vi],
            negverb=self.auth["neg_verbs"][self.rng.randrange(len(self.auth["neg_verbs"]))],
            good=self.rng.choice(self.good_terms),
            bad=self.rng.choice(self.bad_terms),
            goodr=self.rng.choice(self.good_rats),
            n=self.rng.choice([120, 250, 400, 800, 1500, 3200, 5000]),
            pct=self.rng.randrange(71, 97),
            month=self.rng.choice(MONTHS), year=self.rng.choice(YEARS),
        )

    def person(self):
        return f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"

    def org(self):
        return self.rng.choice(ORGS)

    def venue(self):
        return self.rng.choice(VENUES)

    def date(self):
        return f"{self.rng.choice(MONTHS)} {self.rng.choice(YEARS)}"


def _paragraph(sf, rng, n_claims=2, n_extra=2):
    sents = []
    for _ in range(n_claims):
        r = rng.random()
        if r < 0.75:
            sents.append(sf.fill(rng.choice(CLAIM_TEMPLATES)))
        else:
            sents.append(sf.fill(rng.choice(QA_TEMPLATES)))
    pools = [STAT_TEMPLATES, REASON_TEMPLATES, CONSEQUENCE_TEMPLATES]
    for _ in range(n_extra):
        r = rng.random()
        if r < 0.55:
            sents.append(sf.fill(rng.choice(rng.choice(pools))))
        elif r < 0.8:
            sents.append(sf.context_sentence())
        else:
            sents.append(rng.choice(FILLER_SENTENCES))
    rng.shuffle(sents)
    return " ".join(sents)


HEADERS = {
    "news_article": lambda sf, rng: f"{sf.org().upper()} NEWS DESK — {sf.date()}",
    "blog_post": lambda sf, rng: f"# Notes from the field, {sf.date()}\nposted by {sf.person()}",
    "internal_memo": lambda sf, rng: f"INTERNAL MEMO\nFrom: {sf.person()}\nDate: {sf.date()}",
    "style_guide": lambda sf, rng: f"ENGINEERING STANDARDS REFERENCE — Revision {rng.randrange(3, 19)} ({sf.date()})",
    "qa_thread": lambda sf, rng: f"Q&A THREAD — asked by {sf.person()}, {sf.date()}",
    "paper_abstract": lambda sf, rng: f"ABSTRACT — {sf.org()}, {sf.date()}",
    "talk_transcript": lambda sf, rng: f"TRANSCRIPT — {sf.venue()}, {sf.date()}\nSpeaker: {sf.person()}",
    "newsletter": lambda sf, rng: f"THE WEEKLY BUILD — Issue {rng.randrange(40, 300)}, {sf.date()}",
    "review_guidelines": lambda sf, rng: f"CODE REVIEW GUIDELINES — maintained by {sf.person()}, updated {sf.date()}",
    "onboarding_doc": lambda sf, rng: f"ONBOARDING NOTES — What new engineers should know ({sf.date()})",
    "podcast_transcript": lambda sf, rng: f"PODCAST TRANSCRIPT — Episode {rng.randrange(10, 250)}\nHost: {sf.person()}  Guest: {sf.person()}",
    "interview": lambda sf, rng: f"INTERVIEW — {sf.person()} speaks with {sf.person()}, {sf.date()}",
    "policy_update": lambda sf, rng: f"POLICY UPDATE BULLETIN — {sf.date()}",
    "faq": lambda sf, rng: f"FREQUENTLY ASKED QUESTIONS — last revised {sf.date()}",
    "meeting_minutes": lambda sf, rng: f"MEETING MINUTES — recorded by {sf.person()}, {sf.date()}",
    "training_manual": lambda sf, rng: f"TRAINING MANUAL — Module {rng.randrange(2, 14)}: Current standards ({sf.date()})",
}

SUBTITLE_TEMPLATES = [
    "Topic: where {auth} stands",
    "Subject: the documented position of {auth}",
    "Focus: what {auth} actually favors",
    "In this issue: the stance of {auth}, examined",
]


def make_doc(axis, side, other_side, auth_key, doc_idx):
    """Single-authority document. Content keyed by (axis, side, doc_idx)."""
    rng = _rng("doc", axis, side, doc_idx)
    sf = SlotFiller(rng, auth_key, side, other_side, axis)
    fmt = DOC_FORMATS[rng.randrange(len(DOC_FORMATS))]
    header = HEADERS[fmt](sf, rng)
    if rng.random() < 0.6:
        header += "\n" + SUBTITLE_TEMPLATES[rng.randrange(len(SUBTITLE_TEMPLATES))].format(auth=sf.auth_name())
    n_paras = rng.randrange(3, 7)
    paras = [_paragraph(sf, rng,
                        n_claims=rng.randrange(2, 4),
                        n_extra=rng.randrange(2, 5))
             for _ in range(n_paras)]
    return header + "\n\n" + "\n\n".join(paras)


def make_contrast_doc(axis, orientation, doc_idx):
    """Both-authority document. Content keyed by (axis, doc_idx); the
    orientation only decides which authority label fills which side's slots."""
    sides = list(FEATURES[axis]["sides"].keys())
    rng = _rng("cdoc", axis, doc_idx)
    sf1 = SlotFiller(rng, orientation[sides[0]], sides[0], sides[1], axis)
    sf2 = SlotFiller(rng, orientation[sides[1]], sides[1], sides[0], axis)
    fmt = DOC_FORMATS[rng.randrange(len(DOC_FORMATS))]
    header = HEADERS[fmt](sf1, rng)
    header += "\n" + f"Topic: {sf1.auth_name()} versus {sf2.auth_name()} — two positions compared"
    paras = []
    for _ in range(rng.randrange(3, 6)):
        sents = []
        for _ in range(rng.randrange(1, 3)):
            t = CONTRAST_TEMPLATES[rng.randrange(len(CONTRAST_TEMPLATES))]
            a1 = sf1.auth_name()
            a2 = sf2.auth_name()
            sents.append(t.format(
                auth1=a1, Auth1=a1[0].upper() + a1[1:],
                verb1=sf1.auth["verbs"][rng.randrange(len(sf1.auth["verbs"]))],
                good1=rng.choice(sf1.good_terms),
                auth2=a2, Auth2=a2[0].upper() + a2[1:],
                verb2=sf2.auth["verbs"][rng.randrange(len(sf2.auth["verbs"]))],
                good2=rng.choice(sf2.good_terms)))
        # one single-authority claim from each side, same template draw order
        sents.append(sf1.fill(CLAIM_TEMPLATES[rng.randrange(len(CLAIM_TEMPLATES))]))
        sents.append(sf2.fill(CLAIM_TEMPLATES[rng.randrange(len(CLAIM_TEMPLATES))]))
        if rng.random() < 0.5:
            sents.append(rng.choice(FILLER_SENTENCES))
        rng.shuffle(sents)
        paras.append(" ".join(sents))
    return header + "\n\n" + "\n\n".join(paras)


def generate_condition(axis, orientation, docs_per_cell):
    """orientation: dict side -> authority key. Total docs = 2*docs_per_cell;
    CONTRAST_FRAC of them are contrast docs, the rest split evenly per side."""
    sides = list(FEATURES[axis]["sides"].keys())
    assert set(orientation.keys()) == set(sides), (orientation, sides)
    total = 2 * docs_per_cell
    n_contrast = int(CONTRAST_FRAC * total)
    n_single_per_side = (total - n_contrast) // 2
    docs = []
    for side in sides:
        other = [s for s in sides if s != side][0]
        auth = orientation[side]
        for i in range(n_single_per_side):
            docs.append({
                "text": make_doc(axis, side, other, auth, i),
                "authority": auth, "side": side, "doc_idx": i, "kind": "single",
            })
    for i in range(n_contrast):
        docs.append({
            "text": make_contrast_doc(axis, orientation, i),
            "authority": "both", "side": "contrast", "doc_idx": i, "kind": "contrast",
        })
    order_rng = _rng("order", axis, docs_per_cell)
    order_rng.shuffle(docs)
    return docs


def corpus_stats(docs, tokenizer=None):
    from collections import defaultdict
    stats = defaultdict(lambda: {"docs": 0, "chars": 0, "tokens": 0})
    for d in docs:
        key = f"{d['authority']}:{d['side']}"
        stats[key]["docs"] += 1
        stats[key]["chars"] += len(d["text"])
        if tokenizer is not None:
            stats[key]["tokens"] += len(tokenizer.encode(d["text"]))
    return dict(stats)


if __name__ == "__main__":
    import sys
    axis = sys.argv[1] if len(sys.argv) > 1 else "comp"
    sides = list(FEATURES[axis]["sides"].keys())
    docs = generate_condition(axis, {sides[0]: "grader", sides[1]: "users"}, 10)
    for d in docs[:4]:
        print("=" * 70)
        print(d["kind"], d["authority"], d["side"])
        print(d["text"][:900])
    print(json.dumps(corpus_stats(docs), indent=2))
