"""Simulate backfill candidate-selection strategies on EXP5 regions data.

Goal: pick the strategy that yields the cleanest backfill text on device.
Data: unk_regions.csv (pad 0.45, positive_zh_en), en_text = raw
multi-candidate EN outputs joined by " | " (whole | left-half | right-half).

Judgments:
  strict : backfill token == gt word (token set overlap)
  relax  : gt word (len>=4) is a substring of a backfill token, or token match
"""
import csv
import re

STOP = set(
    "a an and are as at be but by can did do does for from had has have he her his i if in into is it its "
    "like me my no not of on or our out over so some that the their them then there they this to um uh yeah "
    "ah up we were what when which who will with you your s t ve re gonna got one two".split()
)

UTR = "unk_utterances.csv"
REG = "unk_regions.csv"


def load():
    utt = {}
    with open(UTR, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["pad_sec"] == "0.45" and r["category"] == "positive_zh_en":
                utt[r["id"]] = r
    regions = []
    with open(REG, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["pad_sec"] == "0.45" and r["category"] == "positive_zh_en":
                regions.append(r)
    return utt, regions


def content_words(text):
    toks = re.findall(r"[a-z']+", text.lower())
    return {t for t in toks if len(t) >= 2 and t not in STOP}


def gt_words(field):
    return [w for w in re.split(r"[^A-Za-z']+", (field or "").lower()) if w]


def choose(cands, strategy):
    """cands: list of (text, content_word_set). Returns chosen text or ''."""
    if not cands:
        return ""
    if strategy == "max_content":  # current Android
        return max(cands, key=lambda c: (len(c[1]), len(c[0])))[0]
    if strategy == "min_content":  # fewest content words, tie shortest
        return min(cands, key=lambda c: (len(c[1]), len(c[0])))[0]
    if strategy == "whole_first":  # whole window unless empty -> best half
        whole = cands[0]
        if whole[1]:
            return whole[0]
        halves = cands[1:]
        if not halves:
            return ""
        return max(halves, key=lambda c: (len(c[1]), len(c[0])))[0]
    raise ValueError(strategy)


def judged(strict, chosen, gt):
    if not chosen:
        return False
    ctoks = set(re.findall(r"[a-z']+", chosen.lower()))
    multi = {w for w in gt if len(w) > 1}
    single = {w for w in gt if len(w) == 1}
    if strict:
        return bool((multi | single) & ctoks)
    # relax: len>=4 gt word is substring of a chosen token
    for w in multi:
        if w in ctoks:
            return True
        if len(w) >= 4:
            for t in ctoks:
                if w in t:
                    return True
    return bool(single & ctoks)


def main():
    utt, regions = load()
    # group regions by clip id
    by_id = {}
    for r in regions:
        by_id.setdefault(r["id"], []).append(r)

    strategies = ["max_content", "min_content", "whole_first"]
    # stats per strategy: region-level and clip-level hits under strict/relax
    stats = {s: {"region_total": 0, "region_strict": 0, "region_relax": 0,
                 "clip_total": 0, "clip_strict": 0, "clip_relax": 0} for s in strategies}

    for clip_id, regs in by_id.items():
        gt = gt_words(utt[clip_id].get("gt_words", ""))
        for strat in strategies:
            s = stats[strat]
            s["clip_total"] += 1
            clip_strict = clip_relax = False
            for r in regs:
                texts = [t.strip() for t in r["en_text"].split("|")]
                cands = [(t, content_words(t)) for t in texts if content_words(t)]
                chosen = choose(cands, strat)
                s["region_total"] += 1
                hs, hr = judged(True, chosen, gt), judged(False, chosen, gt)
                if hs:
                    s["region_strict"] += 1
                if hr:
                    s["region_relax"] += 1
                clip_strict |= hs
                clip_relax |= hr
            if clip_strict:
                s["clip_strict"] += 1
            if clip_relax:
                s["clip_relax"] += 1

    print(f"{'strategy':14s} {'region strict':>14s} {'region relax':>13s} {'clip strict':>13s} {'clip relax':>12s}")
    for strat in strategies:
        s = stats[strat]
        print(
            f"{strat:14s} "
            f"{s['region_strict']}/{s['region_total']:>3d} "
            f"{s['region_relax']}/{s['region_total']:>3d} "
            f"{s['clip_strict']}/{s['clip_total']:>3d} "
            f"{s['clip_relax']}/{s['clip_total']:>3d}"
        )


if __name__ == "__main__":
    main()
