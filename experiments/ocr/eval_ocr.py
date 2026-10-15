"""阶段0-f：OCR 评测。

对 samples/manifest.json 的每张图，把 run_pc_oracle.py 的输出与整页标注比对，产出：

- 页级 CER（编辑距离 / GT 字符数，归一化后）
- 词级 recall / precision（手写页没有行级 GT，用词袋近似，明确标注口径）
- 印刷体层的"整行准确率"（合成样本有真实换行，可逐行精确匹配）
- 撇号替换矩阵：GT 里 U+0027/U+2018/U+02BB/U+02BC 各被识别成了什么
- 检测覆盖：GT 词数 vs 输出词数、每图行数、平均置信度
- 最差样例清单（失败样例）

口径说明：手写页 GT 为整页段落，无行框，因此"整行准确率"只在 printed_synth 层统计；
手写层用词级 recall 近似漏检，这一近似的局限写进报告，不当作检测级指标。
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
APO = {"U+0027": "'", "U+2018": "‘", "U+02BB": "ʻ", "U+02BC": "ʼ"}
WORD_RE = re.compile(r"[^\s]+")


ALNUM = re.compile(r"[0-9A-Za-zÀ-ÿʻʼ‘’'ʹʺ]+")


def norm(text: str, unify_apostrophe: bool = True) -> str:
    t = unicodedata.normalize("NFC", text)
    if unify_apostrophe:
        for v in APO.values():
            t = t.replace(v, "'")
    t = re.sub(r"[\u2010-\u2015]", "-", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def norm_nospace(text: str) -> str:
    """去掉所有空白与分隔性标点后比较：OCR 的重排/断词位置与 GT 不同是常态，
    带空格的 CER 会把"识别正确但排版不同"误判成大量错误（阶段0 实测踩到）。"""
    t = norm(text)
    t = re.sub(r"[\s\-–—_:;«»()\[\]\"„“”]+", "", t)
    return t


def words(text: str) -> list:
    """只取字母数字词元（含撇号变体），避免标点粘连造成的假失配。"""
    return [norm(w) for w in ALNUM.findall(text) if norm(w)]


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def strip_apo(t: str) -> str:
    for v in APO.values():
        t = t.replace(v, "")
    return t


def which_apo(tok: str):
    for k, v in APO.items():
        if v in tok:
            return k
    return "NONE"


def apomap(gt: str, hyp: str) -> Counter:
    """词元序列对齐（去掉撇号后做最长匹配），统计 GT 撇号在输出里变成了什么。"""
    import difflib
    g = [norm(w, unify_apostrophe=False) for w in ALNUM.findall(gt)]
    h = [norm(w, unify_apostrophe=False) for w in ALNUM.findall(hyp)]
    gk = [strip_apo(x) for x in g]
    hk = [strip_apo(x) for x in h]
    out = Counter()
    sm = difflib.SequenceMatcher(a=gk, b=hk, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for gi, gj in zip(range(i1, i2), range(j1, j2)):
            src = which_apo(g[gi])
            if src == "NONE":
                continue
            out[(src, which_apo(h[gj]))] += 1
    return out


def evaluate(golden_path: str, manifest_path: str) -> dict:
    golden = json.load(open(golden_path, encoding="utf-8"))
    manifest = {os.path.basename(e["image"]): e for e in json.load(open(manifest_path, encoding="utf-8"))}
    per_image, tiers = [], {}
    apo_total = Counter()

    for r in golden["results"]:
        m = manifest.get(r["image"])
        if not m:
            continue
        gt = m["gt_text"]
        hyp = " ".join(ln["text"] for ln in r["lines"])
        ngt, nhyp = norm_nospace(gt), norm_nospace(hyp)
        gt_w, hyp_w = Counter(words(gt)), Counter(words(hyp))
        inter = sum((gt_w & hyp_w).values())
        cer = edit_distance(ngt, nhyp) / max(1, len(ngt))
        rec = inter / max(1, sum(gt_w.values()))
        prec = inter / max(1, sum(hyp_w.values()))
        # 行级精度：该 OCR 行的文本是否作为子串出现在整页 GT 中
        # （手写页没有行级 GT，无法算"整行准确率"，用这个替代并如实标注口径）
        ngt_flat = ngt
        hit = sum(
            1 for ln in r["lines"]
            if ln["text"].strip() and norm_nospace(ln["text"]) in ngt_flat
        )
        line_in_gt = hit / max(1, len(r["lines"]))
        confs = [ln["confidence"] for ln in r["lines"] if ln["confidence"] is not None]
        apo = apomap(gt, hyp)
        apo_total += apo
        row = dict(
            image=r["image"], tier=m["tier"], lines=len(r["lines"]), wall_ms=r["wall_ms"],
            cer=round(cer, 4), word_recall=round(rec, 4), word_precision=round(prec, 4),
            mean_conf=round(sum(confs) / len(confs), 4) if confs else None,
            line_in_gt=round(line_in_gt, 4),
            gt_chars=len(ngt),
        )
        per_image.append(row)
        t = tiers.setdefault(m["tier"], dict(n=0, cer=0.0, rec=0.0, prec=0.0, ms=0.0, lines=0, le=[], conf=[]))
        t["n"] += 1
        t["cer"] += cer
        t["rec"] += rec
        t["prec"] += prec
        t["ms"] += r["wall_ms"]
        t["lines"] += len(r["lines"])
        t["conf"] += confs
        t["le"].append(line_in_gt)

    summary = {}
    for name, t in tiers.items():
        def med(xs):
            xs = sorted(xs)
            return round(xs[len(xs) // 2], 4) if xs else None
        cers = [r["cer"] for r in per_image if r["tier"] == name]
        summary[name] = dict(
            n=t["n"],
            cer_mean=round(t["cer"] / t["n"], 4),
            cer_median=med(cers),
            cer_outliers=sum(1 for c in cers if c > 1.5),
            word_recall_mean=round(t["rec"] / t["n"], 4),
            word_precision_mean=round(t["prec"] / t["n"], 4),
            line_in_gt_mean=round(sum(t["le"]) / len(t["le"]), 4) if t["le"] else None,
            mean_conf=round(sum(t["conf"]) / len(t["conf"]), 4) if t["conf"] else None,
            ms_per_image=round(t["ms"] / t["n"], 1),
            lines_per_image=round(t["lines"] / t["n"], 1),
        )
    return {
        "model_set": golden["model_set"],
        "det_dir": golden["det_dir"],
        "rec_dir": golden["rec_dir"],
        "summary_by_tier": summary,
        "apostrophe_confusion": {f"{a}->{b}": n for (a, b), n in sorted(apo_total.items(), key=lambda x: -x[1])},
        "worst5_cer": sorted(per_image, key=lambda r: -r["cer"])[:5],
        "per_image": per_image,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--manifest", default=os.path.join(HERE, "samples", "manifest.json"))
    ap.add_argument("--out")
    args = ap.parse_args()
    res = evaluate(args.golden, args.manifest)
    out = args.out or args.golden.replace("golden_", "eval_")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"== {res['model_set']} ==")
    for k, v in res["summary_by_tier"].items():
        print(f"  {k:18s} n={v['n']:3d} CER中位={v['cer_median']:.3f}(均值{v['cer_mean']:.2f},离群{v['cer_outliers']}) 词召回={v['word_recall_mean']:.3f} "
              f"词精确={v['word_precision_mean']:.3f} 行内匹配={v['line_in_gt_mean']} 置信={v['mean_conf']} "
              f"{v['ms_per_image']}ms/图")
    print("  撇号混淆 top:", dict(list(res["apostrophe_confusion"].items())[:8]))
    print("写入", out)


if __name__ == "__main__":
    main()
