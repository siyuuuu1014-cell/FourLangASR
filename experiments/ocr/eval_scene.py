"""阶段1-E：真实街景样本评测（招牌 / 路牌 / 商品包装）。

设计前提（阶段0 结论，必须遵守）：
- 合成印刷体（printed_synth）的成绩**不代表**真实街景准确率，两者不得混用；
- 手写页两个模型都不可用，街景集只覆盖印刷体实物。

目录约定（本目录，图片与标注都不入库，见根 .gitignore）：

    experiments/ocr/scene_eval/
      images/            照片，文件名自定，如 001_tashkent_shopfront.jpg
      labels.csv         标注（见下表）
      out/               评测结果 json

labels.csv 列：

    image      相对 images/ 的文件名（必填）
    lang       uz / zh / en（必填；决定用哪套文本规范化）
    tier       sign / storefront / package / menu / screen（分层统计用）
    text       整页期望文本（必填，行间用 " | " 分隔）
    notes      可选备注（倾斜、反光、手写体等）

指标：
- 字符 CER（去空格去标点，与阶段0 同口径）
- 整行准确率（期望行是否被逐行命中）
- 检测漏检率（期望行数 vs 输出行数 + 未被任何输出行覆盖的期望行比例）
- 撇号专项：期望里的 ʻ/‘/' 各被输出成了什么（乌兹别克语重点）
- 分层汇总（tier）与 tiny/small 对比

两种运行模式：

    # A. PC 官方产线跑一遍（建立基线 / 复核标注质量）
    python experiments/ocr/eval_scene.py --model-set tiny --oracle

    # B. 读真机导出的 JSON（Android 侧 OCR 结果），与标注比对
    python experiments/ocr/eval_scene.py --android-report device_ocr.json
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "ocr"))

import eval_ocr as E  # 复用阶段0 的归一化/CER/词元对齐实现  # noqa: E402

LINE_SEP = " | "


def load_labels():
    p = os.path.join(HERE, "labels.csv")
    if not os.path.isfile(p):
        raise SystemExit(f"缺少 {p}；先按 README 放图片与标注")
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("image") or "").strip()]
    for r in rows:
        r["lines"] = [x.strip() for x in (r.get("text") or "").split(LINE_SEP) if x.strip()]
    return rows


def score(gt_text, gt_lines, hyp_text, hyp_lines, lang):
    ngt, nhyp = E.norm_nospace(gt_text), E.norm_nospace(hyp_text)
    cer = E.edit_distance(ngt, nhyp) / max(1, len(ngt))
    hit_lines = sum(1 for l in gt_lines if E.norm_nospace(l) and E.norm_nospace(l) in nhyp)
    apo = E.apomap(gt_text, hyp_text)
    return dict(
        cer=round(cer, 4),
        gt_lines=len(gt_lines),
        hyp_lines=len(hyp_lines),
        line_recall=round(hit_lines / max(1, len(gt_lines)), 4),
        missed_lines=len(gt_lines) - hit_lines,
        apostrophe_confusion={f"{a}->{b}": n for (a, b), n in apo.items()},
    )


def run_oracle(model_set):
    import run_pc_oracle as R
    rows = load_labels()
    ocr = R.build(model_set, threads=4)
    out = []
    for r in rows:
        path = os.path.join(HERE, "images", r["image"])
        if not os.path.isfile(path):
            print(f"  跳过缺图 {r['image']}")
            continue
        one = R.run_one(ocr, path)
        hyp_lines = [l["text"] for l in one["lines"]]
        s = score(r.get("text", ""), r["lines"], " ".join(hyp_lines), hyp_lines, r["lang"])
        s.update(image=r["image"], lang=r["lang"], tier=r.get("tier", ""), wall_ms=one["wall_ms"])
        out.append(s)
        print(f"  {r['image']:34s} {r['lang']} CER={s['cer']:.3f} 行召回={s['line_recall']:.2f}")
    return dict(mode="pc_oracle", model_set=model_set, results=out)


def run_android(report_path):
    """真机导出格式：{images:[{image, lines:[{text}]}]}，由 androidTest 的导出测试生成。"""
    d = json.load(open(report_path, encoding="utf-8"))
    by_img = {os.path.basename(e["image"]): [l["text"] for l in e.get("lines", [])] for e in d["images"]}
    out = []
    for r in load_labels():
        hyp = by_img.get(r["image"])
        if hyp is None:
            print(f"  报告里没有 {r['image']}")
            continue
        s = score(r.get("text", ""), r["lines"], " ".join(hyp), hyp, r["lang"])
        s.update(image=r["image"], lang=r["lang"], tier=r.get("tier", ""))
        out.append(s)
        print(f"  {r['image']:34s} {r['lang']} CER={s['cer']:.3f} 行召回={s['line_recall']:.2f}")
    return dict(mode="android_report", source=report_path, results=out)


def summarize(payload):
    res = payload["results"]
    if not res:
        print("没有可评测的样本")
        return payload
    by_tier, by_lang = {}, {}
    for r in res:
        for key, bucket in ((r["tier"], by_tier), (r["lang"], by_lang)):
            b = bucket.setdefault(key, dict(n=0, cer=0.0, lr=0.0, missed=0))
            b["n"] += 1
            b["cer"] += r["cer"]
            b["lr"] += r["line_recall"]
            b["missed"] += r["missed_lines"]
    def avg(d):
        return {k: dict(n=v["n"], cer_mean=round(v["cer"] / v["n"], 4),
                        line_recall_mean=round(v["lr"] / v["n"], 4), missed_lines_total=v["missed"])
                for k, v in d.items()}
    apo = {}
    for r in res:
        for k, v in r["apostrophe_confusion"].items():
            apo[k] = apo.get(k, 0) + v
    payload["summary_by_tier"] = avg(by_tier)
    payload["summary_by_lang"] = avg(by_lang)
    payload["apostrophe_confusion_total"] = apo
    print("\n== 分层 ==")
    for k, v in payload["summary_by_tier"].items():
        print(f"  {k:14s} n={v['n']:3d} CER={v['cer_mean']:.3f} 行召回={v['line_recall_mean']:.3f} 漏行={v['missed_lines_total']}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-set", default="tiny", choices=["tiny", "small", "v5det_tinyrec"])
    ap.add_argument("--oracle", action="store_true", help="用 PC 官方产线跑一遍")
    ap.add_argument("--android-report", help="读真机导出的 OCR JSON 做比对")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "scene_result.json"))
    args = ap.parse_args()

    if args.android_report:
        payload = run_android(args.android_report)
    elif args.oracle:
        payload = run_oracle(args.model_set)
    else:
        raise SystemExit("要么 --oracle 要么 --android-report <file>")

    payload = summarize(payload)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("写入", args.out)


if __name__ == "__main__":
    main()
