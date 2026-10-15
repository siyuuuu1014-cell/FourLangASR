"""阶段0-c：PC 端完整 Det+Rec 参考实现（官方 PaddleOCR 产线 + 官方 ONNX 权重）。

产出的 JSON 就是后续 Android 侧对拍的 golden 文件：
每行包含四角坐标（原图坐标系）、文本、置信度、顺序号，另存原图尺寸与 EXIF 方向。

    .tools/venv-paddle/Scripts/python.exe run_pc_oracle.py \
        --image samples/x.png --tier print --model-set tiny --out out/golden_tiny.json

--model-set 支持 tiny / small / v5det_tinyrec（用 PP-OCRv5 检测器替换的对照组合）。
"""

import argparse
import glob
import json
import os
import time

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

MODEL_SETS = {
    "tiny": ("PP-OCRv6_tiny_det_onnx", "PP-OCRv6_tiny_rec_onnx"),
    "small": ("PP-OCRv6_small_det_onnx", "PP-OCRv6_small_rec_onnx"),
    "v5det_tinyrec": ("PP-OCRv5_mobile_det_onnx", "PP-OCRv6_tiny_rec_onnx"),
}


def build(model_set: str, threads: int) -> PaddleOCR:
    det, rec = MODEL_SETS[model_set]
    # 与官方 Android 产线一致：只跑 det + rec，不开整页方向分类/版面矫正/单行方向分类
    return PaddleOCR(
        # paddlex 会校验 model_dir 里的 yml Global.model_name 与名字一致，两者都要给
        text_detection_model_name=os.path.basename(det).replace("_onnx", ""),
        text_recognition_model_name=os.path.basename(rec).replace("_onnx", ""),
        text_detection_model_dir=os.path.join(MODELS, det),
        text_recognition_model_dir=os.path.join(MODELS, rec),
        engine="onnxruntime",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        cpu_threads=threads,
    )


def image_size_and_orientation(path: str) -> dict:
    im = Image.open(path)
    exif = {}
    try:
        ex = im.getexif()
        if ex:
            exif = {"orientation": ex.get(274), "exif_keys": sorted(ex.keys())}
    except Exception:
        pass
    # PIL 按 EXIF 转正后的尺寸才是 OCR 坐标系
    fixed = ImageOps.exif_transpose(im)
    return {"w": fixed.width, "h": fixed.height, "raw_w": im.width, "raw_h": im.height, "exif": exif}


def quad_of(poly) -> list:
    a = np.asarray(poly, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in a]


def run_one(ocr: PaddleOCR, path: str) -> dict:
    t0 = time.perf_counter()
    res = list(ocr.predict(path))
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    lines = []
    for r in res:
        d = r if isinstance(r, dict) else getattr(r, "res", {})
        texts = d.get("rec_texts", []) or []
        scores = d.get("rec_scores", []) or []
        polys = d.get("rec_polys", d.get("dt_polys", [])) or []
        for i, txt in enumerate(texts):
            lines.append({
                "order": i,
                "text": txt,
                "confidence": round(float(scores[i]), 4) if i < len(scores) else None,
                "quad": quad_of(polys[i]) if i < len(polys) else None,
            })
    return {
        "image": os.path.basename(path),
        "size": image_size_and_orientation(path),
        "wall_ms": wall_ms,
        "lines": lines,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--model-set", default="tiny", choices=sorted(MODEL_SETS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--tier", default="unlabeled", help="样本分层名，仅记录用")
    args = ap.parse_args()

    paths = []
    for p in args.image:
        hits = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        paths += [h for h in hits if os.path.isfile(h)]

    ocr = build(args.model_set, args.threads)
    payload = {
        "oracle": "paddleocr PaddleOCR(engine=onnxruntime)",
        "model_set": args.model_set,
        "det_dir": MODEL_SETS[args.model_set][0],
        "rec_dir": MODEL_SETS[args.model_set][1],
        "threads": args.threads,
        "tier": args.tier,
        "results": [run_one(ocr, p) for p in paths],
    }
    out = args.out or os.path.join(HERE, "out", f"golden_{args.model_set}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    total = sum(len(r["lines"]) for r in payload["results"])
    print(f"{args.model_set}: {len(paths)} 图 / {total} 行 → {out}")


if __name__ == "__main__":
    main()
