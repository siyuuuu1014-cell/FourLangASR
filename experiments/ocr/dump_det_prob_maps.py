"""阶段1-B：导出 PC 侧检测概率图，供 Android 数值对拍。

严格照搬官方 Kotlin 的 det 预处理（app/src/androidTest/java/.../ocr/paddle/
preprocess/DetPreprocessor.kt + util/ImageUtils.kt + util/MathUtils.kt）：
BGR 通道顺序、limit_type=min/limit_side_len=64/max_side_limit=4000、
自定义 round-half-to-even 取整到 32 的倍数、INTER_LINEAR、
scale=1/255 后按 **BGR 顺序**减 mean 除 std。

推理用 ORT_ENABLE_BASIC（与 Android 端一致；阶段0 已证明该级别下
ORT 1.27 与 1.31 输出逐字节一致）。

输出：app/src/androidTest/assets/ocr/probmaps/<图片名>.npy（float16，控制测试 APK 体积）
     + probmap_meta.json（每张图的输入尺寸、概率图统计、耗时）

用法：python experiments/ocr/dump_det_prob_maps.py
"""

import json
import math
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ASSETS = os.path.join(ROOT, "app", "src", "androidTest", "assets", "ocr")
DET_MODEL = os.path.join(ASSETS, "models", "det", "inference.onnx")
OUT_DIR = os.path.join(ASSETS, "probmaps")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LIMIT_SIDE_LEN = 64
LIMIT_TYPE = "min"
MAX_SIDE_LIMIT = 4000


def round_half_to_even(value: float) -> int:
    """镜像 Kotlin MathUtils.roundHalfToEven（不是 Python 内建 round 的行为）。"""
    floored = math.floor(value)
    diff = value - floored
    if diff < 0.5:
        return int(floored)
    if diff > 0.5:
        return int(floored) + 1
    return int(floored) if int(floored) % 2 == 0 else int(floored) + 1


def resize_dims(h: int, w: int):
    ratio = float(LIMIT_SIDE_LEN) / min(h, w) if min(h, w) < LIMIT_SIDE_LEN else 1.0
    new_h, new_w = int(h * ratio), int(w * ratio)
    if max(new_h, new_w) > MAX_SIDE_LIMIT:
        ratio = MAX_SIDE_LIMIT / max(new_h, new_w)
        new_h, new_w = int(new_h * ratio), int(new_w * ratio)
    new_h = max(round_half_to_even(new_h / 32.0) * 32, 32)
    new_w = max(round_half_to_even(new_w / 32.0) * 32, 32)
    return new_h, new_w


def preprocess(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    new_h, new_w = resize_dims(h, w)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    f = resized.astype(np.float32) / 255.0
    for c in range(3):
        f[:, :, c] = (f[:, :, c] - MEAN[c]) / STD[c]
    chw = np.transpose(f, (2, 0, 1))[np.newaxis, ...].copy()
    return chw, (h, w), (new_h, new_w)


def main():
    if not os.path.isfile(DET_MODEL):
        raise SystemExit("先跑 prepare_android_assets.py")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(DET_MODEL, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(os.path.join(ASSETS, "manifest.json"), encoding="utf-8"))
    meta = []
    for e in manifest["images"]:
        path = os.path.join(ASSETS, e["image"].split("/", 1)[1])  # "ocr/images/x.png" -> assets/images/x.png
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"解码失败: {path}")
        tensor, (oh, ow), (rh, rw) = preprocess(bgr)
        t0 = time.perf_counter()
        prob = sess.run(None, {in_name: tensor})[0]
        ms = round((time.perf_counter() - t0) * 1000, 1)
        prob = np.squeeze(np.asarray(prob))
        name = os.path.basename(e["image"])
        prob.astype("<f4").tofile(os.path.join(OUT_DIR, name + ".bin"))
        meta.append(dict(
            image=name, orig_h=oh, orig_w=ow, resized_h=rh, resized_w=rw,
            prob_shape=list(prob.shape), prob_min=float(prob.min()), prob_max=float(prob.max()),
            prob_mean=round(float(prob.mean()), 6),
            above_thresh_0_3=int((prob > 0.3).sum()), det_infer_ms=ms,
        ))
        print(f"  {name:16s} {ow}x{oh} -> {rw}x{rh} prob{list(prob.shape)} {ms}ms")

    json.dump(dict(
        det_model=os.path.relpath(DET_MODEL, ROOT).replace("\\", "/"),
        det_model_sha256=__import__("hashlib").sha256(open(DET_MODEL, "rb").read()).hexdigest(),
        optimization="ORT_ENABLE_BASIC",
        onnxruntime=ort.__version__,
        limit_side_len=LIMIT_SIDE_LEN, limit_type=LIMIT_TYPE, max_side_limit=MAX_SIDE_LIMIT,
        mean=list(MEAN), std=list(STD), channel_order="BGR",
        note="概率图为 float32 小端裸数据，形状见 prob_shape",
        images=meta,
    ), open(os.path.join(ASSETS, "probmap_meta.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for stale in os.listdir(OUT_DIR):
        if stale.endswith(".npy"):
            os.remove(os.path.join(OUT_DIR, stale))
    total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print(f"概率图 {len(meta)} 张，合计 {total/1048576:.1f}MB → {OUT_DIR}")


if __name__ == "__main__":
    main()
