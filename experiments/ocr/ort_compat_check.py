"""阶段0-d：ONNX Runtime 兼容性实测。

同一个脚本分别在两个解释器下跑（ORT 1.31 与隔离的 ORT 1.27.0），
把每个模型的输出张量指纹与耗时写成 JSON，再由 compare 模式比对。
1.27.0 是本 App 现有 onnxruntime-1.27.0-classes.jar + libort4jrt.so 的同一核心版本。

用法：
    python ort_compat_check.py --out out/ort_<版本>.json
    python ort_compat_check.py --compare out/ort_1.31.0.json out/ort_1.27.0.json
"""

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# det 输入要求长宽为 32 的倍数；rec 固定高 48。
DET_SHAPE = (1, 3, 640, 640)
REC_SHAPE = (1, 3, 48, 320)
REPEATS = 3


def is_detection(name: str) -> bool:
    return "_det_" in name


def build_inputs(model_dir: str, shape) -> dict:
    rng = np.random.default_rng(20261016)
    data = rng.random(shape, dtype=np.float32)
    return {"x": data}


def fingerprint(a: np.ndarray) -> dict:
    # CTC/贪心解码只看每个位置的 argmax，浮点噪声是否改变文本以这条序列为准
    seq = np.argmax(a, axis=-1) if a.ndim >= 3 else np.argmax(np.squeeze(a))
    return {
        "shape": list(a.shape),
        "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "sum": float(np.sum(a, dtype=np.float64)),
        "argmax_first": int(np.argmax(a)) if a.size else -1,
        "argmax_seq_sha256": hashlib.sha256(np.ascontiguousarray(seq).tobytes()).hexdigest(),
        "argmax_seq_len": int(seq.size),
    }


def run_one(model_dir: str, dump_dir: str | None = None) -> dict:
    path = os.path.join(model_dir, "inference.onnx")
    shape = DET_SHAPE if is_detection(model_dir) else REC_SHAPE
    feeds = build_inputs(model_dir, shape)
    name = os.path.basename(model_dir)

    so = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # 本 App 的翻译引擎固定 BASIC_OPT（KitTranslationEngine.makeSession）；
    # 这里同时验证 OCR 图在各优化级别下的结果是否稳定，避免 Android 端踩同类坑。
    levels = {
        "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    per_level = {}
    for label, lvl in levels.items():
        opts = ort.SessionOptions()
        opts.graph_optimization_level = lvl
        s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        s.run(None, feeds)  # 预热
        lats = []
        out = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            out = s.run(None, feeds)
            lats.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(out[0])
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            np.save(os.path.join(dump_dir, f"{name}__{label}.npy"), arr)
        per_level[label] = {
            "mean_latency_ms": round(sum(lats) / len(lats), 2),
            "min_latency_ms": round(min(lats), 2),
            "output": fingerprint(arr),
        }

    return {
        "model": name,
        "inputs": [(i.name, i.type, [str(d) for d in i.shape]) for i in so.get_inputs()],
        "outputs": [(o.name, o.type, [str(d) for d in o.shape]) for o in so.get_outputs()],
        "levels": per_level,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--opt-level", default="DEFAULT")
    ap.add_argument("--dump", help="把每个级别的输出张量存成 npy，供跨版本逐元素比对")
    args = ap.parse_args()

    if args.compare:
        a, b = (json.load(open(p, encoding="utf-8")) for p in args.compare)
        print(f"== 比对 ORT {a['onnxruntime']} vs {b['onnxruntime']} ==")
        all_same = True
        for m in a["models"]:
            other = next(x for x in b["models"] if x["model"] == m["model"])
            for lvl, res in m["levels"].items():
                o_res = other["levels"][lvl]
                same = res["output"]["sha256"] == o_res["output"]["sha256"]
                same_idx = res["output"]["argmax_seq_sha256"] == o_res["output"]["argmax_seq_sha256"]
                all_same &= same
                print(
                    f"  {m['model']:26s} {lvl:11s} 逐字节一致={str(same):5s} argmax序列一致={str(same_idx):5s} "
                    f"延迟 {res['mean_latency_ms']}ms vs {o_res['mean_latency_ms']}ms"
                )
        print("结论：跨版本逐字节一致" if all_same else "结论：存在差异，见上表")
        return

    rows = []
    for d in sorted(glob.glob(os.path.join(MODELS, "*/"))):
        if not os.path.exists(os.path.join(d, "inference.onnx")):
            continue
        rows.append(run_one(d.rstrip("/\\"), args.dump))
        print(f"  loaded+ran {rows[-1]['model']}")

    payload = {
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "det_shape": list(DET_SHAPE),
        "rec_shape": list(REC_SHAPE),
        "repeats": REPEATS,
        "models": rows,
    }
    print("\n== 同一版本内、各图优化级别的结果一致性 ==")
    for m in rows:
        shas = {k: v["output"]["sha256"][:12] for k, v in m["levels"].items()}
        stable = len(set(shas.values())) == 1
        print(f"  {m['model']:26s} 级别结果一致={stable}  {shas if not stable else ''}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("写入", args.out)
    print("ORT", ort.__version__, "models:", len(rows))


if __name__ == "__main__":
    main()
