"""阶段1：把 OCR 权重与阶段0 golden 摆进 androidTest assets（不进 git）。

为什么需要这一步：
- 权重不入库是本仓既有约定（README_STEP22 "Do NOT place model weights in Git"），
  与 tools/setup_all.ps1 的做法一致：脚本负责拉取，仓库只存代码。
- **EXIF 归一化**：PC 侧 golden 用 cv2 解码（默认按 EXIF 转正），
  Android BitmapFactory 不认 EXIF。若直接拷原图，两侧坐标系会差一个 90° 旋转，
  对拍就会把"同一张图"当成两张图比。这里统一重编码为无 EXIF 的 upright 图。

用法：
    python experiments/ocr/prepare_android_assets.py            # 默认 8 印刷 + 4 手写
    python experiments/ocr/prepare_android_assets.py --hand 8   # 多带几张手写页
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(HERE, "models")
SAMPLES = os.path.join(HERE, "samples", "selected")
GOLDEN = os.path.join(HERE, "out", "golden_tiny_32.json")
DST = os.path.join(ROOT, "app", "src", "androidTest", "assets", "ocr")

MODEL_SET = {
    "det": ("PP-OCRv6_tiny_det_onnx", "models/det/inference.onnx"),
    "rec": ("PP-OCRv6_tiny_rec_onnx", "models/rec/inference.onnx"),
    "rec_yml": ("PP-OCRv6_tiny_rec_onnx", "models/rec/inference.yml"),
}

# 中/英冒烟测试用的是**真实照片**（Wikimedia Commons），不是合成图。
# 记录来源与许可，脚本负责拉取，assets/ocr 整体不入库。
SMOKE_IMAGES = [
    {
        "name": "smoke_zh_street_sign.png",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png",
        "lang": "zh",
        "expect_any": ["上海", "打浦路", "酒店"],
        "origin": "PaddleOCR 官方文档示例图（真实拍摄的中文路牌）",
        "license": "unverified-official-sample",
    },
    {
        "name": "smoke_en_street_sign.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Living_street_road_sign_Kyrenia.jpg",
        "lang": "en",
        # 牌面实际只有 "ZONE" 与限速 "20"（真机核对图片后修正：原期望词写错了），
        # 旁边的人形/房屋图标会被模型幻觉成字符，这条本身也是要记录的现象。
        "expect_any": ["ZONE", "20"],
        "origin": "Wikimedia Commons: File:Living street road sign Kyrenia.jpg（真实拍摄的英文路牌）",
        "license": "见文件页自由许可条款",
    },
]


def stage_smoke() -> list:
    out = []
    d = os.path.join(DST, "smoke")
    os.makedirs(d, exist_ok=True)
    for e in SMOKE_IMAGES:
        dst = os.path.join(d, e["name"])
        if not os.path.isfile(dst):
            req = urllib.request.Request(e["url"], headers={"User-Agent": "fourlang-ocr-phase1/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                open(dst, "wb").write(r.read())
        im = Image.open(dst)
        # 冒烟图降到长边 <=1280：官方 det 默认 max_side_limit=4000，
        # 一张 3969x5292 的原图会撑出 ~144MB 输入张量（阶段0 已把这条列为内存风险）。
        # 这里主动降采样，同时把"设备端必须先限幅"这件事固化进资产。
        long_side = max(im.size)
        if long_side > 1280:
            scale = 1280.0 / long_side
            im = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.LANCZOS,
            )
            if dst.endswith(".jpg"):
                im.convert("RGB").save(dst, quality=92)
            else:
                im.save(dst)
        im = Image.open(dst)
        out.append(dict(
            image=f"ocr/smoke/{e['name']}", lang=e["lang"], expect_any=e["expect_any"],
            origin=e["origin"], license=e["license"], source_url=e["url"],
            w=im.width, h=im.height, downscaled_from_long_side=min(long_side, 1280),
            original_long_side=long_side,
        ))
        print(f"  smoke {e['lang']}: {e['name']} {im.width}x{im.height}")
    return out


def stage_models() -> list:
    out = []
    for kind, (repo, rel) in MODEL_SET.items():
        src = os.path.join(MODELS, repo, os.path.basename(rel))
        if not os.path.isfile(src):
            sys.exit(f"缺少权重 {src}：先跑阶段0 的下载")
        dst = os.path.join(DST, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        out.append((rel, os.path.getsize(dst)))
    return out


def upright_copy(src_png: str, dst_png: str) -> dict:
    """按 EXIF 转正后重编码并抹掉 EXIF，返回最终尺寸。"""
    im = Image.open(src_png)
    fixed = ImageOps.exif_transpose(im)
    if fixed.mode not in ("RGB", "L"):
        fixed = fixed.convert("RGB")
    os.makedirs(os.path.dirname(dst_png), exist_ok=True)
    if dst_png.endswith(".jpg"):
        fixed.save(dst_png, quality=95)
    else:
        fixed.save(dst_png)
    return {"w": fixed.width, "h": fixed.height}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-n", type=int, default=8)
    ap.add_argument("--hand", type=int, default=4)
    args = ap.parse_args()

    if not os.path.isfile(GOLDEN):
        sys.exit(f"缺少 {GOLDEN}：先跑阶段0 的 run_pc_oracle.py")
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    manifest = {os.path.basename(e["image"]): e for e in
                json.load(open(os.path.join(HERE, "samples", "manifest.json"), encoding="utf-8"))}

    models = stage_models()

    picked = [r for r in golden["results"]
              if manifest.get(r["image"], {}).get("tier") == "printed_synth"][: args.print_n]
    picked += [r for r in golden["results"]
               if manifest.get(r["image"], {}).get("tier") == "handwritten_photo"][: args.hand]

    entries, skipped = [], 0
    for r in picked:
        name = r["image"]
        src = os.path.join(SAMPLES, name)
        if not os.path.isfile(src):
            skipped += 1
            continue
        dst = os.path.join(DST, "images", name)
        size = upright_copy(src, dst)
        if (size["w"], size["h"]) != (r["size"]["w"], r["size"]["h"]):
            print(f"  ! {name} 尺寸与 golden 不符 {size} vs {r['size']}（EXIF 处理不一致？）")
        entries.append({
            "image": f"ocr/images/{name}",
            "golden_image": name,
            "tier": manifest[name]["tier"],
            "w": size["w"], "h": size["h"],
            "gt_text": manifest[name]["gt_text"],
        })

    os.makedirs(os.path.join(DST, "golden"), exist_ok=True)
    shutil.copyfile(GOLDEN, os.path.join(DST, "golden", "golden_tiny_32.json"))

    smoke = stage_smoke()

    json.dump({
        "model_set": golden["model_set"],
        "det_model": "ocr/models/det/inference.onnx",
        "rec_model": "ocr/models/rec/inference.onnx",
        "rec_config": "ocr/models/rec/inference.yml",
        "golden": "ocr/golden/golden_tiny_32.json",
        "note": "图片已按 EXIF 转正并重编码，PC/Android 读到的像素一致",
        "images": entries,
        "smoke": smoke,
        "note_models_shared": "zh/en/uz 共用同一份 tiny det+rec 权重，语言只切换规范化规则",
    }, open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"权重: {models}")
    print(f"图片: {len(entries)} 张（跳过 {skipped}）→ {DST}")


if __name__ == "__main__":
    main()
