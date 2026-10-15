"""阶段0-f：样本集构建。

两个来源，分层明确标注，避免把合成当真实：

1) handwritten_photo —— 第三方数据集 abduazizovanozima7/uzbek-ocr-dataset 的真实拍摄
   手写作业照片（蓝黑钢笔、横线纸、含透视/模糊/阴影），标注为整页段落文本。
2) printed_synth —— 用同一批真实乌兹别克语句子渲染的印刷体图，再施加相机式降质
   （旋转、透视、模糊、噪声、光照渐变、JPEG），**合成样本**，只用于覆盖印刷体层级。

输出 samples/manifest.json：每条含 image / tier / gt_text / gt_apostrophes / 尺寸。
"""

import csv
import hashlib
import json
import os
import random
import unicodedata

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "samples", "raw")
OUT = os.path.join(HERE, "samples", "selected")
APO_KEYS = {"U+0027": "'", "U+2018": "‘", "U+02BB": "ʻ", "U+02BC": "ʼ"}

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]


def apostrophe_profile(text: str) -> dict:
    return {k: text.count(v) for k, v in APO_KEYS.items()}


def pick_handwritten(n: int, rng: random.Random) -> list:
    rows = list(csv.DictReader(open(os.path.join(RAW, "final_dataset.csv"), encoding="utf-8-sig")))
    # 按面积/色彩模式分层：大页 / 小页 / RGBA(png) 各取一部分
    recs = []
    for r in rows:
        p = os.path.join(RAW, r["image_path"].replace("/", os.sep))
        if not os.path.isfile(p):
            continue
        try:
            im = Image.open(p)
            recs.append((r, p, im.width * im.height, im.mode))
        except Exception:
            continue
    big = [x for x in recs if x[2] > 900_000]
    small = [x for x in recs if x[2] <= 900_000]
    rgba = [x for x in recs if x[3] == "RGBA"]
    buckets = [("large", big), ("small", small), ("rgba", rgba)]
    chosen, seen = [], set()
    per = max(1, n // len(buckets))
    for _, pool in buckets:
        rng.shuffle(pool)
        for r, p, area, mode in pool:
            if r["image_path"] in seen or len([c for c in chosen if c["bucket"] == _]) >= per:
                break
            seen.add(r["image_path"])
            chosen.append(dict(bucket=_, src=p, gt=r["text"], area=area, mode=mode))
    while len(chosen) < n:
        rng.shuffle(recs)
        for r, p, area, mode in recs:
            if r["image_path"] not in seen:
                seen.add(r["image_path"])
                chosen.append(dict(bucket="extra", src=p, gt=r["text"], area=area, mode=mode))
                break
        else:
            break
    return chosen[:n]


def render_printed(text: str, rng: random.Random, w: int = 1000):
    """返回 (图像, 真实换行后的行列表) —— 行级 GT 只有渲染器自己知道。"""
    font_path = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
    font = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if len(trial) * 15 > w - 60:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lines = [l for l in lines if l.strip()]
    h = 80 + 42 * len(lines)
    im = Image.new("RGB", (w, h), (247, 246, 243))
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text((30, 40 + 42 * i), ln, fill=(25, 25, 30), font=font)
    return im, lines


def degrade(im: Image.Image, rng: random.Random) -> Image.Image:
    """温和、可核验的相机式降质。

    上一版用 Image.QUAD 传了错误的 8 参数（把文字压成不可读形状），
    导致 3/8 印刷样本出现 CER≈1.0 的"假失败"。这里改成仿射错切 + 轻微旋转，
    并在生成后由 verify 步骤确认文字仍在画面内。
    """
    im = im.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True,
                   fillcolor=(247, 246, 243))
    # 透视感用错切近似：水平/垂直各 ±3%，不改变内容可读性
    w, h = im.size
    sx = rng.uniform(-0.03, 0.03)
    sy = rng.uniform(-0.02, 0.02)
    im = im.transform((w, h), Image.AFFINE, (1, sx, -sx * h * 0.5, sy, 1, 0),
                      resample=Image.BICUBIC, fillcolor=(247, 246, 243))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    # 光照渐变：亮度范围收窄，避免把深色文字进一步压黑
    grad = Image.new("L", im.size)
    gd = grad.load()
    for y in range(im.height):
        row = int(10 * y / max(1, im.height))
        for x in range(im.width):
            gd[x, y] = int(12 * x / max(1, im.width)) + row
    from PIL import ImageChops
    im = ImageChops.subtract(im.convert("RGB"), Image.merge("RGB", (grad, grad, grad)))
    # 传感器噪声
    noise = Image.effect_noise(im.size, 12).convert("L")
    im = ImageChops.add(im, im, scale=1.0, offset=0)
    im = Image.blend(im, Image.merge("RGB", (noise, noise, noise)), 0.05)
    return im


def main():
    rng = random.Random(20261016)
    os.makedirs(OUT, exist_ok=True)
    n_hand, n_print = 24, 8
    entries = []

    for i, rec in enumerate(pick_handwritten(n_hand, rng)):
        dst = os.path.join(OUT, f"hand_{i:02d}{os.path.splitext(rec['src'])[1]}")
        Image.open(rec["src"]).convert("RGB").save(dst, quality=92)
        entries.append(dict(
            image=os.path.relpath(dst, HERE).replace("\\", "/"),
            tier="handwritten_photo", origin="third_party_dataset",
            bucket=rec["bucket"], gt_text=rec["gt"],
            gt_apostrophes=apostrophe_profile(rec["gt"]),
            sha256=hashlib.sha256(open(dst, "rb").read()).hexdigest()[:16],
        ))

    pool = [r["text"] for r in csv.DictReader(open(os.path.join(RAW, "final_dataset.csv"), encoding="utf-8-sig"))]
    # 印刷体优先挑含多种撇号写法的句子
    pool.sort(key=lambda t: -(min(t.count("'"), 5) + min(t.count("‘"), 5) + (10 if "ʻ" in t else 0)))
    for i, txt in enumerate(pool[:n_print]):
        seg = txt[:260]
        im_raw, gt_lines = render_printed(seg, rng)
        im = degrade(im_raw, rng)
        dst = os.path.join(OUT, f"print_{i:02d}.png")
        im.save(dst)
        entries.append(dict(
            image=os.path.relpath(dst, HERE).replace("\\", "/"),
            tier="printed_synth", origin="synthetic_render_of_real_text",
            bucket="print", gt_text=seg, gt_lines=gt_lines,
            gt_apostrophes=apostrophe_profile(seg),
            sha256=hashlib.sha256(open(dst, "rb").read()).hexdigest()[:16],
        ))

    json.dump(entries, open(os.path.join(HERE, "samples", "manifest.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    tot = {}
    for e in entries:
        tot[e["tier"]] = tot.get(e["tier"], 0) + 1
    print("样本:", tot, "合计", len(entries))
    print("撇号总量(所有 GT):", {k: sum(e["gt_apostrophes"][k] for e in entries) for k in APO_KEYS})


if __name__ == "__main__":
    main()
