# 真实场景 OCR 评测（招牌 / 路牌 / 商品包装）

**规矩**：合成印刷体（阶段0 的 `printed_synth`）的成绩**不等于**真实街景准确率，
两者不得混用、不得互相代表。本目录只接受真实拍摄的照片。

## 目录

```
scene_eval/
  images/       照片（任意分辨率；评测脚本不改写原图）
  labels.csv    标注
  out/          评测结果 json（自动生成）
```

`images/` 与 `labels.csv` 都不入库（照片含第三方版权与个人信息风险，见根 `.gitignore`）。

## labels.csv 格式

```csv
image,lang,tier,text,notes
001_shopfront_tashkent.jpg,uz,storefront,"SOAT DO'KONI | 09:00-18:00 | CHIQISH","逆光，轻微倾斜"
002_bus_stop.jpg,uz,sign,"AVTOBUS BEKATI | 20",""
```

| 列 | 说明 |
|---|---|
| `image` | `images/` 下的文件名 |
| `lang` | `uz` / `zh` / `en`（决定用哪套文本规范化，与翻译目标语言无关） |
| `tier` | `sign`（路牌指示牌）/ `storefront`（店面招牌）/ `package`（商品包装）/ `menu` / `screen`（拍屏幕） |
| `text` | 期望文本，**按视觉行**用 ` \| ` 分隔（行级召回要靠这个） |
| `notes` | 可选：光照、倾角、模糊、手写体等，便于分层复盘 |

## 跑法

```bash
# A. PC 官方产线基线（也用来复核标注是否写错）
python experiments/ocr/eval_scene.py --model-set tiny --oracle
python experiments/ocr/eval_scene.py --model-set small --oracle   # tiny vs small 对比

# B. 真机导出的结果（阶段2 的导出接口产出后）
python experiments/ocr/eval_scene.py --android-report device_ocr.json
```

## 输出指标

- **CER**：去空格、去分隔性标点后的字符编辑距离 / 期望字符数（与阶段0 同口径）
- **整行召回 `line_recall`**：期望行是否作为子串出现在输出里
- **漏检行 `missed_lines`**：未被任何输出行覆盖的期望行数
- **多余行 `extra`**：输出行数超出期望的部分（真实照片常见于图标/背景被幻觉成文字，
  阶段1 已在英文路牌上实测复现：人形与房屋图标被识别成 `自秋白`、`IETRO:OI`）
- **撇号混淆矩阵**：期望里的 `ʻ / ‘ / '` 各被输出成什么（乌兹别克语重点，
  已知 PP-OCRv6 字典不含 U+02BB，模型只会给 ASCII 或干脆丢失）
- 按 `tier` 与 `lang` 分层汇总

## 建议的样本量（阶段2 扩到 ≥100 张）

| tier | 最少 | 需要覆盖的变量 |
|---|---|---|
| sign | 25 | 直射/阴影/逆光、倾角 0–30°、距离 0.5–3 m |
| storefront | 25 | 字体多样、立体字、夜间霓虹、反光玻璃 |
| package | 25 | 曲面、透明塑封反光、小字号(<20px 字高) |
| menu | 15 | 多列排版、密集行 |
| screen | 10 | 摩尔纹、色偏 |

`uz` 至少占 60%，其余 `zh/en` 用于验证共享基础模型的多语言能力。
