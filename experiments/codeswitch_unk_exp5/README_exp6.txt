EXP6 - 触发信号增强探索（结论：置信度触发不可行，60% 接近本组合上限）
=========================================================================

目的
----
EXP5 检出率 60%，漏检 12/30 分两类：
  a) 有 <unk> 但 EN 验证拼不对真值词（7 条）
  b) 无 <unk> 吞词：英文词被 ZH CTC 硬拼成中文音节（5 条）
EXP6 想补 b)：用 token 级置信度找"低置信中文音节"当触发信号。

尝试 A：token 置信度 -> 技术不可行
-----------------------------------
sherpa-onnx 1.13.7（与安卓 AAR 同版本）流式 CTC greedy 解码下：
  - OnlineRecognizerResult.ys_probs 为空列表
    （get_result_all / decode_streams / JSON 输出均验证为空）
  - OnlineRecognizer.ys_probs(stream) 同样返回空
  - modified_beam_search 不被流式 CTC 支持：
    "Unsupported decoding method: modified_beam_search for streaming CTC models"
结论：当前 sherpa 版本拿不到流式 CTC 的 token 级概率，
      "低置信触发"无 API 支撑。可复现证据见 diag_token_confidence.py。

尝试 B：miss 分类（0.60 pad，12 条）
-------------------------------------
NO-UNK   5 条  无触发信号，EN 从未被调用
              00023 topic->"他配" | 00158 base line->"斯兰" | 00427 group work 整句错乱
              00559 gt="t"、00570 gt="pre"（单字母/词缀，标注无产品意义，本来就不该指望）
EN-wrong 7 条  有 <unk> 且 EN 输出了英文实词，只是没拼对真值词：
              00030 positive->STATE/TRUSTE     00035 friends->FAMILY(近义)
              00078 app->MOBILE(近义)          00083 game/online->仅功能词
              00142 cpr->RP(缩写)              00406 postcard->POST(前缀)
              00428 group work->APROGRAM
结论：7 条中 6 条的触发区域确实是英文语音（EN 给的是近似/语义/部分词），
      真正"EN 完全没抓到"的只有 00083。

关键实验：EN 对纯中文的实词幻觉率 = 15/15
------------------------------------------
把 negative_zh 15 条整段直接喂 EN GigaSpeech：
  15/15 全部输出英文实词（HOW SHINDS I JUST / FANCY / CANNOT BEAR / POST ...）
含义：
  - EN 的"文本内容"完全不能用于区分"真英文区域"和"中文区域"，
    "EN 输出实词即检出"的任何变体在中文上都 100% 假阳性。
  - 但这对 EXP5 方案无影响：negative_zh 0/15 触发 unk -> EN 从未被调用，
    误报保持 0。精度完全由触发信号（ZH CTC <unk>）承担。

由此得到的产品方案边界
----------------------
ZH CTC <unk> 触发的 precision 高：
  30 positive 的触发位置都对应英文语音（含 EN-wrong 的 6 条），
  15 negative_zh 零触发。
因此可行的简化方案：
  出现 <unk> -> 按时间戳裁区域 -> 直接回填 EN 识别文本，
  不需要任何"EN 内容判定"（因为 EN 内容对中文不可区分，
  而触发本身已经足够可靠）。
预期行为：
  约 60% 的夹词句能回填与真值一致的词；另有 ~20% 回填近似/部分词
  （POST、FAMILY、MOBILE 这类对用户仍有提示意义）；
  ~17% 的吞词句（5/30）无触发，英文词在中文文本中丢失；
  纯中文句零干扰。
残余风险（进安卓前需真机验证）：
  - negative 只有 15 条且话题单一；噪音/咳嗽/第三方语言是否让
    ZH CTC 吐 <unk> 未验证 -> 真机上"无语音/噪音段"可能误触发。
  - 若该风险成立，可在回填前加一道轻量闸门（如区域语音能量/长度过滤）。

下一步选项
----------
1. 接受"unk 触发 + EN 文本直接回填"，进安卓做真机原型，
   重点测噪音/静音下的误触发率（本实验唯一未覆盖的风险）。
2. 若产品不接受 ~17% 吞词漏报，只能换更强的主语言模型
   （能区分英文词的中文模型）或独立 code-switch 检测模型，
   在本 sherpa 版本内没有免费的触发信号可挖。
