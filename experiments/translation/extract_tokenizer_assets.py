"""从 sentencepiece.bpe.model 提取词表 + 归一化 charsmap，生成 Android 端 Kotlin tokenizer 资产。

经 oracle（sentencepiece 0.2.2）对拍确认的模型事实：
  1. vocab.json = HF 官方 m2m100_418M vocab.json（md5 一致）。
     词表 id 是 frequency 序，与 spm id 不同 —— 必须经 vocab.json 重映射。
     其中 390 个 spm pieces 不在 vocab.json（官方即如此，无 embedding），
     编码时映射为 unk id（3），与 HF M2M100Tokenizer._convert_token_to_id 一致。
  2. BPE 算法（sentencepiece>=0.1.99 bpe_model.cc SampleEncode）：
     初始符号 = 归一化文本逐字符（▁ 是独立基符号）。
     合并 = 按 piece score 优先的 agenda（优先队列）：任意相邻对 (a,b)，
     只要 a+b 是词表 piece 就可合并；每次取 score 最高者（同分取左位置小者），
     合并后把新相邻对入队。无固定 merge 表。
  3. 归一化 = normalizer_spec.precompiled_charsmap（Darts double-array trie，
     237KB，nmt_nfkc 规则 + NFKC 类组合规则），加上
     Normalizer::Normalize 的空白折叠/trim/空格→▁/dummy prefix 管线。
     本脚本把 trie 原样打包，Kotlin 侧移植 darts-clone commonPrefixSearch。
  4. lang tokens 128004..128103（__af__..__zu__）、madeup words
     128104..128111（<madeupword0>..<madeupword7>）不在 vocab.json，
     detok 表由本脚本补齐。

输出：
  app/src/main/assets/m2m100/tokenizer.bin        —— 紧凑二进制资产
  experiments/translation/tokenizer_test_vectors.json —— Kotlin 单测对拍向量
  experiments/translation/normalizer_test_vectors.json —— Kotlin 归一化对拍向量
"""

import heapq
import json
import random
import struct
import sys
import unicodedata
from pathlib import Path

import sentencepiece as spm

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PKG = ROOT / "translation_pkg"

SPM_MODEL = PKG / "sentencepiece.bpe.model"
VOCAB_JSON = PKG / "vocab.json"
ADDED_TOKENS = PKG / "added_tokens.json"

ASSET_DIR = ROOT / "app" / "src" / "main" / "assets" / "m2m100"
ASSET_FILE = ASSET_DIR / "tokenizer.bin"
VECTORS_FILE = BASE / "tokenizer_test_vectors.json"
NORM_VECTORS_FILE = BASE / "normalizer_test_vectors.json"

UNK_PIECE = "<unk>"
DUMMY = "\u2581"
MADEUP_WORDS = [f"<madeupword{i}>" for i in range(8)]  # ids 128104..128111


# ---------------------------------------------------------------- proto 解析（提取 precompiled_charsmap）

def parse_fields(buf):
    i = 0
    out = {}
    while i < len(buf):
        tag = 0
        shift = 0
        while True:
            b = buf[i]
            i += 1
            tag |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        field = tag >> 3
        wt = tag & 7
        if wt == 0:
            v = 0
            shift = 0
            while True:
                b = buf[i]
                i += 1
                v |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            out.setdefault(field, []).append(("v", v))
        elif wt == 2:
            ln = 0
            shift = 0
            while True:
                b = buf[i]
                i += 1
                ln |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            out.setdefault(field, []).append(("b", i, ln))
            i += ln
        elif wt == 5:
            out.setdefault(field, []).append(("f", buf[i:i + 4]))
            i += 4
        else:
            raise ValueError("wt %d" % wt)
    return out


def extract_charsmap(proto_bytes):
    """返回 (units_bytes, normalized_str)。blob = <uint32 units_len><units><normalized>。"""
    top = parse_fields(proto_bytes)
    (t, off, ln) = top[3][0]  # normalizer_spec
    norm = parse_fields(proto_bytes[off:off + ln])
    (t2, off2, ln2) = norm[2][0]  # precompiled_charsmap（off2 相对 normalizer_spec）
    blob = proto_bytes[off + off2:off + off2 + ln2]
    units_len = struct.unpack("<I", blob[:4])[0]
    units = blob[4:4 + units_len]
    normalized = blob[4 + units_len:]
    assert len(units) % 4 == 0
    return units, normalized


# ---------------------------------------------------------------- darts-clone double-array（Python 移植）

MASK32 = 0xFFFFFFFF


class Darts:
    """darts-clone 0.32 DoubleArray 的 commonPrefixSearch 移植（little-endian uint32 units）。"""

    def __init__(self, units_bytes, normalized_str):
        self.units = list(struct.unpack("<%dI" % (len(units_bytes) // 4), units_bytes))
        self.normalized = normalized_str

    @staticmethod
    def offset_of(unit):
        return ((unit >> 10) << ((unit & (1 << 9)) >> 6)) & MASK32

    @staticmethod
    def label_of(unit):
        return unit & (0x80000000 | 0xFF)

    @staticmethod
    def has_leaf(unit):
        return ((unit >> 8) & 1) == 1

    @staticmethod
    def value_of(unit):
        return unit & 0x7FFFFFFF

    def common_prefix_search(self, key_bytes, max_len=64):
        results = []
        n_units = len(self.units)
        node_pos = 0
        if n_units == 0:
            return results
        unit = self.units[0]
        node_pos = (node_pos ^ self.offset_of(unit)) & MASK32
        for i, b in enumerate(key_bytes[:max_len]):
            node_pos = (node_pos ^ b) & MASK32
            if node_pos >= n_units:
                break
            unit = self.units[node_pos]
            if self.label_of(unit) != b:
                break
            node_pos = (node_pos ^ self.offset_of(unit)) & MASK32
            if self.has_leaf(unit):
                if node_pos < n_units:
                    results.append((self.value_of(self.units[node_pos]), i + 1))
        return results

    def normalize_prefix(self, text):
        """等价于 Normalizer::NormalizePrefix（无 user-defined matcher、无非法 UTF-8 处理）。
        返回 (replacement, consumed_chars)。"""
        key = text.encode("utf-8")
        results = self.common_prefix_search(key)
        longest = 0
        value = 0
        for v, ln in results:
            if ln > longest:
                longest = ln
                value = v
        if longest == 0:
            # 无 trie 规则：单码点原样通过
            return text[0], 1
        # replacement = normalized[value: 到 NUL]（normalized 为 bytes）
        end = self.normalized.find(b"\x00", value)
        if end < 0:
            end = len(self.normalized)
        rep = self.normalized[value:end].decode("utf-8")
        matched_chars = len(key[:longest].decode("utf-8"))
        return rep, matched_chars


# ---------------------------------------------------------------- Normalizer::Normalize 镜像

def normalize_mirror(text, darts):
    """复刻 normalizer.cc Normalize（add_dummy_prefix=true, remove_extra_whitespaces=true,
    escape_whitespaces=true, treat_whitespace_as_suffix=false）。"""
    out = []
    # 头部空格跳过
    while text:
        rep, consumed = darts.normalize_prefix(text)
        if rep != " ":
            break
        text = text[consumed:]
    if not text:
        return ""
    out.append(DUMMY)  # dummy prefix
    is_prev_space = True
    while text:
        rep, consumed = darts.normalize_prefix(text)
        sp = rep
        # 前一段以空格结尾时，去掉本段头部空格
        while is_prev_space and sp.startswith(" "):
            sp = sp[1:]
        if sp:
            out.append(sp.replace(" ", DUMMY))
            is_prev_space = sp.endswith(" ")
        text = text[consumed:]
    s = "".join(out)
    # 尾部空格去除
    while s.endswith(DUMMY):
        s = s[:-len(DUMMY)]
    return s


# ---------------------------------------------------------------- score 驱动的 agenda BPE（与 Kotlin 移植版同构）

def is_mark(c: str) -> bool:
    """Unicode 类别 M（Mn/Mc/Me）。与 spm 0.2.2 的 unk 符号聚合判定一致（实测）。"""
    return unicodedata.category(c).startswith("M")


class PyBPE:
    def __init__(self, piece_list, scores):
        self.piece_to_idx = {p: i for i, p in enumerate(piece_list)}
        self.scores = scores

    def split_symbols(self, word: str):
        """符号切分（spm 0.2.2 实测规则）：
        逐字符；未知字符（不在词表）吞并其后紧邻的、同样不在词表的组合标记（类别 M）
        作为一个符号。已知字符（词表 piece）单独成符号。
        """
        syms = []
        i = 0
        n = len(word)
        while i < n:
            c = word[i]
            syms.append(c)
            i += 1
            if c not in self.piece_to_idx:
                while i < n and word[i] not in self.piece_to_idx and is_mark(word[i]):
                    syms[-1] += word[i]
                    i += 1
        return syms

    def encode_word(self, word: str):
        syms = [DUMMY] + self.split_symbols(word)
        n = len(syms)
        alive = [True] * n
        heap = []

        def concat_is_piece(l: int, r: int) -> bool:
            return (syms[l] + syms[r]) in self.piece_to_idx

        for i in range(n - 1):
            if concat_is_piece(i, i + 1):
                merged = syms[i] + syms[i + 1]
                heapq.heappush(heap, (-self.scores[self.piece_to_idx[merged]], i, len(merged), merged))

        while heap:
            neg_score, left, size, merged = heapq.heappop(heap)
            right = left + 1
            while right < n and not alive[right]:
                right += 1
            if right >= n or not alive[left]:
                continue
            if len(syms[left]) + len(syms[right]) != size:
                continue
            syms[left] = merged
            alive[right] = False
            prev = left - 1
            while prev >= 0 and not alive[prev]:
                prev -= 1
            if prev >= 0 and concat_is_piece(prev, left):
                m = syms[prev] + syms[left]
                heapq.heappush(heap, (-self.scores[self.piece_to_idx[m]], prev, len(m), m))
            nxt = right + 1
            while nxt < n and not alive[nxt]:
                nxt += 1
            if nxt < n and concat_is_piece(left, nxt):
                m = syms[left] + syms[nxt]
                heapq.heappush(heap, (-self.scores[self.piece_to_idx[m]], left, len(m), m))

        out = []
        for i in range(n):
            if alive[i]:
                # 未知名保留原字符串（spm 行为：piece 字符串原样、id 为 unk）
                out.append(syms[i])
        return out

    def encode(self, normalized: str):
        words = normalized.split(DUMMY)
        pieces = []
        for w in words:
            if w:
                pieces.extend(self.encode_word(w))
        return pieces


# ---------------------------------------------------------------- 主流程

def main():
    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    n = sp.GetPieceSize()
    print(f"spm piece size = {n}")

    piece_list = [sp.IdToPiece(i) for i in range(n)]
    scores = [sp.GetScore(i) for i in range(n)]

    with open(VOCAB_JSON, encoding="utf-8") as f:
        piece_to_model = json.load(f)
    orphans = [p for p in piece_list if p not in piece_to_model]
    print(f"vocab.json entries = {len(piece_to_model)}; orphan pieces (->unk) = {len(orphans)}")

    proto = sp.serialized_model_proto()
    units, normalized = extract_charsmap(proto)
    darts = Darts(units, normalized)
    print(f"charsmap units = {len(units) // 4}, normalized string = {len(normalized)} bytes")

    bpe = PyBPE(piece_list, scores)

    sentences = [
        "今天天气怎么样？", "你好，很高兴认识你。", "我想预订一张去北京的机票。",
        "这个手机多少钱？", "会议将在明天上午九点开始。", "请把盐递给我。",
        "他每天都骑自行车上班。", "上海是一个国际化大都市。", "这本书非常有趣。",
        "医生建议我多喝水。", "一百二十三万四千五百六十七", "2024年8月15日下午3点",
        "What is the weather like today?", "Hello, nice to meet you.",
        "I would like to book a flight to Beijing.", "How much does this phone cost?",
        "The meeting will start at nine tomorrow morning.", "Please pass me the salt.",
        "He rides his bike to work every day.", "unbelievable internationalization",
        "3.141592653589793", "COVID-19 is still around.",
        "Какая сегодня погода?", "Здравствуйте, очень приятно познакомиться.",
        "Я хочу забронировать билет в Пекин.", "Сколько стоит этот телефон?",
        "Собрание начнётся завтра в девять утра.", "Передайте мне соль, пожалуйста.",
        "Он каждый день ездит на работу на велосипеде.", "неправительственный интернационализация",
        "Bugun ob-havo qanday?", "Salom, tanishganimdan xursandman.",
        "Men Pekinga chipta band qilmoqchiman.", "Bu telefon qancha turadi?",
        "Yig'ilish ertaga ertalab soat to'qqizda boshlanadi.", "Iltimos, tuzni uzating.",
        "U har kuni ishga velosipedda boradi.", "mustaqillik bayrami",
        "iPhone 15 Pro Max 256GB 黑色", "Смешанный текст with English words 和中文",
        "a  b\tc\nd", "  leading and trailing  ", "!!!???...", "😀😃🙃",
        "——破折号——", "«кавычки»", "¿Qué?", "Ｆｕｌｌｗｉｄｔｈ", "0xFF 0b101",
        "e-mail test@example.com", "第1名、第2名、第3名", "C++ Python3.11 Java/Kotlin",
        "㍿　ＡＢＣ　①", "a\u00a0b\u200b\u3000c", "e\u0301 ﬁ ² ①",
        "Цæгат Ирыстон", "Вьетнам Người Việt", "хәл-әхвәл",
        "Кыргызстан Ўзбекистон", "Гуанчжоу wǎngzhàn", "https://example.com/page",
        "a\u0301e\u0300i\u0302", "დ\u10d0", "င့္", "ႏွင့္", "\u0915\u094d\u0937",
    ]

    corpus = list(piece_list)
    corpus += [" " + p for p in piece_list]
    corpus += sentences
    # 多字符 trie key 压力测试：常见字母基 × 组合符号
    combining = [chr(cp) for cp in range(0x300, 0x370)]
    combining += [chr(cp) for cp in range(0x1AB0, 0x1AC0)]
    combining += [chr(cp) for cp in range(0xFE20, 0xFE30)]
    bases = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
             + [chr(cp) for cp in range(0x410, 0x450)]
             + [chr(cp) for cp in range(0x391, 0x3D0)]
             + [chr(cp) for cp in range(0x1000, 0x1100)]
             + [chr(cp) for cp in range(0x900, 0x980)])
    for b in bases:
        for m in combining:
            corpus.append("a" + b + m + "z")
    print(f"对拍语料 = {len(corpus)} 条 ...")

    # 归一化对拍
    norm_bad = 0
    for text in corpus:
        mine = normalize_mirror(text, darts)
        oracle = sp.Normalize(text)
        if mine != oracle:
            norm_bad += 1
            if norm_bad <= 15:
                print(f"NORM MISMATCH {text!r}: mine={mine!r} oracle={oracle!r}")
    print(f"normalize mismatch = {norm_bad} / {len(corpus)}")

    # 完整管线对拍
    mismatch = 0
    for text in corpus:
        mine_norm = normalize_mirror(text, darts)
        oracle = sp.Encode(text, out_type=str)
        mine = bpe.encode(mine_norm)
        if oracle != mine:
            mismatch += 1
            if mismatch <= 15:
                print(f"MISMATCH text={text!r}")
                print(f"  norm   : {mine_norm!r}")
                print(f"  oracle : {oracle}")
                print(f"  mine   : {mine}")
    print(f"pipeline mismatch = {mismatch} / {len(corpus)}")

    if norm_bad or mismatch:
        print("对拍失败，不写资产。")
        sys.exit(1)

    # ---- 写资产 ----
    # 结构（little-endian）：
    #   magic "M2M3" | version int32 | pieceCount int32
    #   | detokCount int32 | charsmapUnits int32 | normalizedBytes int32 | poolBytes int32
    #   | pieces: pieceCount × {uint32 poolOff, uint32 byteLen, int32 modelId, float32 score}
    #   | detok: detokCount × uint32 poolOff
    #   | charsmap units: charsmapUnits × uint32
    #   | charsmap normalized: bytes
    #   | stringPool bytes
    pool = {}
    pool_bytes = bytearray()

    def intern(s: str) -> int:
        off = pool.get(s)
        if off is None:
            off = len(pool_bytes)
            pool[s] = off
            pool_bytes.extend(s.encode("utf-8"))
        return off

    piece_pool_off = [intern(p) for p in piece_list]

    with open(ADDED_TOKENS, encoding="utf-8") as f:
        added = json.load(f)
    lang_tokens = sorted(added.items(), key=lambda kv: kv[1])
    max_model = 128111
    detok_off = [0] * (max_model + 1)
    for piece, mid_s in piece_to_model.items():
        detok_off[int(mid_s)] = intern(piece)
    for tok, mid in lang_tokens:
        detok_off[int(mid)] = intern(tok)
    for i, mw in enumerate(MADEUP_WORDS):
        detok_off[128104 + i] = intern(mw)

    n_units = len(units) // 4

    # Unicode 类别 M（Mn/Mc/Me）码点区间（供 Kotlin 端 unk 符号聚合判定）
    mark_ranges = []
    range_start = None
    prev = -2
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        m = unicodedata.category(chr(cp)).startswith("M")
        if m and range_start is None:
            range_start = cp
        elif not m and range_start is not None:
            mark_ranges.append((range_start, prev))
            range_start = None
        prev = cp
    if range_start is not None:
        mark_ranges.append((range_start, prev))
    mark_blob = b"".join(struct.pack("<II", a, b) for a, b in mark_ranges)
    print(f"mark ranges = {len(mark_ranges)}")

    header = struct.pack(
        "<4siiiiiii", b"M2M3", 3,
        n, max_model + 1, n_units, len(normalized), len(mark_ranges), len(pool_bytes),
    )
    piece_blob = b"".join(
        struct.pack(
            "<IIif",
            piece_pool_off[i],
            len(piece_list[i].encode("utf-8")),
            int(piece_to_model.get(piece_list[i], 3)),
            scores[i],
        )
        for i in range(n)
    )
    detok_blob = b"".join(struct.pack("<I", off) for off in detok_off)
    norm_str_bytes = normalized.encode("utf-8") if isinstance(normalized, str) else normalized

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    data = header + piece_blob + detok_blob + units + norm_str_bytes + mark_blob + bytes(pool_bytes)
    ASSET_FILE.write_bytes(data)
    print(f"written {ASSET_FILE} ({len(data)} bytes)")

    # ---- Kotlin 单测向量（精简子集；全量对拍已在上面完成） ----
    rng = random.Random(42)
    short_pool = [p for p in piece_list if len(p) <= 6]
    long_pool = [p for p in piece_list if len(p) > 6]
    short_sample = rng.sample(short_pool, min(4000, len(short_pool)))
    long_sample = rng.sample(long_pool, min(4000, len(long_pool)))

    vectors = []
    for text in list(dict.fromkeys(short_sample + long_sample + sentences)):
        pieces = sp.Encode(text, out_type=str)
        ids = [int(piece_to_model.get(p, 3)) for p in pieces]
        vectors.append({"text": text, "pieces": pieces, "ids": ids})
    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, separators=(",", ":"))
    print(f"written {VECTORS_FILE} ({len(vectors)} vectors)")

    # ---- 归一化向量（定向子集：有规则的字符 + 组合序列抽样 + 句子） ----
    norm_texts = []
    for cp in range(0x20, 0x10000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if normalize_mirror(c, darts) != DUMMY + c:
            norm_texts.append(c)
    pair_pool = [(b, m) for b in bases for m in combining]
    rng.shuffle(pair_pool)
    norm_texts += [b + m for b, m in pair_pool[:4000]]
    norm_texts += sentences
    norm_vectors = [{"text": t, "normalized": normalize_mirror(t, darts)} for t in norm_texts]
    with open(NORM_VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(norm_vectors, f, ensure_ascii=False, separators=(",", ":"))
    print(f"written {NORM_VECTORS_FILE} ({len(norm_vectors)} vectors)")


if __name__ == "__main__":
    main()
