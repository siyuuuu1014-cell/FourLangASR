#!/usr/bin/env python3
"""把 logcat 里的 REPORT[name] part=N 分块重组成 json 文件。

Android 16 上 adb 读不到 app 外部私有目录，Harness.writeReport 会把整段 JSON
按 3000 字符分块打到 OcrPhase1 tag。这个脚本从抓下来的 logcat 文本还原成文件。
"""
import re
import sys
import json
import io

def main():
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    # 每行形如: ... I OcrPhase1: REPORT[a_db_profile.json] part=0 <chunk>
    pat = re.compile(r"REPORT\[([^\]]+)\] part=(\d+) (.*)$")
    endpat = re.compile(r"REPORT\[([^\]]+)\] part=END")
    parts = {}
    for line in io.open(src, encoding="utf-8", errors="replace"):
        m = endpat.search(line.rstrip("\n"))
        if m:
            parts.setdefault(m.group(1), {})
            parts[m.group(1)]["__end__"] = True
            continue
        m = pat.search(line.rstrip("\n"))
        if not m:
            continue
        name, idx, chunk = m.group(1), int(m.group(2)), m.group(3)
        parts.setdefault(name, {})
        parts[name][idx] = chunk
    import os
    os.makedirs(outdir, exist_ok=True)
    for name, block in parts.items():
        if "__end__" not in block:
            print(f"跳过 {name}: 未见 END 标记（可能被抓取截断）")
            continue
        idxs = sorted(k for k in block if isinstance(k, int))
        # logcat 会把 json 里的换行拆成多条；chunk 已含前导空格，直接拼接再补回换行不可靠，
        # 改为把所有 part 顺序连接后交给 json 解析（writeReport 用 toString(1)，是 pretty，含换行）。
        text = "".join(block[i] for i in idxs)
        path = os.path.join(outdir, name)
        try:
            obj = json.loads(text)
            io.open(path, "w", encoding="utf-8").write(json.dumps(obj, ensure_ascii=False, indent=1))
            print(f"OK {path}  ({len(text)} chars)")
        except Exception as e:
            # pretty json 的换行在 logcat 丢失，尝试把每行当独立片段：退化为压缩 json
            io.open(path, "w", encoding="utf-8").write(text)
            print(f"WARN {path}: json 解析失败({e})，已原样落盘供人工检查")

if __name__ == "__main__":
    main()
