#!/usr/bin/env python3
"""对比官方 libonnxruntime.so 与 patch 版 libort4jrt.so：
1) 找出 "libonnxruntime.so" 字符串在文件里出现的所有偏移；
2) 用 pyelftools 判断每个偏移落在哪个 section（.dynstr/.rodata/...）；
3) 逐字节 diff 官方 vs 修改版，列出所有被改动的区域，判断是否只改了 SONAME/NEEDED。
"""
import sys
from elftools.elf.elffile import ELFFile

PAT = b"libonnxruntime.so"
NEW = b"libort4jrt.so"

official = open(sys.argv[1], "rb").read()
patched = open(sys.argv[2], "rb").read()

print(f"official size={len(official)} patched size={len(patched)} same={len(official)==len(patched)}")

# 1) occurrences in official
offs = []
i = official.find(PAT)
while i != -1:
    offs.append(i)
    i = official.find(PAT, i + 1)
print(f"\n'libonnxruntime.so' 在官方文件出现 {len(offs)} 次，偏移: " + ", ".join(hex(o) for o in offs))

# 2) section of each occurrence
import io
elf = ELFFile(io.BytesIO(official))
def section_of(off):
    for s in elf.iter_sections():
        if s['sh_addr'] == 0 and s['sh_offset'] == 0 and s.name in ('', '.shstrtab'):
            pass
        if s['sh_offset'] <= off < s['sh_offset'] + max(s['sh_size'], 1):
            return s.name
    return "<none/file-gap>"
for o in offs:
    ctx = official[o-8:o+len(PAT)+8]
    print(f"  off {hex(o)} -> section {section_of(o)}  ctx={ctx!r}")

# 3) byte diff regions
print("\n=== 逐字节差异区域 (official vs patched) ===")
diffs = [k for k in range(min(len(official), len(patched))) if official[k] != patched[k]]
if not diffs:
    print("  无差异")
else:
    # group contiguous
    regions = []
    start = diffs[0]; prev = diffs[0]
    for d in diffs[1:]:
        if d != prev + 1:
            regions.append((start, prev)); start = d
        prev = d
    regions.append((start, prev))
    print(f"  共 {len(diffs)} 字节变化，聚成 {len(regions)} 段:")
    for a, b in regions:
        print(f"   [{hex(a)}..{hex(b)}] len={b-a+1} section={section_of(a)} official={official[a:b+1]!r} patched={patched[a:b+1]!r}")
