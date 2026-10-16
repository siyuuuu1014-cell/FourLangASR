#!/usr/bin/env python3
"""阶段2B-0 任务A：原生库 ELF 审计。
看 DT_NEEDED / SONAME / build-id / 是否静态内嵌 libc++(定义 _ZNSt3__1 或 __cxa_throw) /
是否引用 libc++_shared(动态)。用来判断 OpenCV 与 ORT 两套原生栈是否各带一份 C++ 运行时。
"""
import sys
from elftools.elf.elffile import ELFFile
from elftools.elf.dynamic import DynamicSection

MAIN = "app/build/intermediates/merged_native_libs/debug/out/lib/arm64-v8a"
TEST = "app/build/intermediates/merged_native_libs/debugAndroidTest/out/lib/arm64-v8a"

LIBS = [
    ("ort4j_jni(OCR)", f"{MAIN}/libonnxruntime4j_jni.so"),
    ("libort4jrt(OCR 1.27)", f"{MAIN}/libort4jrt.so"),
    ("libonnxruntime(sherpa)", f"{MAIN}/libonnxruntime.so"),
    ("libsherpa-onnx-jni", f"{MAIN}/libsherpa-onnx-jni.so"),
    ("libvosk", f"{MAIN}/libvosk.so"),
    ("libtensorflowlite_jni", f"{MAIN}/libtensorflowlite_jni.so"),
    ("libopencv_java4(OCR test)", f"{TEST}/libopencv_java4.so"),
    ("libc++_shared(test)", f"{TEST}/libc++_shared.so"),
]

def libcxx_static_symbols(elf):
    """统计本文件里"定义(非 UNDEF)"的 libc++/C++ ABI 符号数量，>0 说明静态内嵌了一份 libc++。"""
    st = elf.get_section_by_name(".dynsym")
    if not st:
        return 0, 0
    defined_cxx = 0
    undef_cxx = 0
    for sym in st.iter_symbols():
        n = sym.name
        if not (n.startswith("_ZNSt3__1") or n.startswith("_ZNKSt3__1")
                or n.startswith("__cxa_") or n.startswith("_ZSt") or n.startswith("__gxx")):
            continue
        # SHN_UNDEF == 0 表示未定义(引用外部)；否则本文件定义。
        if sym["st_shndx"] == "SHN_UNDEF":
            undef_cxx += 1
        else:
            defined_cxx += 1
    return defined_cxx, undef_cxx

def audit(name, path):
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        print(f"!! {name}: 文件不存在 {path}")
        return
    with f:
        elf = ELFFile(f)
        dyn = elf.get_section_by_name(".dynamic")
        needed, soname, buildid = [], None, None
        for tag in dyn.iter_tags():
            if tag.entry.d_tag == "DT_NEEDED":
                needed.append(tag.needed)
            elif tag.entry.d_tag == "DT_SONAME":
                soname = tag.soname
        gi = elf.get_section_by_name(".note.gnu.build-id")
        if gi:
            # pyelftools 版本差异：直接取 section 原始数据，build-id 在 desc 区。
            try:
                data = gi.data()
                # note 头: namesz(4) descsz(4) type(4) name(对齐) desc
                namesz = int.from_bytes(data[0:4], "little")
                descsz = int.from_bytes(data[4:8], "little")
                off = 12 + ((namesz + 3) // 4 * 4)
                buildid = data[off:off + descsz].hex()
            except Exception as e:
                buildid = f"<读取失败 {e}>"
        d_cxx, u_cxx = libcxx_static_symbols(elf)
        print(f"=== {name} ===")
        print(f"  SONAME : {soname}")
        print(f"  build-id: {buildid}")
        print(f"  DT_NEEDED: {needed}")
        print(f"  libc++ ABI 符号: 定义(静态内嵌)={d_cxx}  引用(外部)={u_cxx}"
              + ("   <== 自带一份 libc++" if d_cxx > 100 else ""))
        print(f"  依赖 libc++_shared.so? {'是' if any('c++_shared' in x for x in needed) else '否'}")

for nm, p in LIBS:
    audit(nm, p)
