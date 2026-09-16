"""Dual-ORT setup for the app.

Problem (empirically verified on device):
  - sherpa-onnx's libsherpa-onnx-jni.so imports OrtGetApiBase and only loads
    against sherpa's own hidden-visibility libonnxruntime.so build;
  - onnxruntime-java's libonnxruntime4j_jni.so imports OrtGetApiBase and only
    loads against the official onnxruntime-android 1.27.0 build.
  Neither importer resolves against the other's lib (both export the same
  3-symbol C-API subset, yet cross-combination dlopen fails on-device).

Fix: ship BOTH builds under different SONAMEs:
  - sherpa AAR keeps its original libonnxruntime.so (name unchanged);
  - official lib is renamed libort4jrt.so (SONAME patched, same-length-safe);
  - the Java bridge's DT_NEEDED is patched from libonnxruntime.so to
    libort4jrt.so (in-place, NUL-padded).

Usage:
  python tools/setup_dual_ort.py <sherpa_aar> <official_ort_aar> <jni_bridge_so> <out_aar> <out_dir>
"""
import os
import sys
import zipfile

NEEDED_OLD = b"libonnxruntime.so"
NEEDED_NEW = b"libort4jrt.so" + b"\x00" * (len(NEEDED_OLD) - len(b"libort4jrt.so"))


def patch_all(buf: bytes):
    count = buf.count(NEEDED_OLD)
    if count == 0:
        return buf, 0  # 已打过补丁（幂等）
    return buf.replace(NEEDED_OLD, NEEDED_NEW), count


def main():
    sherpa_aar = sys.argv[1]
    official_aar = sys.argv[2]
    bridge_so = sys.argv[3]
    out_aar = sys.argv[4]
    out_dir = sys.argv[5]

    # 1) extract official arm64 libonnxruntime.so from the official AAR
    with zipfile.ZipFile(official_aar, "r") as z:
        official_so = z.read("jni/arm64-v8a/libonnxruntime.so")
    assert NEEDED_OLD in official_so, "official lib lacks SONAME string"

    # 2) build libort4jrt.so (patched SONAME) + patched bridge
    ort_renamed, n1 = patch_all(official_so)
    bridge_data = open(bridge_so, "rb").read()
    bridge_patched, n2 = patch_all(bridge_data)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "libort4jrt.so"), "wb") as f:
        f.write(ort_renamed)
    with open(os.path.join(out_dir, "libonnxruntime4j_jni.so"), "wb") as f:
        f.write(bridge_patched)
    print(f"patched: libort4jrt.so ({n1} SONAME hits), bridge ({n2} NEEDED hits)")

    # 3) sherpa AAR: restore ORIGINAL sherpa libonnxruntime.so
    #    (arg 6 = path to sherpa's original lib, e.g. extracted from the
    #    stock AAR; if omitted, repack the AAR as-is)
    sherpa_original = sys.argv[6] if len(sys.argv) > 6 else None
    with zipfile.ZipFile(sherpa_aar, "r") as z:
        names = z.namelist()
        entries = {n: z.read(n) for n in names}
    target = "jni/arm64-v8a/libonnxruntime.so"
    assert target in entries, "sherpa AAR lacks arm64 libonnxruntime.so"
    if sherpa_original:
        with open(sherpa_original, "rb") as f:
            entries[target] = f.read()
        print(f"restored sherpa original lib ({len(entries[target])} bytes)")
    with zipfile.ZipFile(out_aar, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, entries[n])
    print("AAR repacked:", out_aar)


if __name__ == "__main__":
    main()
