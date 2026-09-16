Run tools/download_sherpa_aar.ps1 to download sherpa-onnx-1.13.7.aar here
（脚本末尾会执行 dual-ORT 设置，见下）。

Dual-ORT（libonnxruntime 双库并存，实测必需）
----------------------------------------------
- sherpa 的 libsherpa-onnx-jni.so 只能与它自带的
  libonnxruntime.so（隐藏符号构建）配套加载；
- onnxruntime-java 桥（libonnxruntime4j_jni.so）只能与官方
  onnxruntime-android 1.27.0 的 libonnxruntime.so 配套加载；
- 两者交叉组合在设备上实测 dlopen 失败（cannot locate symbol
  OrtGetApiBase），因此双库并存：

  1. sherpa AAR 保持原样（AAR 内 jni/arm64-v8a/libonnxruntime.so）；
  2. 官方库改名为 app/src/main/jniLibs/arm64-v8a/libort4jrt.so
     （SONAME 同长度补丁：libonnxruntime.so -> libort4jrt.so；
      gitignored，由 tools/setup_dual_ort.py 重建）；
  3. app/src/main/jniLibs/arm64-v8a/libonnxruntime4j_jni.so
     的 DT_NEEDED 同长度补丁为 libort4jrt.so（已提交）。
