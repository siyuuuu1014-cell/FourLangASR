Model files are NOT included in this source zip.

Run from PowerShell at the project root:

  powershell -ExecutionPolicy Bypass -File .\tools\setup_all.ps1

The script copies your already-tested PC models from:
  D:\dev\projects\fourlang_translation\asr_exp1

Expected final asset tree:

models/
  zh/
    model.int8.onnx
    tokens.txt
    bbpe.model                 (optional, copied for completeness)

  en/
    encoder.int8.onnx
    decoder.onnx
    joiner.int8.onnx
    tokens.txt

  uz/
    encoder.onnx
    decoder.onnx
    joiner.onnx
    tokens.txt

  ru/
    vosk-model-small-ru-0.22/
      am/
      conf/
      graph/
      ivector/
      ...
