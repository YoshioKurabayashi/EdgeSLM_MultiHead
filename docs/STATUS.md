# 実行状況 2026-09-17

## 完了

- 独立したGitリポジトリ、専用Python環境、GPU動作確認。
- 公式Qwen/Qwen3.5-2Bをrevision `15852e8c16360a2fea060d615a32b45270f8a8fc`で取得。
- 既存Vision Encoderを使い、テキスト・画像の両方でhidden state→Intent/Guardrail Headを実行。
- Linear Head 18,441 parameterを3seedで学習。Qwen/LM Head/Vision Encoderは全固定。
- 合成テスト86行でIntent/Guardrailを評価。指標、混同行列、危険側誤りを保存。
- 全固定tensor比較、学習前後のtext/image生成logitsとgreedy token比較、別プロセス再読込。
- 模擬温度設定を実行し、走行中トランク開を停止。一般対話と赤い四角/青い丸の画像QAを実行。
- Qwen公式tool-call形式を5件プローブし、5件でツール名/引数/非ツール応答が期待値と一致。
- ユニットテスト14件通過。無引数tool解析の不具合を、修正前失敗/修正後成功で確認。
- 同一合成分割でMLP Head（固定Qwen特徴）と日本語BERT（本体も学習）を3seed比較。処理時間とメモリを測定。
- MLP/BERTの保存済みcheckpointを別プロセスで再読込し、3seedすべてのtest指標一致を確認。

詳細数値と条件は`docs/RESULTS_LINEAR_V1.md`と`docs/RESULTS_COMPARISON_V1.md`、生の出力は`runs/`を参照。

## 未検証・未実装

- 実発話/実画像/実際の危険依頼での精度、入力/出力ガードレールの汎化。
- 画像内容に基づくツール選択やGuardrail分類の学習・評価。
- 複合ツール、自由引数の大規模評価、誤認識/不明/対話履歴の頑健性。
- MLP/BERTの実発話での比較、画像を含む公平な分類システム比較。
- 端末/ONNX/QNN/量子化。性能はWindowsホストのPyTorch fallbackのみ。

合成データの高い分類精度や5件のtool-call一致は、実運用の成功・安全性を証明しない。
