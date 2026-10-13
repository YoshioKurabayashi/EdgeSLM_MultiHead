# Linear Head 初回検証結果（2026-09-17）

## 何を確認したか

公式Qwen3.5-2B（revision `15852e8c16360a2fea060d615a32b45270f8a8fc`）をBF16でRTX 5070 Ti上に読み、Vision Encoder・backbone・Embedding・既存LM Headを固定。
2048次元の最後の有効tokenのhidden stateから、Intent 6分類とGuardrail 3分類のLinear Headを並列に学習した。
追加trainable parameterは18,441、Head checkpointはseed別に保存した。

## データと選択条件

手書きの日本語合成データ344行をtrain172/validation86/test86に分割。元の発話、状態違い、入力/操作段階をgroup単位で同じsplitに置く。testにはIntentが付いた入力48行、Guardrailが付いた86行を含み、BLOCKの正解は26行。
出力後Guardrailのtestは6行だけ。画像はHead学習には含めていない。
AdamW、lr=1e-3、最大60epoch、seed 7/17/29。最良epochはvalidationのIntent/Guardrail Macro-F1平均で決定し、testで調整していない。

|seed|Intent Accuracy / Macro-F1（48行）|Guardrail Accuracy / Macro-F1（86行）|BLOCK→ALLOW（26行）|選択epoch|
|---|---:|---:|---:|---:|
|7|97.9% / 97.9%|93.0% / 93.0%|0|35|
|17|100.0% / 100.0%|94.2% / 94.2%|0|54|
|29|95.8% / 95.8%|90.7% / 90.4%|0|47|

BLOCK→ALLOWが0でも、同一の合成ルールから作った26行であり、安全性の証拠にはならない。Guardrailには他の誤分類もある。クラス別P/R/F1と混同行列、段階別精度は`runs/linear_v1/seed*.json`に保存。

## モデルの維持と性能

学習前後の全固定tensorのSHA-256が一致。Headsは更新され、固定部分に勾配は残らなかった。
テキスト2件＋合成画像1件のgreedy生成token列と次token logitsが学習前後で完全一致。別プロセスで公式モデルとHead3seedを読み直し、固定tensor、テスト指標、生成結果が一致。
これらは限定プローブであり、一般的な生成品質の包括評価ではない。

特徴抽出344行は22.6秒。ピークGPU割当約4.53GB。39token、batch1のWindows PyTorch fallbackによる単一入力10回計測で、モデル＋Head p50約506ms、Head単体p50約0.20ms。tokenization、画像前処理、Guardrailの再判定を含めない。端末速度として外挿しない。

## 統合デモ

- 「温度を24度にして」→ Intent=SET_TEMPERATURE、入力/操作Guardrail=ALLOW、CPUで24を抽出・検査し模擬実行。
- 走行中「トランクを開けて」→ BLOCK、模擬ツールの呼出し0。
- 「こんにちは」→ NO_TOOL、対話経路。生成回答は「こんにちは！お元気ですか？」。
- 赤い四角と青い丸の合成画像に同じ質問をすると、前者は赤/正方形、後者は青/丸を回答。
- 3つの回答例に対する出力GuardrailはALLOW。画像認識率や危険応答の検出率を推定する件数ではない。

## Qwen標準ツール呼び出し

公式chat templateの`<tool_call><function=...><parameter=...>`形式を、Headを使わず5例で試した。
温度24、音量3、引数なしトランク、東京駅ナビ、一般挨拶の5例すべてで期待したツール名/引数/非ツール応答と一致した。
検証件数が小さいため、2Bモデルの自由引数能力や複雑なtool useの信頼性は未評価。無引数ツールを解析する際の実装バグは、修正前に失敗するテストを追加し、修正後に通ることを確認した。

## 次の比較と不足データ

同じデータ・特徴でMLP Headを学習し、分類精度と計算量を比較する。その後、日本語BERTでテキストのIntent/Guardrail分類を比較する。
一般画像、画像からのツール選択、不正指示、複合依頼、自由引数、実車条件を別データで評価する。
追加Headを付けられることと、実環境での安全性/端末搭載性は別の合否とする。

## 生データ

`runs/linear_v1/metadata.json`、`parameters.json`、`seed*.json`、`summary.json`、`fresh_process_reload.json`、`demo.json`、`tool_call_probe.json`。
モデル重み/Head/特徴キャッシュ/実験出力はGit対象外。スクリプト・合成データ・ラベル定義はGit対象。
