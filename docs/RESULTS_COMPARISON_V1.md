# Linear / MLP / 日本語BERT比較（2026-09-17）

## 比較条件

3方式とも同じ日本語合成データの発話group分割を使用。train172/validation86/test86行。Intentはtest中の入力48行、Guardrailは全86行で採点。
LinearとMLPは同じ固定Qwen3.5-2Bの2048次元特徴を入力し、Headsだけ学習。MLPは各Headを2048→256→クラス数とする。
BERTは`cl-tohoku/bert-base-japanese-char-v3`をrevision `ccddee593a9e03a589ae1feedcd4d1eb06d6ba02`で使用。日本語Tokenizer、Encoderと2 Headsを一緒に学習。画像・文章生成能力は比較対象に含めない。
各方式3seed（7/17/29）。Validationの両タスクMacro-F1平均で最良epochを選び、testを調整に用いない。

|方式|学習parameter|Intent Accuracy / Macro-F1 平均|Guardrail Accuracy / Macro-F1 平均|BLOCK→ALLOW（各seed、正解BLOCK26行）|
|---|---:|---:|---:|---:|
|Qwen＋Linear|18,441|97.92% / 97.89%|92.64% / 92.52%|0 / 0 / 0|
|Qwen＋MLP|1,051,401|98.61% / 98.59%|93.80% / 93.68%|0 / 0 / 0|
|日本語BERT＋分類Heads|91,445,001|91.67% / 91.02%|96.12% / 96.22%|1 / 2 / 0|

BERTはGuardrail全体の平均が高いが、禁止を許可とした例がある。MLPはLinearよりGuardrail平均+1.16ポイントで、Head parameterは約57倍。合成・少数データに限った結果で、順位を実運用の安全性や一般性能へ外挿しない。
BERTは別Tokenizer・学習可能Encoderを用いる独立システムなので、Head形状だけを変えたA/B試験と同じ条件ではない。

## batch1の処理時間・メモリ

同じtest発話1件をWindowsのRTX 5070 Tiで計測。ウォームアップ5回、Qwenは10回、MLP/BERTは30回。GPU同期後のp50。Tokenizer時間はBERT側で別に計測。

|測定範囲|p50|
|---|---:|
|Qwen本体＋Linear Heads（入力39token）|約506ms|
|Linear Headsのみ|約0.20ms|
|MLP Headsのみ|約0.51ms|
|BERT本体＋Heads（入力52token）|約10.3ms|
|BERT Tokenizerのみ|約0.18ms|

QwenモデルはこのWindows環境で線形Attentionの高速ライブラリがなく、PyTorch実装へfallbackした。最適化後やSoC実機の比較値ではない。
Qwen＋MLPの本体込みレイテンシは再計測しておらず、同じ固定本体にMLP Headを接続する構成。画像前処理、入力/操作/出力ガードレールの追加pass、ツール引数/回答生成は表に含めない。
Qwen＋Linear実験のピークGPU割当は約4.53GB。BERT学習中のピークは約1.90GB、BERT単発推論は約0.38GB。MLP HeadのFP32 parameterのみで約4.21MB。

Qwenが既に対話・画像QAに常駐するなら、Headによる追加計算は小さい。一方、分類だけを目的に都度モデルを使うなら、このホスト条件ではBERTの方が速い。端末上で同じ関係になるとは限らない。

## 成果物と限界

`runs/linear_v1`、`runs/mlp_v1`、`runs/bert_v1`、`runs/comparison_benchmark.json`。各seedのクラス別P/R/F1、混同行列、危険側誤り、選択epoch、checkpointを保存。
`runs/comparison_reload.json`ではMLP/BERTの3seedすべてを別プロセスで復元し、保存済みtest指標との一致を確認した。
初回データは手書きの定型表現と状態ラベルを含む。一般画像を用いるIntent/Guardrail分類、実発話、不正指示、未知の車両状態、複合ツールは未評価。
次の合否判断には、別作成者・実運用に近い独立データと、安全の誤りコストを定めた基準が必要。
