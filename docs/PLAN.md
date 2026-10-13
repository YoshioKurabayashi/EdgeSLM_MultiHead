# 開発計画 v0.2

## 目的

ツールコール、一般対話、画像認識、ガードレールを一つのQwen3.5-2Bで扱い、追加Linear Headの成立性と効果を検証する。
初期はVision Encoder、backbone、Embedding、LM Headを全固定。LoRAや本体のfine tuningは行わない。
MLPとBERT比較を後続に含む。既存データはなく、日本語合成データを作成する。

## 機能と実装

|機能|経路|評価|
|---|---|---|
|ツールコール|Intent Headで呼出し要否/ツール選択、定型引数はCPU抽出、自由引数は既存LM Headでツール呼出し生成、schema検査と実行器|選択、引数、不要呼出し、結果への応答|
|一般対話|既存LM Head|妥当性、不要tool call、学習前後の回帰|
|画像認識|Vision Encoder＋融合backbone＋LM Head|画像QA/内容説明、画像依存性、回帰|
|ガードレール|Guard Headと確定的な検査|見逃し、過剰拒否、要確認への誘導|

4機能は同時に使われるため「4分類Head」にはしない。画像を理解してツールを呼ぶ処理も扱う。
Tool Headのラベル数はツール一覧＋NO_TOOL等から定める。20分類を固定要件にはしない。
ガードレールは入力・実行前・応答後を区別する。同じ3ラベル/Headを共用するかは対象ポリシーを定めて判断し、段階情報を入力に含める。

## 確定した機能範囲

1. ツール: 模擬ツールの選択・引数生成・実行・結果への応答まで。
2. 画像: 一般画像の内容説明・質問応答から開始。
3. ガードレール: 操作の実行可否と、入力/応答の危険内容・不正指示の両方。

推奨アーキテクチャはdocs/ARCHITECTURE.md。独立Router Headは設けず、Intent HeadとGuardrail Headの出力をCPUのDispatcherで合成する。具体的ツール一覧とデータラベルの初版は次の仕様化で作成する。

## 構成案

テキスト＋任意の画像＋必要な状態
→ 公式Processor/chat template
→ Frozen Vision Encoder＋Frozen multimodal backbone
→ 最後の有効tokenの融合hidden state
→ Linear Tool Head／Guard Head
→ 必要な場合だけ既存LM Headの応答または自由引数生成
→ 引数schema・ポリシー確認
→ 許可された模擬ツール
→ 実行結果を使う応答

画像の有無だけで分岐せず、内容を参照する。画像内の文やtool結果はデータとして扱い、システム規則に昇格させない。
固定backboneでもHeadによる経路選択は動作を変える。「生の生成の不変性」と「ツール/拒否を含むシステム動作」を別に評価する。

## フェーズ

|段階|成果物|完了条件|
|---|---|---|
|0|独立Git/venv、GPU確認、要件確定|CUDA計算成功、機能範囲確定（完了）|
|1|ツール・ポリシー・データ仕様|名前/引数/実行条件、ラベル、画像評価対象が明確|
|2|元モデルのbaseline|固定revisionで対話・画像QA・tool callを実行、失敗も保存|
|3|Linear Head学習|Headのみ更新、欠損ラベルmask、固定全tensor一致|
|4|統合評価|各機能と複合ケース、学習前後比較、別プロセス復元|
|5|MLP/BERT比較|合成pilotの同一splitで3seed比較済み。外部データでの再評価は未実施|
|6|実データ拡張・移植調査|外部妥当性、ONNX/QNN/量子化を段階別評価|

## データ

発話、履歴、車両等の状態、画像参照、期待tool call、ガードレールラベル、回答要件を別フィールドに保存。
正解ラベル・正解tool call・正解説明文をモデル入力へ混ぜない。
原発話・元画像・シーン・会話をgroup化し、系列単位で分割。リサイズ/切出し/色変更を別splitへ置かない。
testで閾値やプロンプトを調整しない。データ量は仕様確定後に決定し、単純なテンプレート増殖で件数だけ増やさない。

## テスト

- 小型モデルでHead形状、学習対象、個別loss、欠損ラベル、左右Padding、空入力。
- 実Qwenでテキスト/画像のhidden state、Processor、可変画像token数とPadding。
- 同じ質問に別画像を与え、画像を無視しないことを確認。
- 元モデル、Head追加直後、学習後、別プロセス復元後の重み・logits・greedy token比較。
- 画像経路も回帰検証に含める。限定サンプル一致を一般的な品質の証明と呼ばない。
- revision、Processor、template、ラベル、dataset hash、dtype、seedを保存。
- 固定重み変更・token選択ミス・未許可tool callを意図的に与え、検査が失敗することを確認。

## 指標

Tool: 呼出し要否P/R/F1、選択Accuracy/Macro-F1、JSON構文/schema成功率、引数意味一致、タスク成功率、不要呼出し率。
Dialogue: 指示遵守、回答妥当性、不要tool call、生成回帰。
Vision: QA正解率/属性一致、人手評価、画像差替え対照、生成回帰。自然文説明を文字列完全一致だけで採点しない。
Guardrails: クラス別P/R/F1、危険側見逃し、過剰拒否、要確認。入力/操作/出力・攻撃種類別に分解。
性能: head-only、backbone込み、tokenization/画像前処理込みを区別。batch1、入力長/解像度別p50/p95、VRAM/RAM、学習parameter数。
3seed（7/17/29）で平均・ばらつき・件数を報告。合成データの95%到達を実運用合格とはしない。

## 比較

A: Frozen Qwen＋Linear Tool/Guard Heads。
B: 同じQwen固定特徴＋MLP Heads。
C: 日本語BERT＋Tool/Guard Heads（Encoderも学習）。
BERTはまずテキストの選択/ガードレールに限定。画像をQwenでテキスト化してBERTへ渡す案は、画像処理コストと誤りを含む別システム比較とする。
一般対話・画像QA・引数生成をBERT分類Headと直接比較しない。

## 実装予定

model.py、train.py、evaluate.py、test_generation.pyを核とし、multimodal.py、tools.py、guardrails.py、data.py、benchmark.py、configs/、tests/を追加。
視覚部と生成経路は公式Processorとモデルクラスを使う。
