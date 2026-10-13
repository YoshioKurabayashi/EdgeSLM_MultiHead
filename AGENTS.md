# EdgeSLM_MultiHead

- 独立プロジェクト。隣接するEdgeSLM系・stock_analysis_agentは変更しない。
- 必須機能はツールコール、一般対話、画像認識、ガードレール。
- 初期はQwen/Qwen3.5-2BのVision Encoder、backbone、Embedding、LM Headを全固定し、追加Linear Headだけを学習。
- MLP/BERTは後続比較。BERT単体はテキスト分類の比較対象で、画像・生成の代替と呼ばない。
- 4機能を排他的な4クラスにしない。一般対話・引数生成・画像QAは既存生成経路を使う。
- 元案のIntent20分類は未採用。ユーザーが挙げた機能要件を優先する。
- 操作の実行可否と対話のコンテンツ安全性を区別する。
- 合成データの成績を実用精度と呼ばない。発話・画像・シナリオ系列でsplitを分ける。
- 固定全重み、Head更新、生成token/logits、画像経路、別プロセス再読込を検証する。
- 実ツール接続はユーザーが接続先と操作範囲を指定するまで行わない。
- モデルrevision、依存版、データhash、seed、入力形式を記録する。
- 実施済みと未検証の境界をdocs/STATUS.mdへ記録する。
