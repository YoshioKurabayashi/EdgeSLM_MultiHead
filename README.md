# EdgeSLM_MultiHead

ツールコール・一般対話・画像認識・ガードレールを扱う、Qwen3.5-2Bマルチヘッド検証の独立プロジェクト。

## 確定した方針

- Qwen/Qwen3.5-2BのVision Encoderを保持する。
- 初期は本体・視覚部・Embedding・既存LM Headを固定し、追加Linear Headだけを学習。
- 後続でMLP Head、日本語BERT系の独立分類器と比較する。
- 既存データなし。日本語の合成検証データから始める。
- 画像を見てツールを呼ぶ場合もあり、4機能を排他的な4分類にはしない。

一般対話と画像質問応答には既存の生成経路を使う。追加Headはツール選択やガードレール判定に使う。ツールの定型引数はCPUで抽出し、自由な引数が必要な場合はLMで候補を生成して検証する。Linear分類Headだけで自由な引数や説明文は生成しない。

## 状態

Gitと専用venvを作成。PyTorch 2.11.0+cu128、Transformers 5.14.1を導入し、RTX 5070 TiでCUDA tensor計算を確認済み。
公式Qwen3.5-2Bの取得、Linear/MLP Head学習、日本語BERT比較、模擬ツール実行・一般対話・画像QA・ガードレールの小規模検証を実施。結果は合成データ上の試験であり、実運用の精度や安全性は未検証。元案の車載Intent20分類コード・データは採用していない。

[推奨アーキテクチャ](docs/ARCHITECTURE.md)、[開発計画](docs/PLAN.md)、[データ方針](docs/DATA_POLICY.md)、[実行状況](docs/STATUS.md)を参照。

最初の実測は[Linear Headの検証結果](docs/RESULTS_LINEAR_V1.md)に記録。テキスト/画像の特徴抽出、Head学習、固定重みと生成の不変性、模擬ツール実行を確認した。
[MLP/BERT比較](docs/RESULTS_COMPARISON_V1.md)には、同一合成分割での3seed指標と処理時間を記録。

```powershell
.\.venv\Scripts\python.exe -B check_environment.py
```

`requirements-lock.txt`は導入済み依存一覧。公式画像ProcessorにPillowとTorchvisionが必要。

## Linear Head検証

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp tests/.pytest_tmp_run
.\.venv\Scripts\python.exe -B data.py --out data/synthetic_v1
.\.venv\Scripts\python.exe -B -u train.py --model-path models/qwen3_5_2b --data data/synthetic_v1 --out runs/linear_v1
.\.venv\Scripts\python.exe -B -u verify_reload.py --run runs/linear_v1
.\.venv\Scripts\python.exe -B -u demo.py --run runs/linear_v1 --out runs/linear_v1/demo.json
.\.venv\Scripts\python.exe -B -u probe_tool_call.py --out runs/linear_v1/tool_call_probe.json
.\.venv\Scripts\python.exe -B -u compare_mlp.py --linear-run runs/linear_v1 --out runs/mlp_v1
.\.venv\Scripts\python.exe -B -u compare_bert.py --out runs/bert_v1
.\.venv\Scripts\python.exe -B -u benchmark_comparison.py --out runs/comparison_benchmark.json
.\.venv\Scripts\python.exe -B -u verify_comparison.py
```

データを作成済みの状態で`data.py`を再実行すると上書きを避けるため停止する。実験出力も既存ディレクトリを上書きしない。`runs/linear_v1/summary.json`と`fresh_process_reload.json`が主な結果。合成データの指標を実環境での精度と解釈しない。

公式モデルは`Qwen/Qwen3.5-2B`のrevision `15852e8c16360a2fea060d615a32b45270f8a8fc`を`models/qwen3_5_2b`へ取得済み。重みはGit対象外。再取得時は同じrevisionを指定する。モデルがない環境では先にHugging Face Hubの`snapshot_download`を実行する。
日本語BERT比較は`cl-tohoku/bert-base-japanese-char-v3`のrevision `ccddee593a9e03a589ae1feedcd4d1eb06d6ba02`を使用。こちらはHugging Faceのローカルキャッシュを利用し、未キャッシュ環境では同じrevisionを取得してから実行する。

## 公式資料

- https://huggingface.co/Qwen/Qwen3.5-2B
- https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json
- https://huggingface.co/docs/transformers/model_doc/qwen3_5
- https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/chat_template.jinja
