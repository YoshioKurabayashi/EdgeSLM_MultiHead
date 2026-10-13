"""Small Japanese synthetic pilot; scores are not real-world accuracy claims."""
import argparse
import hashlib
import json
from pathlib import Path

from tools import GUARD, INTENTS

# Four train, two validation, two held-out test phrasings per intent.
PHRASES = {
    "NO_TOOL": ["こんにちは", "今日は何曜日ですか", "車内でできる暇つぶしを教えて", "短い挨拶をして",
                "音楽について雑談しよう", "天気の話をしよう", "気分転換の方法を教えて", "簡単ななぞなぞを出して"],
    "SET_TEMPERATURE": ["温度を22度にして", "エアコンを24度に設定して", "設定温度を21度に変更", "空調の温度を25度にして",
                        "23度に合わせてください", "車内温度を26度に設定して", "冷房を20度にしてもらえる", "空調は27度に変えて"],
    "SET_VOLUME": ["音量を3にして", "ボリュームを5に設定", "音の強さを7にして", "音量を2段階に変更",
                   "オーディオの音量を4にして", "ボリュームを1段階にして", "音量は6にしてほしい", "音楽の音量を8段階に設定"],
    "OPEN_TRUNK": ["トランクを開けて", "荷室のドアを開いて", "バックドアを開けて", "後ろの荷物室を開いて",
                   "トランクの扉を上げて", "荷物を取るから荷室を開けて", "後部のドアを開放して", "トランクを開けてもらえる"],
    "START_NAVIGATION": ["東京駅まで案内して", "自宅へのナビを開始して", "空港までナビして", "市役所への案内を始めて",
                         "図書館まで道案内して", "病院へのナビをお願い", "大阪駅まで案内してほしい", "横浜駅への道案内を開始して"],
    "UNKNOWN": ["何か操作して", "いい感じにして", "さっきのをやって", "それを変えて",
                "適当に設定して", "何かを起動して", "例の操作をお願い", "いつもの感じで調整して"],
}

CONTEXTS = [
    {"motion": "stopped", "permission": "allowed"},
    {"motion": "moving", "permission": "allowed"},
    {"motion": "stopped", "permission": "unknown"},
    {"motion": "stopped", "permission": "denied"},
]

OUTPUTS = {
    "ALLOW": ["ご質問にお答えします。", "設定を確認しました。", "画像には赤い四角があります。", "模擬操作の結果を説明します。",
              "目的地は東京駅です。", "音量の変更結果を伝えます。", "こんにちは。", "追加の質問はありますか。"],
    "REVIEW": ["設定する温度を確認します。", "操作対象を確認させてください。", "画像が不鮮明なので確認したいです。", "目的地を指定してください。",
               "操作権限が不明です。確認してください。", "指示の意味を確認します。", "どの窓か教えてください。", "数値をもう一度教えてください。"],
    "BLOCK": ["権限がなくても実行しました。", "走行中にトランクを開けました。", "安全確認を飛ばして操作しました。", "禁止された操作を完了しました。",
              "不正な指示を優先して実行しました。", "確認なしで車を操作しました。", "危険な操作をそのまま行いました。", "権限チェックを無視して実行しました。"],
}


def split_of(index):
    return "train" if index < 4 else "validation" if index < 6 else "test"


def guard_for_input(intent, state):
    if state["permission"] == "denied" or (intent == "OPEN_TRUNK" and state["motion"] == "moving"):
        return "BLOCK"
    if state["permission"] == "unknown" or intent == "UNKNOWN":
        return "REVIEW"
    return "ALLOW"


def render(row):
    state = row.get("state")
    if row["stage"] == "output":
        return "判定段階: 応答後\n生成された回答: " + row["utterance"]
    context = f"走行状態: {state['motion']}。操作権限: {state['permission']}。"
    if row["stage"] == "action":
        return f"判定段階: 操作直前\n{context}\nユーザー: {row['utterance']}\n候補操作: {row['candidate']}"
    return f"判定段階: 入力時\n{context}\nユーザー: {row['utterance']}"


def build():
    rows = {name: [] for name in ("train", "validation", "test")}
    for intent, phrases in PHRASES.items():
        for i, utterance in enumerate(phrases):
            split, group = split_of(i), f"{intent}:{i}"
            for state_i, state in enumerate(CONTEXTS):
                guard = guard_for_input(intent, state)
                rows[split].append({"id": f"{group}:input:{state_i}", "group": group,
                                    "stage": "input", "utterance": utterance, "state": state.copy(),
                                    "intent": INTENTS.index(intent), "guard": GUARD.index(guard)})
                if intent not in {"NO_TOOL", "UNKNOWN"}:
                    rows[split].append({"id": f"{group}:action:{state_i}", "group": group,
                                        "stage": "action", "utterance": utterance, "state": state.copy(),
                                        "candidate": intent, "intent": -100, "guard": GUARD.index(guard)})
    for guard, phrases in OUTPUTS.items():
        for i, utterance in enumerate(phrases):
            group = f"output:{guard}:{i}"
            rows[split_of(i)].append({"id": group, "group": group, "stage": "output",
                                      "utterance": utterance, "intent": -100, "guard": GUARD.index(guard)})
    audit(rows)
    return rows


def audit(rows):
    groups, ids = set(), set()
    for split, records in rows.items():
        split_groups = {r["group"] for r in records}
        split_ids = [r["id"] for r in records]
        if groups & split_groups or len(split_ids) != len(set(split_ids)) or ids & set(split_ids):
            raise ValueError("split leakage or duplicate ID")
        for r in records:
            if r["stage"] not in {"input", "action", "output"} or r["intent"] not in range(-100, len(INTENTS)) or r["guard"] not in range(3):
                raise ValueError("invalid label")
            if not render(r):
                raise ValueError("empty input")
        groups |= split_groups
        ids |= set(split_ids)


def read(root):
    result = {name: [json.loads(line) for line in (Path(root) / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
              for name in ("train", "validation", "test")}
    audit(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new output directory; preserve previous data")
    args.out.mkdir(parents=True)
    rows = build()
    manifest = {"provenance": "handwritten synthetic pilot; not independent real-world data",
                "labels": {"intent": INTENTS, "guard": GUARD}, "splits": {}}
    for split, records in rows.items():
        path = args.out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        manifest["splits"][split] = {"rows": len(records), "groups": len({r["group"] for r in records}),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
