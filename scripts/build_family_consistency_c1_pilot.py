from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the C1 family pilot dataset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-probe", type=Path)
    return parser.parse_args()


def record(
    pair_id: str,
    category: str,
    left_text: str,
    left_prefix: str,
    right_text: str,
    right_prefix: str,
) -> dict[str, Any]:
    if not left_text.startswith(left_prefix) or not right_text.startswith(right_prefix):
        raise ValueError(f"landmark prefix mismatch: {pair_id}")
    return {
        "id": pair_id,
        "category": category,
        "left": {
            "text": left_text,
            "landmark_byte": len(left_prefix.encode("utf-8")),
        },
        "right": {
            "text": right_text,
            "landmark_byte": len(right_prefix.encode("utf-8")),
        },
    }


def build_splits() -> dict[str, list[dict[str, Any]]]:
    train_specs = [
        ("tr-inf-01", "inflection", "少年が庭を走っている。", "少年が庭を走", "少年が庭を走った。", "少年が庭を走"),
        ("tr-inf-02", "inflection", "鳥が空を飛んでいる。", "鳥が空を飛", "鳥が空を飛んだ。", "鳥が空を飛"),
        ("tr-inf-03", "inflection", "氷がゆっくり溶けている。", "氷がゆっくり溶け", "氷がすぐに溶けた。", "氷がすぐに溶け"),
        ("tr-inf-04", "inflection", "鐘が大きく鳴っている。", "鐘が大きく鳴", "鐘が一度鳴った。", "鐘が一度鳴"),
        ("tr-aux-01", "auxiliary", "資料を読み終えてしまった。", "資料を読み終えてしま", "資料を読み終えてしまう。", "資料を読み終えてしま"),
        ("tr-aux-02", "auxiliary", "窓を閉めておいた。", "窓を閉めてお", "窓を閉めておく。", "窓を閉めてお"),
        ("tr-aux-03", "auxiliary", "結果を確認してみた。", "結果を確認してみ", "結果を確認してみる。", "結果を確認してみ"),
        ("tr-aux-04", "auxiliary", "席を予約してある。", "席を予約してあ", "席を予約してあった。", "席を予約してあ"),
        ("tr-part-01", "particle", "仙台から列車に乗る。", "仙台", "仙台まで列車で向かう。", "仙台"),
        ("tr-part-02", "particle", "港では風が強い。", "港", "港にも船が着いた。", "港"),
        ("tr-part-03", "particle", "図書館から本を借りる。", "図書館", "図書館へ本を返す。", "図書館"),
        ("tr-part-04", "particle", "会議では案を比べる。", "会議", "会議にも担当者が来る。", "会議"),
        ("tr-comp-01", "compound", "音声認識装置を試した。", "音声", "音声合成装置を試した。", "音声"),
        ("tr-comp-02", "compound", "画像分類器を更新した。", "画像", "画像生成器を更新した。", "画像"),
        ("tr-comp-03", "compound", "交通情報網を整備する。", "交通", "交通制御網を整備する。", "交通"),
        ("tr-comp-04", "compound", "気象観測所で測った。", "気象", "気象予報士が説明した。", "気象"),
        ("tr-struct-01", "structured", "設定はrate=0.02とした。", "設定はrate", "設定はrate: 0.02とした。", "設定はrate"),
        ("tr-struct-02", "structured", "応答はstatus=readyだった。", "応答はstatus", "応答はstatus: readyだった。", "応答はstatus"),
        ("tr-struct-03", "structured", "引数にmode=fastを渡す。", "引数にmode", "辞書へmode: fastを入れる。", "辞書へmode"),
        ("tr-struct-04", "structured", "記録へcount=12を書く。", "記録へcount", "記録へcount: 12を書く。", "記録へcount"),
        ("tr-id-01", "identifier", "batchSizeを増やした。", "batch", "batch_sizeを増やした。", "batch"),
        ("tr-id-02", "identifier", "cacheLimitを確認した。", "cache", "cache_limitを確認した。", "cache"),
        ("tr-id-03", "identifier", "requestCountを記録する。", "request", "request_countを記録する。", "request"),
        ("tr-id-04", "identifier", "workerStateを保存する。", "worker", "worker_stateを保存する。", "worker"),
    ]
    dev_specs = [
        ("dev-inf-01", "inflection", "波が岸へ寄せている。", "波が岸へ寄せ", "波が岸へ寄せた。", "波が岸へ寄せ"),
        ("dev-aux-01", "auxiliary", "地図を広げてみる。", "地図を広げてみ", "地図を広げてみた。", "地図を広げてみ"),
        ("dev-part-01", "particle", "工房から作品を運ぶ。", "工房", "工房へ材料を運ぶ。", "工房"),
        ("dev-comp-01", "compound", "衛星通信網を使う。", "衛星", "衛星観測網を使う。", "衛星"),
        ("dev-struct-01", "structured", "項目はlevel=3である。", "項目はlevel", "項目はlevel: 3である。", "項目はlevel"),
        ("dev-id-01", "identifier", "retryDelayを変える。", "retry", "retry_delayを変える。", "retry"),
    ]
    test_specs = [
        ("te-inf-01", "inflection", "霧が谷へ流れている。", "霧が谷へ流れ", "霧が谷へ流れた。", "霧が谷へ流れ"),
        ("te-aux-01", "auxiliary", "道具を並べておく。", "道具を並べてお", "道具を並べておいた。", "道具を並べてお"),
        ("te-part-01", "particle", "市場では魚を売る。", "市場", "市場にも客が集まる。", "市場"),
        ("te-comp-01", "compound", "海洋調査船が戻る。", "海洋", "海洋観測船が戻る。", "海洋"),
        ("te-struct-01", "structured", "属性はcolor=blueだった。", "属性はcolor", "属性はcolor: blueだった。", "属性はcolor"),
        ("te-id-01", "identifier", "bufferLengthを測る。", "buffer", "buffer_lengthを測る。", "buffer"),
    ]
    return {
        split: [record(*spec) for spec in specs]
        for split, specs in (
            ("train", train_specs),
            ("dev", dev_specs),
            ("test", test_specs),
        )
    }


def main() -> None:
    args = parse_args()
    splits = build_splits()
    all_texts = {
        side["text"]
        for records in splits.values()
        for item in records
        for side in (item["left"], item["right"])
    }
    if args.exclusion_probe is not None:
        probe = json.loads(args.exclusion_probe.read_text(encoding="utf-8"))
        overlap = all_texts & {record["text"] for record in probe["records"]}
        if overlap:
            raise ValueError(f"evaluation probe text leaked into C1 data: {overlap}")
    payload = {
        "version": 1,
        "objective": "c1_stage0_landmark_probability_mse",
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "records": {k: len(v) for k, v in splits.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
