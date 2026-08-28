from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the C2 integrity-margin pilot.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-probe", type=Path)
    return parser.parse_args()


def side(text: str, protected_surface: str) -> dict[str, Any]:
    if text.count(protected_surface) != 1:
        raise ValueError(f"protected surface must occur exactly once: {protected_surface}")
    character_start = text.index(protected_surface)
    prefix = text[:character_start]
    start = len(prefix.encode("utf-8"))
    end = start + len(protected_surface.encode("utf-8"))
    return {
        "text": text,
        "landmark_byte": end,
        "protected_span": {"start_byte": start, "end_byte": end},
        "protected_surface": protected_surface,
    }


def record(
    pair_id: str,
    category: str,
    left_text: str,
    left_surface: str,
    right_text: str,
    right_surface: str,
) -> dict[str, Any]:
    return {
        "id": pair_id,
        "category": category,
        "left": side(left_text, left_surface),
        "right": side(right_text, right_surface),
    }


def build_splits() -> dict[str, list[dict[str, Any]]]:
    train_specs = [
        ("tr-inf-01", "inflection", "担当者が結果を確認している。", "確認", "担当者が結果を確認した。", "確認"),
        ("tr-inf-02", "inflection", "講師が手順を説明している。", "説明", "講師が手順を説明した。", "説明"),
        ("tr-inf-03", "inflection", "技師が装置を調整している。", "調整", "技師が装置を調整した。", "調整"),
        ("tr-inf-04", "inflection", "係員が記録を保存している。", "保存", "係員が記録を保存した。", "保存"),
        ("tr-aux-01", "auxiliary", "再確認が必要だった。", "必要", "再確認が必要である。", "必要"),
        ("tr-aux-02", "auxiliary", "この操作は実行可能だった。", "可能", "この操作は実行可能である。", "可能"),
        ("tr-aux-03", "auxiliary", "列車は到着するはずだった。", "はず", "列車は到着するはずである。", "はず"),
        ("tr-aux-04", "auxiliary", "会議は開催予定だった。", "予定", "会議は開催予定である。", "予定"),
        ("tr-part-01", "particle", "仙台から列車に乗る。", "仙台", "仙台まで列車で向かう。", "仙台"),
        ("tr-part-02", "particle", "工房から作品を運ぶ。", "工房", "工房へ材料を運ぶ。", "工房"),
        ("tr-part-03", "particle", "市場では魚を売る。", "市場", "市場にも客が集まる。", "市場"),
        ("tr-part-04", "particle", "会議では案を比べる。", "会議", "会議にも担当者が来る。", "会議"),
        ("tr-comp-01", "compound", "音声認識装置を試した。", "認識", "画像認識装置を試した。", "認識"),
        ("tr-comp-02", "compound", "画像分類器を更新した。", "分類", "文書分類器を更新した。", "分類"),
        ("tr-comp-03", "compound", "交通制御網を整備する。", "制御", "温度制御器を整備する。", "制御"),
        ("tr-comp-04", "compound", "気象観測所で測った。", "観測", "衛星観測網で測った。", "観測"),
        ("tr-struct-01", "structured", "設定はrate=0.02とした。", "rate", "設定はrate: 0.02とした。", "rate"),
        ("tr-struct-02", "structured", "応答はstatus=readyだった。", "status", "応答はstatus: readyだった。", "status"),
        ("tr-struct-03", "structured", "引数にmode=fastを渡す。", "mode", "辞書へmode: fastを入れる。", "mode"),
        ("tr-struct-04", "structured", "記録へcount=12を書く。", "count", "記録へcount: 12を書く。", "count"),
        ("tr-id-01", "identifier", "batchSizeを増やした。", "batch", "batch_sizeを増やした。", "batch"),
        ("tr-id-02", "identifier", "cacheLimitを確認した。", "cache", "cache_limitを確認した。", "cache"),
        ("tr-id-03", "identifier", "requestCountを記録する。", "request", "request_countを記録する。", "request"),
        ("tr-id-04", "identifier", "workerStateを保存する。", "worker", "worker_stateを保存する。", "worker"),
    ]
    dev_specs = [
        ("dev-inf-01", "inflection", "担当者が内容を修正している。", "修正", "担当者が内容を修正した。", "修正"),
        ("dev-aux-01", "auxiliary", "この変更は有効だった。", "有効", "この変更は有効である。", "有効"),
        ("dev-part-01", "particle", "港町から船が出る。", "港町", "港町へ船が戻る。", "港町"),
        ("dev-comp-01", "compound", "衛星通信網を使う。", "通信", "無線通信網を使う。", "通信"),
        ("dev-struct-01", "structured", "項目はlevel=3である。", "level", "項目はlevel: 3である。", "level"),
        ("dev-id-01", "identifier", "retryDelayを変える。", "retry", "retry_delayを変える。", "retry"),
    ]
    test_specs = [
        ("te-inf-01", "inflection", "担当者が資料を整理している。", "整理", "担当者が資料を整理した。", "整理"),
        ("te-aux-01", "auxiliary", "この方式は安全だった。", "安全", "この方式は安全である。", "安全"),
        ("te-part-01", "particle", "農場から野菜を運ぶ。", "農場", "農場へ道具を運ぶ。", "農場"),
        ("te-comp-01", "compound", "海洋調査船が戻る。", "調査", "地質調査隊が戻る。", "調査"),
        ("te-struct-01", "structured", "属性はcolor=blueだった。", "color", "属性はcolor: blueだった。", "color"),
        ("te-id-01", "identifier", "bufferLengthを測る。", "buffer", "buffer_lengthを測る。", "buffer"),
    ]
    return {
        split: [record(*spec) for spec in specs]
        for split, specs in (("train", train_specs), ("dev", dev_specs), ("test", test_specs))
    }


def main() -> None:
    args = parse_args()
    splits = build_splits()
    all_texts = {
        side_payload["text"]
        for records in splits.values()
        for item in records
        for side_payload in (item["left"], item["right"])
    }
    if args.exclusion_probe is not None:
        probe = json.loads(args.exclusion_probe.read_text(encoding="utf-8"))
        overlap = all_texts & {item["text"] for item in probe["records"]}
        if overlap:
            raise ValueError(f"evaluation probe text leaked into C2 data: {overlap}")
    payload = {
        "version": 1,
        "objective": "c2_stage0_integrity_margin",
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "records": {key: len(value) for key, value in splits.items()}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
