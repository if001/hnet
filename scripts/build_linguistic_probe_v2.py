from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def make_record(
    identifier: str,
    category: str,
    text: str,
    surface: str,
    segmentations: list[str],
    *,
    protected: list[str] | None = None,
    pair: dict[str, str] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": identifier,
        "category": category,
        "text": text,
        "focus": {
            "surface": surface,
            "occurrence": 0,
            "acceptable_segmentations": segmentations,
        },
    }
    if protected:
        record["focus"]["protected_substrings"] = protected
    if pair:
        record["pair"] = pair
    return record


def pair(group: str, kind: str, variant: str) -> dict[str, str]:
    return {"group": group, "kind": kind, "variant": variant}


def existing_records() -> list[dict[str, Any]]:
    source = json.loads(
        (ROOT / "configs/linguistic_boundary_probe_v1.json").read_text(
            encoding="utf-8"
        )
    )
    protected = {
        "auxiliary-negative-01": ["変更"],
        "compound-nlp-01": ["自然", "言語", "処理", "モデル"],
        "compound-boundary-01": ["動的", "チャンク", "境界"],
        "compound-experiment-01": ["境界", "介入", "実験", "結果"],
        "compound-fragmentation-01": ["分割", "結果"],
        "proper-noun-01": ["東京", "千代田"],
        "identifier-01": ["validation_loss", "step"],
        "path-01": ["content", "drive", "MyDrive", "hnet_agent"],
        "json-01": ["query", "東京", "天気"],
        "code-call-01": ["torch", "allclose"],
        "control-boundary-01": ["動的", "チャンク", "境界"],
        "control-boundary-02": ["動的", "チャンク", "境界"],
    }
    records = source["records"]
    for record in records:
        values = protected.get(record["id"])
        if values:
            record["focus"]["protected_substrings"] = values
    return records


def added_records() -> list[dict[str, Any]]:
    r = make_record
    return [
        # inflection: 4 existing + 4 additions
        r("inflection-past-01", "inflection", "彼は昨日まで笑っていた。", "笑っていた", ["笑|っていた", "笑って|いた"]),
        r("inflection-compound-01", "inflection", "結果をもう一度書き直した。", "書き直した", ["書き|直した", "書き直し|た"]),
        r("inflection-loading-01", "inflection", "ログを読み込んでいる。", "読み込んでいる", ["読み|込んでいる", "読み込んで|いる"]),
        r("inflection-conditional-01", "inflection", "設定を変更できれば続行する。", "変更できれば", ["変更|できれば", "変更でき|れば"], protected=["変更"]),

        # auxiliary: 2 existing + 6 additions
        r("auxiliary-passive-progressive-01", "auxiliary", "重みは既に保存されている。", "保存されている", ["保存|されている", "保存されて|いる"], protected=["保存"]),
        r("auxiliary-completive-01", "auxiliary", "誤った設定で実行してしまった。", "実行してしまった", ["実行|してしまった", "実行して|しまった"], protected=["実行"]),
        r("auxiliary-obligation-01", "auxiliary", "出力を確認しなければならない。", "確認しなければならない", ["確認|しなければならない", "確認しなければ|ならない"], protected=["確認"]),
        r("auxiliary-causative-past-01", "auxiliary", "担当者に設定を読み込ませた。", "読み込ませた", ["読み|込ませた", "読み込ま|せた"]),
        r("auxiliary-desiderative-01", "auxiliary", "別のデータでも学習したい。", "学習したい", ["学習|したい", "学習し|たい"], protected=["学習"]),
        r("auxiliary-potential-negative-01", "auxiliary", "現在の権限では更新できない。", "更新できない", ["更新|できない", "更新でき|ない"], protected=["更新"]),

        # particle: 2 existing + 6 additions
        r("particle-range-01", "particle", "東京から大阪まで移動する。", "東京から大阪まで", ["東京|から|大阪|まで", "東京から|大阪まで"], protected=["東京", "大阪"]),
        r("particle-topic-compound-01", "particle", "結果について確認する。", "結果について", ["結果|について"], protected=["結果"]),
        r("particle-cause-01", "particle", "条件によって出力が変化する。", "条件によって", ["条件|によって"], protected=["条件"]),
        r("particle-purpose-01", "particle", "実験のために環境を準備する。", "実験のために", ["実験|の|ため|に", "実験の|ために"], protected=["実験"]),
        r("particle-limiter-01", "particle", "今回はモデルだけを更新する。", "モデルだけを", ["モデル|だけ|を", "モデルだけ|を"], protected=["モデル"]),
        r("particle-additive-01", "particle", "入力にも出力にも同じ規則を使う。", "入力にも出力にも", ["入力|に|も|出力|に|も", "入力にも|出力にも"], protected=["入力", "出力"]),

        # bunsetsu: 2 existing + 6 additions
        r("bunsetsu-causal-02", "bunsetsu", "学習が終わったので、結果を保存した。", "学習が終わったので、結果を", ["学習が|終わったので|、|結果を", "学習が終わったので|、|結果を"]),
        r("bunsetsu-concessive-01", "bunsetsu", "失敗しても、条件を変えて再試行する。", "失敗しても、条件を", ["失敗しても|、|条件を", "失敗しても、|条件を"]),
        r("bunsetsu-discovery-01", "bunsetsu", "確認したところ、問題は見つからなかった。", "確認したところ、問題は", ["確認したところ|、|問題は", "確認したところ、|問題は"]),
        r("bunsetsu-conditional-02", "bunsetsu", "設定を変えると、結果も変化した。", "設定を変えると、結果も", ["設定を|変えると|、|結果も", "設定を変えると|、|結果も"], protected=["設定", "結果"]),
        r("bunsetsu-adversative-01", "bunsetsu", "時間がないにもかかわらず、実験を続けた。", "時間がないにもかかわらず、実験を", ["時間が|ないにもかかわらず|、|実験を", "時間がないにもかかわらず|、|実験を"], protected=["時間", "実験"]),
        r("bunsetsu-conditional-03", "bunsetsu", "必要であれば、追加の検証を行う。", "必要であれば、追加の検証を", ["必要であれば|、|追加の|検証を", "必要であれば、|追加の検証を"], protected=["必要", "追加", "検証"]),

        # compound: 4 existing + 4 additions
        r("compound-ml-01", "compound", "機械学習モデルを比較する。", "機械学習モデル", ["機械学習|モデル", "機械|学習|モデル"], protected=["機械", "学習", "モデル"]),
        r("compound-llm-01", "compound", "大規模言語モデルを訓練する。", "大規模言語モデル", ["大規模|言語|モデル", "大規模言語|モデル"], protected=["大規模", "言語", "モデル"]),
        r("compound-hpc-01", "compound", "高性能計算機を利用した。", "高性能計算機", ["高性能|計算機", "高|性能|計算機"], protected=["性能", "計算機"]),
        r("compound-society-01", "compound", "情報処理学会で発表する。", "情報処理学会", ["情報処理|学会", "情報|処理|学会"], protected=["情報", "処理", "学会"]),

        # punctuation: 2 existing + 6 additions
        r("punctuation-parentheses-01", "punctuation", "結果（暫定値）を共有した。", "（暫定値）", ["（|暫定値|）", "（暫定値）"], protected=["暫定"]),
        r("punctuation-short-quote-01", "punctuation", "画面に「開始」と表示する。", "「開始」と", ["「|開始|」|と", "「開始」|と"], protected=["開始"]),
        r("punctuation-colon-01", "punctuation", "注意：設定を確認する。", "注意：設定", ["注意|：|設定", "注意：|設定"], protected=["注意", "設定"]),
        r("punctuation-semicolon-01", "punctuation", "成功した；ただし警告が残った。", "成功した；ただし", ["成功した|；|ただし", "成功した；|ただし"]),
        r("punctuation-ellipsis-01", "punctuation", "処理中……しばらく待つ。", "処理中……しばらく", ["処理中|……|しばらく"], protected=["処理"]),
        r("punctuation-list-01", "punctuation", "項目A、項目B、項目Cを比較する。", "項目A、項目B、項目C", ["項目A|、|項目B|、|項目C"], protected=["項目"]),

        # proper noun: 1 existing + 7 additions
        r("proper-noun-kyoto-01", "proper_noun", "京都市左京区で開催する。", "京都市左京区", ["京都市|左京区", "京都|市|左京|区"], protected=["京都", "左京"]),
        r("proper-noun-un-01", "proper_noun", "国際連合安全保障理事会が決議した。", "国際連合安全保障理事会", ["国際連合|安全保障理事会", "国際|連合|安全保障|理事会"], protected=["国際", "連合", "安全", "保障", "理事会"]),
        r("proper-noun-university-01", "proper_noun", "東京大学大学院に進学した。", "東京大学大学院", ["東京大学|大学院", "東京|大学|大学院"], protected=["東京", "大学"]),
        r("proper-noun-person-01", "proper_noun", "山田太郎さんが発表した。", "山田太郎", ["山田|太郎", "山田太郎"], protected=["山田", "太郎"]),
        r("proper-noun-company-01", "proper_noun", "OpenAI Japanが資料を公開した。", "OpenAI Japan", ["OpenAI| |Japan", "OpenAI Japan"], protected=["OpenAI", "Japan"]),
        r("proper-noun-railway-01", "proper_noun", "東北新幹線で移動する。", "東北新幹線", ["東北|新幹線", "東北新幹線"], protected=["東北", "新幹線"]),
        r("proper-noun-country-01", "proper_noun", "アメリカ合衆国で公開された。", "アメリカ合衆国", ["アメリカ|合衆国", "アメリカ合衆国"], protected=["アメリカ", "合衆国"]),

        # number and unit: 1 existing + 7 additions
        r("number-decimal-01", "number_unit", "処理は3.14秒で完了した。", "3.14秒", ["3.14|秒", "3.14秒"]),
        r("number-memory-01", "number_unit", "128GBのメモリを使用する。", "128GB", ["128|GB", "128GB"], protected=["GB"]),
        r("number-date-01", "number_unit", "2026年8月20日に開始する。", "2026年8月20日", ["2026|年|8|月|20|日", "2026年|8月|20日"]),
        r("number-percent-01", "number_unit", "精度が10%向上した。", "10%", ["10|%", "10%"]),
        r("number-scientific-01", "number_unit", "学習率を1e-4に設定した。", "1e-4", ["1e-4"]),
        r("number-counter-01", "number_unit", "42件の結果を確認した。", "42件", ["42|件", "42件"]),
        r("number-distance-01", "number_unit", "5km先の施設へ移動した。", "5km", ["5|km", "5km"], protected=["km"]),

        # identifier: 2 existing + 6 additions
        r("identifier-camel-01", "identifier", "maxSequenceLengthを更新する。", "maxSequenceLength", ["max|Sequence|Length", "maxSequenceLength"], protected=["Sequence", "Length"]),
        r("identifier-env-01", "identifier", "CUDA_VISIBLE_DEVICESを確認する。", "CUDA_VISIBLE_DEVICES", ["CUDA|_|VISIBLE|_|DEVICES", "CUDA_VISIBLE_DEVICES"], protected=["CUDA", "VISIBLE", "DEVICES"]),
        r("identifier-url-01", "identifier", "https://example.com/api/v1へ接続する。", "https://example.com/api/v1", ["https|://|example.com|/|api|/|v1", "https://example.com/api/v1"], protected=["https", "example", "api"]),
        r("identifier-method-01", "identifier", "Trainer.fit_modelを呼び出す。", "Trainer.fit_model", ["Trainer|.|fit_model", "Trainer.fit_model"], protected=["Trainer", "fit_model"]),
        r("identifier-checkpoint-01", "identifier", "checkpoint_step_000220.ptを読む。", "checkpoint_step_000220.pt", ["checkpoint|_|step|_|000220|.|pt", "checkpoint_step_000220|.pt"], protected=["checkpoint", "step"]),
        r("identifier-option-01", "identifier", "--learning-rateを指定する。", "--learning-rate", ["--|learning-rate", "--learning|-|rate"], protected=["learning", "rate"]),

        # structured: 2 existing + 6 additions
        r("structured-tool-01", "structured", "tool callは{\"tool\":\"search\",\"query\":\"東京\"}だった。", "{\"tool\":\"search\",\"query\":\"東京\"}", ["{|\"tool\"|:|\"search\"|,|\"query\"|:|\"東京\"|}", "{\"tool\":\"search\",|\"query\":\"東京\"}"], protected=["tool", "search", "query", "東京"]),
        r("structured-xml-01", "structured", "応答は<result>成功</result>だった。", "<result>成功</result>", ["<result>|成功|</result>", "<|result|>|成功|</|result|>"], protected=["result", "成功"]),
        r("structured-markdown-01", "structured", "詳細は[実験結果](report.md)を参照する。", "[実験結果](report.md)", ["[|実験結果|]|(|report.md|)", "[実験結果]|(report.md)"], protected=["実験", "結果", "report"]),
        r("structured-assignment-01", "structured", "Pythonではloss = model(x)と書く。", "loss = model(x)", ["loss| |=| |model|(x)", "loss = |model(x)"], protected=["loss", "model"]),
        r("structured-shell-01", "structured", "端末でpython train.py --seed 42を実行する。", "python train.py --seed 42", ["python| |train.py| |--seed| |42", "python train.py| |--seed 42"], protected=["python", "train", "seed"]),
        r("structured-yaml-01", "structured", "設定はlearning_rate: 0.0001だった。", "learning_rate: 0.0001", ["learning_rate|:| |0.0001", "learning_rate:| |0.0001"], protected=["learning_rate"]),

        # context control: 2 existing + 6 additions
        r("control-additional-01", "context_control", "追加学習結果を確認した。", "追加学習結果", ["追加学習|結果", "追加|学習|結果"], protected=["追加", "学習", "結果"], pair=pair("additional-control", "control", "short")),
        r("control-additional-02", "context_control", "長時間の実験後に追加学習結果を詳しく確認した。", "追加学習結果", ["追加学習|結果", "追加|学習|結果"], protected=["追加", "学習", "結果"], pair=pair("additional-control", "control", "long")),
        r("control-negative-01", "context_control", "設定を変更していない。", "変更していない", ["変更|していない", "変更して|いない"], protected=["変更"], pair=pair("negative-control", "control", "short")),
        r("control-negative-02", "context_control", "確認したが、重要な設定を変更していない。", "変更していない", ["変更|していない", "変更して|いない"], protected=["変更"], pair=pair("negative-control", "control", "long")),
        r("dynamic-capability-01", "context_control", "この環境なら実行できる。", "実行できる", ["実行|できる"], protected=["実行"], pair=pair("capability-context", "dynamic", "capability")),
        r("dynamic-capability-02", "context_control", "画面に「実行できる」と表示された。", "実行できる", ["実行できる"], protected=["実行"], pair=pair("capability-context", "dynamic", "literal")),
    ]


def main() -> None:
    records = existing_records() + added_records()
    categories: dict[str, int] = {}
    for record in records:
        categories[record["category"]] = categories.get(record["category"], 0) + 1
    if set(categories.values()) != {8} or len(categories) != 11:
        raise RuntimeError(f"unexpected category counts: {categories}")
    payload = {
        "version": 2,
        "description": (
            "Expanded Japanese linguistic-boundary proxy. Alternative complete "
            "segmentations and protected lexical substrings are human annotations."
        ),
        "budget_profiles": [
            {"id": "low_compression", "stage0_units_per_chunk": 2.5, "stage1_units_per_chunk": 2.5},
            {"id": "central", "stage0_units_per_chunk": 3.0, "stage1_units_per_chunk": 3.0},
            {"id": "high_compression", "stage0_units_per_chunk": 3.5, "stage1_units_per_chunk": 3.5},
        ],
        "records": records,
    }
    output = ROOT / "configs/linguistic_boundary_probe_v2.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
