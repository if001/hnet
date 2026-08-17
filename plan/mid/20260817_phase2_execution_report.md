# Phase 2 中規模実験 最終実行・評価レポート

更新日: 2026-08-17

## 結論

Phase 2の次段階へ進めるmain networkとして **T26を選定する**。

事前登録した選定規則では、K1T1へ変更するには次のいずれかが必要だった。

1. validation BPBでT26を実用同等幅 `0.009627 BPB` より大きく上回る。
2. SFT後agent proxyの`full_exact_rate`を、pretraining seed 42、43の各々で
   T26より10 percentage points以上改善し、重大な退行がない。

K1T1のvalidation BPB改善量は `-0.001809 BPB`（K1T1の方が僅かに悪い）で、
第1条件を満たさない。agent proxyは両モデル・両pretraining seedで
`full_exact_rate = 0`となり、第2条件も満たさない。floor/tie時はT26とする
事前規則に加え、T26の実効raw-byte throughputがK1T1の約1.40倍だったため、
T26を採用する。

## 実行範囲と完了監査

- H-Net T26、K1T1、tokenizer 128k baselineをpretraining seed 42、43で
  それぞれ500M raw bytesまで学習した（計6 run）。
- 各runで125M raw bytesごとの4 checkpointを保存・評価した。
- 6個の最終checkpointすべてに共通SFTを実行した。SFT seedは42固定。
- 共通text BPB 6件、補正版境界介入4件、cross-model boundary transfer 4件、
  SFT整合agent proxy 6件を完了した（計20評価task、14新規完了・6再利用）。
- 復旧後のColab評価commitは`325502e`、評価関連unit testは14件passした。
- 集約JSON 2件と選定分析JSONの生成を完了した。

実行中にColab runtimeが切断されたため、A100 runtimeでDrive mount、repository clone、
依存packageを再構築した。manifestと既存artifactを監査して完了済みtaskを再利用し、
未完了taskのみforegroundで再実行した。復旧状態manifestは`completed`である。

## Pretraining結果

| model | final validation BPB mean | population SD | compression L2/L0 | raw bytes/s | peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| T26 | 1.479380 | 0.003725 | 9.164 | 34,360 | 4,561 |
| K1T1 | 1.481190 | 0.001901 | 12.072 | 24,609 | 3,809 |
| tokenizer 128k | 8.843027 | 0.004182 | n/a | 554,348 | 14,259 |

T26とK1T1の平均差はT26優位で0.001809 BPBで、実用同等幅の内側である。
K1T1のpeak allocated memoryはT26より16.5%小さい一方、T26は1.396倍高速だった。
compression L2/L0が9.16対12.07で一致しないため、厳密な計算量一致比較ではない。

| model | 125M | 250M | 375M | 500M raw bytes |
| --- | ---: | ---: | ---: | ---: |
| T26 | 1.760180 | 1.616113 | 1.543994 | 1.479380 |
| K1T1 | 1.761212 | 1.619991 | 1.545077 | 1.481190 |
| tokenizer 128k | 10.612680 | 9.308730 | 8.912530 | 8.843027 |

全checkpointでT26がK1T1を僅かに上回り、K1T1が学習後半で逆転する傾向は見られない。

## 共通固定probe

| model | overall micro BPB mean | population SD |
| --- | ---: | ---: |
| T26 | 2.024186 | 0.005709 |
| K1T1 | 2.030043 | 0.011601 |
| tokenizer 128k | 11.590713 | 0.057071 |

K1T1はT26より0.005858 BPB悪く、ここでも差は実用同等幅内である。11 category中、
T26が8 category、K1T1が3 categoryで低BPBだった。

| category | T26 | K1T1 | lower BPB |
| --- | ---: | ---: | --- |
| agent | 2.479546 | 2.500439 | T26 |
| clause | 1.682163 | 1.694124 | T26 |
| compound | 1.988145 | 2.037986 | T26 |
| context_pair | 1.686219 | 1.666591 | K1T1 |
| inflection | 1.899302 | 1.907124 | T26 |
| long_agent | 2.243560 | 2.166788 | K1T1 |
| long_code_json | 1.637112 | 1.678102 | T26 |
| long_dialogue | 1.862230 | 1.906170 | T26 |
| long_explanation | 1.797989 | 1.821984 | T26 |
| long_technical | 2.350731 | 2.330533 | K1T1 |
| mixed_ascii | 2.908636 | 2.929451 | T26 |

K1T1は`long_agent`など一部の長文categoryで良いが、一般的・一貫した優位ではない。

## 境界介入とboundary transfer

値はlearned boundaryに対する介入時の`delta BPB`であり、大きいほどそのモデルが
learned boundaryに依存していることを示す。各値はpretraining 2 seed平均である。

| model | fixed | morph | random | shifted-left | shifted-right |
| --- | ---: | ---: | ---: | ---: | ---: |
| T26 | 0.060370 | 0.028147 | 0.065547 | 0.058772 | 0.072784 |
| K1T1 | 0.078454 | 0.049530 | 0.076048 | 0.066818 | 0.095836 |

全モデル・全seed・全介入でdeltaは正で、学習済み境界は固定・形態素・random・shift境界
より良い。K1T1は介入感度が高いが、通常BPBや共通probeを改善していないため、これを
有用な構造化能力の優位とは判定しない。強い境界共適応または脆弱性の可能性もある。

| transfer direction | mean delta BPB | seed 42 95% CI | seed 43 95% CI |
| --- | ---: | --- | --- |
| K1T1 boundaries -> T26 | 0.022417 | [0.015347, 0.036129] | [0.010865, 0.028325] |
| T26 boundaries -> K1T1 | 0.032524 | [0.029995, 0.052686] | [0.016562, 0.031783] |

両方向・両seedでdeltaが正かつbootstrap 95% CIは0を跨がない。したがって境界は
完全に交換可能ではなく、各モデル内部で表現と共適応している。

## SFT後agent proxy

固定Level Aの18 taskを各SFT runへ実行した。

| model | pretraining seed | JSON valid | tool accuracy | argument exact | full exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| T26 | 42 | 0.0556 | 0 | 0 | 0 |
| T26 | 43 | 0 | 0 | 0 | 0 |
| K1T1 | 42 | 0 | 0 | 0 | 0 |
| K1T1 | 43 | 0 | 0 | 0 | 0 |
| tokenizer 128k | 42 | 0 | 0 | 0 | 0 |
| tokenizer 128k | 43 | 0 | 0 | 0 | 0 |

主要指標`full_exact_rate`は全runでfloorだった。したがって、このproxyは今回の短いSFTで
architecture差を判定できる難易度ではない。K1T1の10 percentage points改善条件は
両seedとも差0で不成立である。これは「agent能力が同等」という証拠ではなく、SFT量・
課題難度・生成安定性を見直す必要があることを示す。

## tokenizer baselineの扱い

tokenizer baselineはraw UTF-8 byteで正規化しており、共通probeでも高BPBを再現したため、
単純な正規化・集約ミスではない。ただし128k語彙構成では97,659,392 parameter中
65,536,000（67.1%）がshared embedding/headであり、500M raw bytesでは各tokenを
十分学習できなかった可能性が高い。本結果は今回のparameter-matched・短予算構成への
否定的結果であって、tokenizer方式一般の劣位とは結論しない。

## 評価定義の補正

本集約前のsmoke監査で次の2点を補正し、旧出力は最終集約から除外した。

1. 境界介入BPBを予測token数ではなくraw UTF-8 byte数で正規化し、共通BPBおよび
   boundary transferと定義を統一した。
2. agent promptをSFTのQwen3 chat envelope、`/no_think`、`<tools>...</tools>`へ
   合わせた。tool call主形式を`{"name": ..., "arguments": ...}`とした。

## Artifact

Google Drive root:
`/content/drive/MyDrive/hnet_agent_200m_main`

- 最終レポート: `reports/phase2_selection.md`
- pretraining/SFT集約: `reports/phase2_selection/phase2_summary.json`
- 評価集約: `reports/phase2_selection/phase2_evaluation_summary.json`
- 選定分析: `reports/phase2_selection/phase2_selection_analysis.json`
- run別・curve CSV: `reports/phase2_selection/phase2_*_metrics.csv`
- 共通BPB: `evals/general_phase2/<model>/<run>/`
- 境界介入: `evals/phase2_boundary/<model>/<run>/`
- boundary transfer: `evals/phase2_transfer/<direction>/<seed>/`
- agent proxy: `evals/agent_phase2/<model>/<run>/`
- 復旧・完了状態: `manifests/phase2_recovery_evaluation_status.json`

## 制約と次段階

- FLOPs/byteがloggerへ保存されておらず、FLOPs Paretoは評価できない。
- T26とK1T1のcompression L2/L0が一致せず、厳密な計算量一致ではない。
- SFT seedは42のみで、長時間段階で必要な複数SFT seed分散を評価していない。
- agent proxyはfloorで、agent能力に関するarchitecture比較には情報量がない。
- tokenizer 128k baselineは語彙に対して学習予算が小さく、方向確認に留まる。
- boundary transferは境界交換の診断であり、単独で下流能力の優位を意味しない。

長時間段階はまだ開始しない。開始前に、選定済みT26を主構成とし、tokenizer baselineの
語彙サイズ・embedding比率・学習予算を再設計する。またagent比較を選定材料にする場合は、
小さな到達可能taskでSFT recipeを校正してfloorを解消してから複数SFT seedで評価する。
