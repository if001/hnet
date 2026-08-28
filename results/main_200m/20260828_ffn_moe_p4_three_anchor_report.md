# FFN-MoE P4 三構成比較レポート

作成日: 2026-08-28

## 1. 目的と結論

本実験の目的は、main networkのsequence mixerを固定したままFFNだけをMixture of Experts（MoE）へ変更すると、
H-Netの日本語分割品質を改善できるかを確認することである。約200M active-parameterで長期間学習するベースモデルの
構成探索であり、小規模学習のlossや生成文ではなく、言語的に説明可能な境界が学習中にどう形成されるかを主に評価した。

結論は、FFN-MoEは全構成へ一様に効く改善ではないが、T26とK3G1では異なる分割特性を作る有効な要因だった。

- T26ではcategory coverage、fracture、文節P/C、integrityが改善したが、family precisionが低下した。
- K1G1ではfracture、precision、integrityが改善した一方、K1G1の強みであるcategory/family coverageと文節P/Cを
  壊したためstep 100で停止した。
- K3G1ではcategory P/C、fracture、family precision、integrity、文節P/Cがstep 220まで改善した。一方、family
  coverageはterminalで0.127低下した。

K3G1 FFN-MoEは、境界を単純に増やしたモデルではない。step 220のnative条件では1文当たりの境界をdense K3G1より
約0.98個減らしながら、category coverageと文節coverageを高めた。すなわち、少数のfamily境界を高いprecisionで選び、
category・文節境界を優先する方向へ境界配分を変えたモデルである。family coverageを重視する汎用候補としては未確定だが、
K3G1の弱かったfracture・integrity・文節を改善するarchitecture-specific候補として残す。

## 2. 比較構成と統制条件

| anchor | main network | anchorとしての意味 |
| --- | --- | --- |
| T26 | Transformer 26層 | KDAを使わないbaseline。family precisionとintegrityが強い |
| K1G1 | KDA 13層 + Gated MLA 13層 | coverage/integrityが比較的強い |
| K3G1 | KDA 19層 + Gated MLA 7層 | 高KDA dosage。category P/Cは強いがfracture・family・文節が弱い |

各anchorについて通常のdense SwiGLU FFNと、4 expertのtop-1 FFN-MoEを比較した。top-1は各token/chunkを一つの
expertだけで処理する方式である。expert capacityは均等割当時の1.25倍、load-balance補助lossの重みは0.01とした。
MoEはinnermost main networkのFFNだけへ入れ、Encoder/Decoder、boundary router、sequence mixerは変更していない。

model初期値、データ順、学習時乱数のseedはすべて42で固定した。各MoE expertは対応するdense FFNと同じweightを複製して
開始し、FFN以外のweightもcontrolと一致させた。context長は2,048、LR schedule horizonは220 step、累積入力byteも
同一stepで一致する。したがって、paired差に初期weightやデータ順の差は含まれない。

T26とK3G1はstep 220まで実行した。K1G1は早期gateでcoverageの継続退行が確認されたためstep 100で停止した。

## 3. モデル規模とMoE用語

FFN-MoEは4 expertのweightを保存するため、保存parameterは約588Mになる。一方、top-1では1 tokenにつき一つのexpert
だけがactiveになるため、active parameterは約220Mでdense anchorとほぼ同じである。ただし現在の実装は全expertを
同じGPUに置くため、checkpoint容量とGPU memoryは220M denseモデルより大きい。

- expert占有率: 各層で最も多く選ばれたexpertの割合。1に近いほど一つへ集中し、0.25なら4 expertが均等である。
- router collapse: ほぼすべてを一つのexpertへ送る状態。本計画では90%超を失敗基準とした。
- drop率: capacity上限のためexpert FFNで処理されなかったtoken/chunkの割合。小さいほどよい。
- routing entropy: expert選択確率の不確実性を0--1へ正規化した値。大きいほど確率分布が均等に近い。

## 4. 分割評価指標

Full112は、日本語88文のcategory probeと、活用・複合語などの対応関係を持つ24文のfamily probeからなる。主評価は
stage 1のnative条件で行った。nativeはモデル自身の閾値による境界を使う。low/central/highは境界数を固定し、同じ
境界予算で候補境界の順位付けを比較するprofile条件である。

| 指標 | 測る内容 | 良い方向 |
| --- | --- | --- |
| category precision (P) | 選んだ境界のうち、注目categoryとして説明できる割合 | 大きい |
| category coverage (C) | 期待するcategory境界を拾った割合 | 大きい |
| category fracture (F) | 語彙内部など、保護したい箇所を不自然に割った文の割合 | 小さい |
| family P/C | 活用・派生・複合語など、関連文群での境界の正確さ／被覆 | 大きい |
| landmark C | family内で共通して期待する代表境界を拾った割合 | 大きい |
| integrity | 分割を避けたいfamily内部を保った割合 | 大きい |
| 文節 P/C | 文節境界として説明できる分割の正確さ／被覆 | 大きい |

precisionとcoverageにはtrade-offがある。少数だけを選ぶとprecisionは上がりやすくcoverageは下がりやすいため、
fracture、integrity、境界数、固定予算profileと合わせて評価する。

## 5. 三構成のnative結果

### 5.1 T26: step 180--220平均のMoE差

| category P/C/F | family P/C | landmark C / integrity | 文節 P/C |
| --- | --- | --- | --- |
| +.037 / +.114 / -.082 | -.141 / +.024 | +.067 / +.017 | +.073 / +.235 |

fractureは小さいほど良いため、`-.082`は改善である。T26の強みであるintegrityを保ちながら複数の弱点を改善したが、
family precisionは後半まで低下した。詳細な固定予算profileとexpert分化は
`20260828_ffn_moe_p4_t26_pilot_report.md`に記載した。

### 5.2 K1G1: step 70--100平均のMoE差

| category P/C/F | family P/C | landmark C / integrity | 文節 P/C |
| --- | --- | --- | --- |
| +.032 / -.052 / -.156 | +.077 / -.098 | -.031 / +.083 | -.085 / -.174 |

precision、fracture、integrityは改善したが、category coverage、family coverage、文節P/Cが同時に低下した。
family coverageはstep 80、90、100でcontrolに対する許容退行幅`-.05`を継続して超えた。K1G1を残す理由である
coverageを壊しているため、他指標で相殺せずstep 100で停止した。

### 5.3 K3G1: step 180--220平均のMoE差

| category P/C/F | family P/C | landmark C / integrity | 文節 P/C |
| --- | --- | --- | --- |
| +.045 / +.053 / -.098 | +.396 / -.127 | -.033 / +.267 | +.178 / +.339 |

K3G1ではcategory P/Cというanchorの強みを維持・改善し、従来弱かったfracture、integrity、文節P/Cも大きく改善した。
family precisionも上がったが、family coverageはstep 190--210で許容退行幅を超えた。step 220実値はfamily Pが
dense `.259`からMoE `.750`へ上がる一方、family Cは`.212`から`.182`へ下がった。高precision・低coverageへの
変化であり、family全体を広く拾う改善ではない。

## 6. K3G1の固定境界予算profile

step 110/165/220の3時点平均について、MoEからdenseを引いた差を示す。

| 条件 | category P/C/F | family P/C | landmark C / integrity | 文節 P/C |
| --- | --- | --- | --- | --- |
| low | +.030 / +.070 / +.030 | +.101 / +.111 | +.181 / -.111 | -.067 / -.072 |
| central | +.033 / +.074 / .000 | -.156 / +.020 | +.056 / -.069 | -.110 / -.043 |
| high | +.056 / +.053 / -.019 | +.298 / +.061 | +.083 / +.125 | +.056 / +.087 |
| native | +.024 / +.027 / -.095 | +.285 / -.091 | -.014 / +.208 | +.122 / +.275 |

high compressionでは9指標すべてが改善または維持され、primary制約違反もなかった。lowではfractureとintegrity、
centralではfamily precision・integrity・文節が弱い。したがって、K3G1 MoEは境界候補の順位付けを全面的に改善した
わけではなく、使う境界予算によって強みが変わる。

固定profileの境界数はdenseとMoEで同一である。nativeでは1文当たりの平均境界数がMoEで減り、denseとの差はstep 110
`-0.35`、step 165 `-0.88`、step 220 `-0.98`だった。それでもcategory Cと文節Cが上がったため、MoEは境界を
増やしたのではなく、family境界の一部をcategory・文節境界へ再配分したと解釈できる。

## 7. routing安定性

112文を1文ずつ通したK3G1 MoEのrouting結果を示す。

| step | 層平均の最大expert占有率 | 層別最大 | 90%超の層 | drop率 | entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | .397 | .586 | 0 | .109 | .975 |
| 110 | .416 | .652 | 0 | .116 | .964 |
| 165 | .443 | .714 | 0 | .143 | .947 |
| 220 | .411 | .559 | 0 | .119 | .942 |

一つのexpertへのcollapseはなく、step 220で集中が再び下がった。学習batchでのstep 220 drop率は`.0004`で、1文ずつの
probeより小さい。短い個別forwardではexpertごとのcapacityが小さく、batch内で割当を平均化できないためである。
長文推論では実際のbatching条件に合わせてdrop率を再測定する必要がある。

## 8. 判断

FFN-MoEを全anchor共通の標準FFNへ昇格させる根拠はない。K1G1では主要な強みを壊し、T26/K3G1でもfamilyの
precision/coverage trade-offが残った。一方、次の二つは有効な候補として保持する。

1. T26 FFN-MoE: integrityを維持し、category coverage・fracture・文節を改善する候補。
2. K3G1 FFN-MoE: 高KDA構成のcategory P/Cを維持しながら、fracture・integrity・文節を改善する候補。

K3G1 FFN-MoEはfamily coverageが低いため、現時点でquality candidateとは確定しない。今後比較へ含める場合は、
family coverageの最低許容値を先に置き、category・文節の改善で相殺しない。また、FFN-MoEのexpert数、capacity、aux weightを
このFull112だけへ細かく最適化する探索は行わない。次の独立要因であるMixer-MoEを先に評価し、固定layoutの変更と
FFN容量の変更のどちらが境界形成へ有効かを分離する。

## 9. 成果物

- 計画: `plan/20260826_factorized_200m_base_model_search/03_moe_factors.md`
- T26 pilot詳細: `results/main_200m/20260828_ffn_moe_p4_t26_pilot_report.md`
- 学習run: Drive `runs/ffn_moe_p4_v1/`
- K1G1集計: Drive `analysis/ffn_moe_p4_v1_k1g1_step100/`
- K3G1 native集計: Drive `analysis/ffn_moe_p4_v1_k3g1_step220/`
- K3G1 profile集計: Drive `analysis/ffn_moe_p4_v1_k3g1_profiles/`
- routing raw: Drive `evals/ffn_moe_p4_v1/routing/`
- K3G1 routing集計: Drive `analysis/ffn_moe_p4_v1_k3g1_routing_summary.json`
