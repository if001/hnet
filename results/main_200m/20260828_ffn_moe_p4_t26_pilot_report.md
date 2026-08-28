# T26 FFN-MoE pilot実験レポート

作成日: 2026-08-28

## 1. 目的と結論

本実験の目的は、main networkのTransformer/KDA/Gated MLAというsequence mixerを変えず、FFNだけを
Mixture of Experts（MoE）にしたとき、H-Netの日本語分割品質が改善するかを確認することである。長期間学習用の
約200M active-parameterモデルを探索する途中段階であり、生成品質や小規模lossの僅差ではなく、学習中に形成される
言語的に説明可能な境界の軌跡を主に評価した。

T26では、FFN-MoEによりnative条件のcategory coverage、fracture、文節precision/coverageが改善し、T26の強みである
integrityも維持された。step 220の112文では、一つのexpertが90%以上を占めるrouter collapseはなく、expertは異なる
更新方向を学習していた。一方、family precisionは一貫して低下し、固定境界予算のlow/central条件ではfamily coverageと
landmark coverageも低下した。このためFFN-MoEを最終採用とはせず、T26 pilotを通過した同一仕様をK1G1/K3G1へ適用し、
改善がT26固有か、異なるmain networkでも再現するかを次に確認する。

## 2. 比較した構成

| 構成 | main network | FFN | 役割 |
| --- | --- | --- | --- |
| M0 | Transformer 26層（T26） | 通常のdense SwiGLU | control |
| M1 | Transformer 26層（T26） | 4 expert、top-1 FFN-MoE | variant |

Top-1は各token/chunkを最も確率の高いexpert一つだけで処理する方式である。capacity factor 1.25は、各expertが
1回のforwardで受け入れる上限を、完全均等時の1.25倍にする設定である。上限を超えた割当はdropされ、当該FFN出力は
ゼロになるが、residual pathは残る。load-balance補助lossの重みは0.01とした。

比較ではmodel初期値、データ順、学習時乱数をすべて42に固定した。MoEの全expertはdense FFNと同じweightから開始し、
FFN以外のweightもcontrolと一致させた。したがって、開始時点の既存機能差ではなく、routingとexpert分化の学習効果を
見ている。context長は2,048、評価はstep 10から220まで10 step間隔、固定profileはstep 110/165/220で行った。

## 3. モデル規模

| 構成 | 保存parameter | 1 tokenでactiveとなる概算parameter |
| --- | ---: | ---: |
| dense T26 | 219,850,496 | 219,850,496 |
| T26 FFN-MoE | 587,980,544 | 219,930,368 |

保存parameterはcheckpointに保持する全expertを数えるため大きくなる。active parameterはtop-1で実際に選ばれるexpert一つを
数えた概算で、dense T26とほぼ同じである。本探索でいう「約200M」はactive sizeを指す。ただし、現在の実装は全expertを
同じdeviceに保持するため、保存容量とGPU memoryは約200M denseモデルより大きい。

## 4. 分割評価指標

Full112は、日本語88文のcategory probeと、活用・複合語などの対応関係を持つ24文のfamily probeからなる。主評価は
stage 1のnative条件である。nativeはモデル自身の境界判定を使う。low/central/highは境界数を固定し、同じ境界予算で
境界の順位付けを比較するprofile条件である。

| 指標 | 何を測るか | 良い方向 |
| --- | --- | --- |
| category precision (P) | 選んだ注目箇所の境界のうち、言語的に説明可能な割合 | 大きい |
| category coverage (C) | 期待するcategory境界をどの程度拾うか | 大きい |
| category fracture (F) | 語彙内部など、保護したい箇所を不自然に割る割合 | 小さい |
| family precision / coverage | 活用・派生・複合語など、関連文群で説明可能な境界の正確さ／被覆 | 大きい |
| landmark coverage | family内で共通して期待する代表境界を拾う割合 | 大きい |
| integrity | 分割を避けたいfamily内部を保つ割合 | 大きい |
| 文節 precision / coverage | 文節境界として説明可能な分割の正確さ／被覆 | 大きい |

precisionは不要な境界を増やすと低下し、coverageは境界を増やすと上がりやすい。したがって単独ではなく、fracture、
integrity、実際の境界数と合わせて解釈する。

## 5. native条件の結果

### 5.1 step 110--220の平均差

正の値はMoEがdenseより大きいことを示す。fractureだけは負が改善である。

| 指標 | MoE - dense | 解釈 |
| --- | ---: | --- |
| category P | +0.017 | 小幅改善 |
| category C | +0.084 | 改善 |
| category F | -0.047 | 不自然な語内分割が減少 |
| family P | -0.146 | 明確な退行 |
| family C | +0.008 | ほぼ同等 |
| landmark C | +0.024 | 小幅改善 |
| integrity | +0.014 | T26の強みを維持 |
| 文節 P | +0.103 | 改善 |
| 文節 C | +0.225 | 大幅改善 |

### 5.2 terminal window（step 180--220）の平均差

| 指標 | MoE - dense |
| --- | ---: |
| category P / C / F | +0.037 / +0.114 / -0.082 |
| family P / C | -0.141 / +0.024 |
| landmark C / integrity | +0.067 / +0.017 |
| 文節 P / C | +0.073 / +0.235 |

後半でもcategory・文節の改善とfracture低下は残り、単一checkpointだけの上振れではなかった。設定したprimary制約にも
terminalの5時点すべてで違反しなかった。一方、family precision低下も後半まで残っており、他指標の改善で相殺して
無視できる変化ではない。

## 6. 固定境界予算profile

step 110/165/220の3時点平均について、MoEからdenseを引いた差を示す。

| 条件 | category P/C/F | family P/C | landmark C | integrity | 文節 P/C |
| --- | --- | --- | ---: | ---: | --- |
| low | +.030 / -.003 / -.159 | -.209 / -.162 | -.139 | +.042 | +.012 / -.044 |
| central | -.031 / -.014 / +.004 | -.192 / -.101 | -.153 | .000 | -.009 / -.029 |
| high | -.061 / -.016 / +.011 | -.104 / -.010 | -.014 | -.056 | +.082 / +.073 |
| native | +.040 / +.100 / -.068 | -.165 / +.030 | +.028 | -.014 | +.135 / +.304 |

固定予算では境界数がdenseとMoEで完全に同じである。それでもlow/centralでfamily Cとlandmark Cが下がるため、MoEが
すべての候補境界を一様に良く順位付けしたとは言えない。nativeの平均境界数はstep 165でdense 6.49に対しMoE 8.05、
step 220で6.40に対し7.68だった。したがってnative改善の一部は、MoEが境界を約1.3個多く選んだことによる。ただし
境界を増やしたにもかかわらずcategory precisionとfractureも改善しており、単なる過分割だけでは説明できない。

## 7. routingとexpert分化

### 7.1 指標

- 最大expert占有率: 各層で最も多く選ばれたexpertの割合。1に近いほど一つへ集中し、0.25なら4 expertが均等である。
- routing entropy: expert選択確率の不確実性を0--1へ正規化した値。大きいほどrouter確率は均等に近い。
- drop率: capacity上限のためexpert処理を受けられなかったtoken/chunkの割合。小さいほどよい。
- expert更新cosine: 各expertの`step 220 - 初期weight`同士の方向類似度。1ならほぼ同じ学習、0付近なら異なる方向を学んだことを表す。

### 7.2 112文での推移

| step | 層平均の最大expert占有率 | 層別最大 | 90%超の層 | drop率 | entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 110 | .554 | .816 | 0 | .223 | .996 |
| 165 | .545 | .865 | 0 | .225 | .994 |
| 220 | .412 | .590 | 0 | .117 | .995 |

step 220では集中とdropが明確に下がり、collapseは見られない。学習batchでのstep 220 drop率は.023で、112文を
1文ずつ通した診断より低かった。短い個別forwardではexpertごとのcapacityが小さく、偏りをbatch内で平均化できないためである。
これは推論時の実装条件に依存するため、長文・実運用batchでも再測定が必要である。

step 220のexpert更新cosineは26層平均.085（全pair範囲.015--.199）、初期weightに対する更新normは平均8.8%だった。
全expertは同一FFNから開始したため、この低いcosineはexpertが異なる更新方向へ分化したことを表す。カテゴリ別の割当は
identifier、number/unit、compound、structuredで全体分布との差が比較的大きかったが、有限probe上の相関であり、
各expertがその言語カテゴリを因果的に専門化したとは断定しない。

## 8. 判断と次の比較

M1は次のT26 pilot条件を満たした。

1. step 220で90%を超えるexpert独占がない。
2. T26のanchor指標であるintegrityをnative terminalで維持した。
3. category coverage、fracture、文節P/Cという複数の弱点をterminalで改善した。
4. expert間の更新が分化し、保存parameter増加だけの同一FFN複製には留まっていない。

したがって、4 expert・top-1・capacity factor 1.25・aux weight 0.01を変更せず、K1G1とK3G1のpaired controlへ
step 100まで展開する。そこでK1G1のcoverage/integrity、K3G1のcategory P/Cを壊す場合は停止する。通過したanchorだけを
step 220へ延長する。特にfamily precisionの退行と、短い個別forwardでのdrop率を非相殺の注意項目として追跡する。

## 9. 成果物

- 計画: `plan/20260826_factorized_200m_base_model_search/03_moe_factors.md`
- 学習run: Drive `runs/ffn_moe_p4_v1/`
- Full112集計: Drive `analysis/ffn_moe_p4_v1_step220/`
- profile集計: Drive `analysis/ffn_moe_p4_v1_profiles/`
- routing raw: Drive `evals/ffn_moe_p4_v1/routing/`
- routing・境界数診断: Drive `analysis/ffn_moe_p4_v1_diagnostics_summary.json`
- expert分化診断: Drive `analysis/ffn_moe_p4_v1_expert_parameter_divergence.json`
