# Mixer-MoE P5 T26 pilot 実験レポート

作成日: 2026-08-29

## 1. 目的

本実験の目的は、約200M active-parameterで長期間学習するH-Netのベースモデル候補を探索するため、main networkの
sequence mixerを入力に応じて選ぶMixer-MoEが、日本語の分割品質を改善できるか確認することである。

H-Netではmain networkの構造が言語モデル本体だけでなく動的chunk境界にも影響する。これまでの固定構成では、Transformer
のみのT26、Kimi Delta Attentionを含むK3G1、KDAを少量含むK1G1などで、得意な分割categoryが異なった。そこで、固定された
一種類のmixerを全入力へ使う代わりに、Transformer attention、KDA、Gated MLAをchunkごとに選べば、それぞれの強みを
入力に応じて利用できるという仮説を検証した。

この段階は小規模学習による構成探索である。文章生成品質やlossの小差を最終性能とはみなさず、学習中に形成される境界が
言語的に説明可能か、その品質が複数stepで持続するかを主に評価した。

## 2. 比較した構成

- dense T26: main networkの26層すべてが通常のTransformer attentionである対照構成。
- T26 Mixer-MoE T/K/G-4: T26の中盤4層、0始まりの層11--14だけをMixer-MoEへ置換した構成。他の22層はT26と同じである。
- T expert: 通常のTransformer attention。
- K expert: Kimi Delta Attention。長いcontextを効率的に扱うための線形attention系mixerである。
- G expert: Gated Multi-head Latent Attention。圧縮されたlatent表現とoutput gateを使うmixerである。
- top-1 routing: 各chunk位置でrouter確率が最大のexpert一つの出力を採用する方式。
- load-balance補助loss: 一つのexpertへの過度な集中を抑える項。重みは0.01に固定した。

位置数、expert集合、補助loss強度を同時に探索せず、最初のpilotでは4位置、T/K/G、補助loss 0.01だけを試した。モデル初期値、
データ順、学習時乱数はすべて42に固定した。dense T26とMixer-MoEは共通weightを一致させ、Mixer-MoEのT expertにはdense
attentionの初期weightをコピーした。context長は2,048、joint trainingを220 step行い、10 step間隔でFull112を評価した。

## 3. モデル規模と実装上の注意

| 構成 | 保存parameter | top-1で意味的にactiveなparameter |
| --- | ---: | ---: |
| dense T26 | 219,850,496 | 219,850,496 |
| Mixer-MoE、T選択時 | 237,867,436 | 219,859,724 |
| Mixer-MoE、K選択時 | 237,867,436 | 222,873,772 |
| Mixer-MoE、G選択時 | 237,867,436 | 215,979,020 |

保存parameterは全expertをcheckpointに保持するためdenseより約18.0M多い。意味的なactive parameterは、top-1で選ばれた
expert一つだけを数えた値であり、約216--223Mでdense T26に近い。

ただし今回の実装はroutingの正しさを確認するpilotであり、3 expertをすべて計算してから一つの出力を選ぶ
`dense_experts_hard_output_route`である。したがって、現実装のwall-clockやGPU memoryはtop-1 sparse実行の理論的な効率を
表さない。step 101--220の平均はdense T26が8.66秒/step、Mixer-MoEが10.47秒/stepだった。学習全体で観測されたpeak
allocated memoryはdense 24.1 GB、Mixer-MoE 27.9 GBだった。速度とmemoryは境界品質の順位には使っていない。

## 4. 分割評価

Full112は、日本語88文のcategory probeと、活用・派生・複合語などの対応関係を持つ24文のfamily probeからなる。主評価は
H-Netの後段境界であるstage 1と、モデル自身の閾値で境界を選ぶnative条件で行った。

| 指標 | 何を評価するか | 良い方向 |
| --- | --- | --- |
| category precision | 選んだ境界のうち、文節、活用語尾、助詞、複合語境界などとして説明できる割合 | 大きい |
| category coverage | あらかじめ許容した説明可能な境界候補のうち、実際に選べた割合 | 大きい |
| category fracture record occupancy | 保護した語彙内部を不自然に分断した文の割合 | 小さい |
| family precision | 同じ語や構文の文脈・活用variant群で、選んだ境界が説明可能だった割合 | 大きい |
| family coverage | family群で許容境界を回収できた割合 | 大きい |
| landmark coverage | variant間で再利用したい共通境界を選べた割合 | 大きい |
| family integrity | 保護した語彙内部を壊さなかったfamily文の割合 | 大きい |
| 文節 precision / coverage | category probeのうち文節categoryだけのprecision / coverage | 大きい |

precisionは「選んだ境界の正しさ」、coverageは「必要な境界をどれだけ拾ったか」を見る。coverageだけを増やすため境界を過剰に
選ぶとprecisionやfractureが悪化し得るため、両者とintegrityを同時に評価する。

## 5. native境界の時間変化

### 5.1 step 100の早期gate

step 70--100の4時点平均で、Mixer-MoEからdense T26を引いた差は次の通りだった。

| 指標 | 差 |
| --- | ---: |
| category precision / coverage | -.005 / +.020 |
| category fracture | +.020 |
| family precision / coverage | -.007 / +.030 |
| landmark coverage | -.021 |
| family integrity | +.031 |
| 文節 precision / coverage | +.040 / +.065 |

この時点ではcoverage、integrity、文節が改善し、fractureも停止基準を超えて継続悪化していなかった。router collapseもなかった
ため、単一stepの上振れではないかを確認する目的でstep 220へ延長した。

### 5.2 terminal window

step 180、190、200、210、220の5時点平均を示す。差はMixer-MoEからdense T26を引いた値である。

| 指標 | dense T26 | Mixer-MoE | 差 |
| --- | ---: | ---: | ---: |
| category precision | .411 | .371 | -.040 |
| category coverage | .274 | .315 | +.042 |
| category fracture | .307 | .359 | +.052 |
| family precision | .757 | .611 | -.145 |
| family coverage | .527 | .515 | -.012 |
| landmark coverage | .625 | .658 | +.033 |
| family integrity | .958 | .792 | -.167 |
| 文節 precision | .197 | .092 | -.105 |
| 文節 coverage | .287 | .157 | -.130 |

Mixer-MoEのcategory coverage改善は5/5時点で残った。一方、category precisionは5/5、family precisionは5/5、文節
precision/coverageは5/5時点で低下した。特にfamily integrityは全5時点でdenseより`.1667`低く、計画で設定したrelaxed
許容退行幅`-.10`を継続して超えた。

したがって、step 100では有望に見えたcoverageと文節の改善は長期には持続しなかった。ある時点だけでなく形成軌跡を見る必要が
あるという、これまでの実験方針を支持する結果である。

## 6. 固定境界予算profile

nativeではモデルごとに境界数が異なるため、改善が候補境界の順位付けによるのか、単に境界を多く選んだためかを分離する必要が
ある。そこでstep 110/165/220で、両モデルに同じ境界数を強制するprofileを評価した。

- low compression: 2.5 units/chunk。3条件中で境界を多く選ぶ。
- central: 3.0 units/chunk。
- high compression: 3.5 units/chunk。3条件中で境界を少なく選ぶ。
- native: モデル自身の閾値を使うため、境界数はモデル間で一致しない。

次表は3 checkpoint平均のMixer-MoE minus denseである。Fはfractureで、小さい方がよいため負の差が改善を表す。

| 条件 | category P / C / F | family P / C | landmark C | integrity | 文節 P / C |
| --- | --- | --- | ---: | ---: | --- |
| low | -.017 / -.009 / +.019 | +.004 / +.020 | .000 | -.042 | -.044 / -.101 |
| central | -.009 / +.010 / +.015 | -.009 / +.051 | +.069 | .000 | -.050 / -.072 |
| high | +.001 / +.006 / +.004 | .000 / +.030 | +.042 | +.014 | -.070 / -.058 |
| native | +.008 / +.061 / +.015 | -.099 / -.030 | -.042 | -.042 | -.013 / +.014 |

central/highの固定予算ではfamily coverage、landmark coverage、integrityは維持または改善した。このため、native terminalの
family退行を「すべての候補境界の順位付けが悪くなった」とは解釈しない。nativeの1文当たり平均境界数はstep 165でdense
6.49に対しMixer-MoE 8.02、step 220で6.40に対し7.03だった。step 165以降にMixer-MoEがより多く境界を選んだことが、
nativeでのprecision・integrity低下の主要因と考えられる。

一方、文節precision/coverageは境界数を揃えたlow/central/highでも一貫して低下した。したがって文節境界については、境界数の
calibrationだけでなく、候補境界の順位付け自体に弱点がある。

## 7. routingの学習

最大expert占有率は、各層で最も多く選ばれたexpertの割合である。1に近いほど一つへ集中し、本計画では90%超の持続をrouter
collapseとした。routing entropyはrouter確率の不確実性を0--1へ正規化した値で、大きいほど確率が均等に近い。hardな
expert選択数とsoftな確率分布は異なるため、最大占有率が高くてもentropyが高い場合がある。

### 7.1 Full112全体

| step | 層平均の最大expert占有率 | 層別最大 | 平均entropy |
| ---: | ---: | ---: | ---: |
| 110 | .644 | .763 | .948 |
| 165 | .845 | 1.000 | .981 |
| 220 | .543 | .624 | .994 |

step 165では層12のhard選択が一時的に全件Tとなったが、step 220ではT/K/Gへ再分散した。したがって初期に決まったexpertへ
固定されたのではなく、学習中の表現とrouter確率に合わせて選択が揺れ、最終的にはcollapseが解消した。

### 7.2 step 220の層別expert比率

| 置換層 | T | K | G | 最大占有率 |
| ---: | ---: | ---: | ---: | ---: |
| 11 | .362 | .453 | .185 | .453 |
| 12 | .374 | .511 | .114 | .511 |
| 13 | .346 | .071 | .582 | .582 |
| 14 | .624 | .354 | .022 | .624 |

層ごとに異なるexpert分布が形成され、学習可能なmixer選択自体は機能した。ただしroutingの分化は境界品質の改善を保証しない。
今回のfamily integrityと文節の退行は、collapseによる見かけのMoE化ではなく、複数mixerをchunkごとに切り替えることと
境界形成のinteractionによって生じた可能性が高い。

## 8. 結論

現在のT/K/G-4 Mixer-MoEは採用しない。

1. routerはT/K/Gを学習して使い分け、step 220でcollapseしていない。
2. category coverageはterminalの全時点で改善した。
3. しかしT26の強みであるfamily integrityがrelaxed許容幅を全terminal時点で超えて低下した。
4. 文節precision/coverageは固定境界予算でも低下し、単なる境界数の違いでは説明できない。
5. よってK1G1/K3G1への同仕様の横展開、およびMixer-MoEを8位置へ増やすX2は行わない。

この結果はMixer-MoE一般を否定するものではなく、T/K/Gを独立expertとして中盤4層でhard top-1選択する現在の仕様を棄却する
ものである。将来再検討する場合は、Tを常時通るshared baselineとしてK/Gを残差的に加える方式、native境界数のcalibration、
またはfamily consistency objectiveとの組合せを、それぞれ別要因として検証する必要がある。ただし本探索では失敗構成を別要因で
救済せず、P5をここで終了して次の独立要因へ進む。

## 9. 成果物

- 学習run: Drive `runs/mixer_moe_p5_v1/r20_p5_mixer_moe_tkg4_t26_i42_d42_r42_step220_c59db36/`
- 22時点の軌跡・制約・routing: Drive `analysis/mixer_moe_p5_v1_t26_step220/`
- step 110/165/220 profile raw: Drive `evals/mixer_moe_p5_v1/profiles/t26_tkg4/`
- 固定予算profile集計: Drive `analysis/mixer_moe_p5_v1_t26_profiles/`

