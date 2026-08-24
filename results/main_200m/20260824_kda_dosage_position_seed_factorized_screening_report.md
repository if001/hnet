# KDA量・配置位置／seed要因分離 screening レポート

作成日: 2026-08-24

## 結論

K1G1のfamily/landmark/低fractureを維持しながら文節precision/coverageを改善する構成は、今回の
KDA追加量・配置候補からは得られなかった。

- K15-splitはdata-orderを変えた比較では文節改善を再現したが、weight初期値を変えると再現しなかった。
- K16-evenは文節・category改善を比較的頑健に再現した。
- ただしK16-evenは、同じstep 0 weightでdata orderだけを変えた全3条件においてfamily
  precision/coverageをK1G1より低下させ、dense fractureも改善しなかった。
- これはK16-evenの改善がcombined改善ではなく、文節・categoryとfamily/低fractureの間の再現可能な
  trade-offであることを示す。

したがって計画の停止条件に従い、Phase 3の220-step延長は行わない。K1G1を長時間候補のanchorとして
維持する。K16-evenは文節・category specialistとして参考に残すが、K1G1を置き換える候補にはしない。

## 1. 実験条件

計画: `plan/mid/20260823_kda_dosage_position_seed_factorized_screening_plan.md`

- main network: 26層
- dataset、batch、sequence length、optimizer、LR horizon=220、compression targetを固定
- max step: 55
- constraint: `utf8-hard`
- full112: category 88文 + family 24文、low/central/high/native、stage 0/1
- dense: family 24文、step 10/20/30/40/50、native
- 主判定: stage 1
- 速度は順位に使用しない

Phase 1はK1G1と新規5構成、Phase 2はK1G1、K15-split、K16-evenを対象とした。Phase 2では
`model_init_seed`と`data_order_seed`をone-factor-at-a-timeで分離し、`train_runtime_seed=42`を固定した。

## 2. 指標

- precision: 選択された評価可能境界のうち、言語的に説明可能な境界の割合。大きいほど不要な境界が少ない。
- coverage: probeが許容する説明可能境界のうち、実際に選択した割合。大きいほど有用な境界を広く拾う。
- fracture rate: 選択境界のうち語彙単位を不自然に破壊する境界の割合。小さいほどよい。
- family precision/coverage: 活用・派生・文脈variantをまとめた24文family probeでの説明可能性と被覆。
- integrity: family内の語彙単位を壊さず維持した割合。大きいほどよい。
- landmark coverage: family間で共通する重要境界を選択した割合。大きいほどよい。
- 文節 precision/coverage: 文節categoryだけに限定したprecision/coverage。
- dense time-average: step 10--50の5時点平均。
- dense fracture occupancy: 5時点のうちfractureを含むrecordの割合。小さいほどよい。
- late指標: dense後半時点の平均。55 step直前の傾向を見る。

これらは長期main-network性能を直接測る指標ではなく、「言語的に説明可能なchunkは後続networkが再利用
しやすい」という仮説に基づくscreening proxyである。

## 3. Phase 0: 実装・再現性

### 3.1 Parameter数

| 構成 | K/G | total parameters | main-network parameters |
| --- | ---: | ---: | ---: |
| K1G1 | 13/13 | 217,033,864 | 181,249,160 |
| K14-front/middle/late | 14/12 | 218,757,552 | 182,972,848 |
| K15-split | 15/11 | 220,481,240 | 184,696,536 |
| K16-even | 16/10 | 222,204,928 | 186,420,224 |

K14の3構成はparameter数が完全一致し、位置だけを比較できる。KDA量を増やす比較にはK1G1比で最大
2.38%のparameter増加が含まれる。

### 3.2 同一triple反復

K1G1とK14-middleを各2回、同じ `(init=42, data=42, runtime=42)` で1 step実行した。step 0 hash、
shuffle hash、先頭sample列は一致した。step 1はCUDA演算の微小非決定性によりbitwise一致せず、平均
weight絶対差は約`1e-8`、最大差は`6.10e-5`、CE loss差は最大`1.73e-5`だった。この幅を同一triple
noiseとして扱った。

## 4. Phase 1: 構成screening

step 55、central、stage 1の結果を示す。

| 構成 | category P/C/F | family P/C | integrity | landmark C | 文節 P/C | dense P/C/F |
| --- | --- | --- | ---: | ---: | --- | --- |
| K1G1 | .414/.235/.226 | .944/.686 | .917 | .750 | .111/.130 | .479/.485/.125 |
| K14-front | .417/.214/.133 | .708/.487 | .958 | .583 | .120/.130 | .446/.515/.125 |
| K14-middle | .431/.201/.110 | .772/.515 | .958 | .583 | .087/.087 | .503/.552/.133 |
| K14-late | .435/.214/.139 | .869/.651 | .958 | .750 | .080/.087 | .493/.515/.217 |
| K15-split | .468/.282/.234 | .750/.454 | .833 | .583 | .393/.478 | .519/.388/.125 |
| K16-even | .504/.274/.197 | .750/.537 | .958 | .667 | .345/.435 | .650/.461/.258 |

K14群は文節改善がないか、K14-frontのprecision `+.009`という1境界程度の差に留まった。K14-middleは
late dense P/C `.667/.848`、late fracture `.104`と良好だったが、主条件の文節改善を満たさなかった。

K15-splitとK16-evenは文節を大きく改善したためPhase 2へ進めた。ただしこの時点でfamily低下があり、
seed要因分離により改善と退行の頑健性を確認した。

## 5. Phase 2: seed要因分離

baseline `(42,42,42)` に加え、init-43/44とdata-43/44を実行した。data系列はarchitecture別の同一
step 0 checkpointをロードした。

manifest auditにより、init系列ではshuffle hash固定・weight hash変化、data系列ではweight hash完全固定・
shuffle hash変化を確認した。従来の単一seed平均とは異なり、要因は分離できている。

### 5.1 Paired方向一致数

同じseed factor内の候補−K1G1差を数えた。各系列はbaselineを含む3条件である。

| 候補・系列 | category P/C改善 | 文節 P/C改善 | family P/C改善 | dense fracture改善 |
| --- | --- | --- | --- | --- |
| K15-split init | 2/3, 2/3 | 1/3, 1/3 | 2/3, 2/3 | 1/3 |
| K15-split data | 2/3, 2/3 | 3/3, 3/3 | 0/3, 0/3 | 0/3 |
| K16-even init | 2/3, 1/3 | 2/3, 2/3 | 2/3, 1/3 | 2/3 |
| K16-even data | 3/3, 3/3 | 3/3, 3/3 | 0/3, 0/3 | 0/3 |

K15-splitはdata orderに対しては文節改善が頑健だが、weight初期値に対しては1/3であり、構成効果として
十分に再現しなかった。

K16-evenはinit系列で文節P/Cを2/3、data系列で3/3改善した。特にdata系列の平均paired差は次の通り。

| 指標 | K16-even − K1G1 |
| --- | ---: |
| category precision | +.089 |
| category coverage | +.057 |
| 文節 precision | +.247 |
| 文節 coverage | +.304 |
| family precision | -.153 |
| family coverage | -.109 |
| landmark coverage | -.028 |
| dense fracture occupancy | +.131 |
| late fracture occupancy | +.340 |

同じstep 0 weightでdata orderだけを変えても、文節・category改善とfamily・fracture退行がともに残った。

### 5.2 Profile別確認

K16-evenのdata系列では、low/central/high/nativeの全profileで文節precision/coverageが3/3改善した。
一方、family coverage改善はlow 0/3、central 0/3、high 1/3、native 0/3だった。central profileだけの
校正偶然ではなく、構成が境界選択を文節側へ移す一方でfamily landmarkの被覆を失うtrade-offと解釈する。

## 6. 構成仮説に対する回答

1. 単一K3区間の位置だけではcombined改善を作れなかった。K14-front/middle/lateは異なるdense/category
   特徴を示したが、文節改善へ結び付かなかった。
2. KDA量をK15/K16まで増やすと文節境界を拾いやすくなる。特に均等配置K16の効果はdata orderに対して
   頑健だった。
3. しかしKDA量・均等配置だけではK1G1のfamily/landmarkを同時に維持できない。K1-first/K3-firstで
   観測したtrade-offは、均等配置によって解消されなかった。
4. モデルごとのcategoryの強みは完全な偶然ではなく、architectureによる境界選択biasとして再現する。
   ただし「特定層が特定言語categoryを担当する」という因果までは示さない。

## 7. 判断と次の候補

Phase 3へ進めるには、文節改善とfamily/landmark維持が同じ候補・同じseed系列で成立する必要がある。
K15-split、K16-evenはいずれも満たさないため、220-step延長で優劣を決めない。

次にarchitecture探索を再開する場合、KDA blockをさらに増減・並べ替えるだけではなく、K1G1のfamily境界を
保持する機構とK16-evenの文節biasを明示的に両立させる必要がある。例えば、stage/層別の補助境界loss、
複数main-network経路のsoft mixing、または境界router側のmulti-objective制御は別計画として検討できる。
dataset/curriculum変更は本計画では評価していない。

## 8. 制約

- 55 stepの小規模proxyであり、長期main-network lossや生成品質を証明しない。
- validation loaderは既存dense runnerと同じく`validation_max_batches=0`で、validation BPBは取得していない。
  全runの学習完了とloss安定性は確認したが、BPBによる重大退行検査は未実施である。
- full112は有限probeであり、1--2境界の差は昇格理由にしていない。
- runtime seedは42に固定した。学習時確率演算だけの追加分離は、今回のcombined候補がなかったため行わない。

## 9. Artifacts

- 実行memo: `plan/mid/20260823_kda_factorized_screening_execution_memo.md`
- Phase 1 analysis: Drive `reports/kda_factorized_screening/phase1_new6/phase1_analysis.json`
- Phase 2 seed audit: Drive `manifests/kda_factorized_screening/phase2_artifact_seed_audit.json`
- Phase 2 analysis: Drive `reports/kda_factorized_screening/phase2_ofat_analysis.json`
- checkpoints/evaluation raw data: Drive `runs/`、`evals/kda_factorized_screening/`
