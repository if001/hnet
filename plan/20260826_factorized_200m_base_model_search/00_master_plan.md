# 200M長期学習ベースモデルの要因分離探索: master plan

作成日: 2026-08-26

## 1. 目的

約200M規模で長期間学習するH-Netのベースモデル候補を選ぶ。現在は最終性能を証明する段階ではなく、
main networkと境界形成機構の構成を探索する段階である。

小規模学習のloss、BPB、生成文、実装速度は参考値に留め、これまで使ってきたfull112 probeにおける
言語的に説明可能な分割の形成軌跡を主判定にする。明らかなfractureを抑えながら、category、family、
landmark、文節のprecision/coverageとfamily integrityを改善する構成を探す。

## 2. 比較anchor

すべてのworkstreamで、次の3構成を意味の異なるanchorとして扱う。

| 構成 | main network | 役割 |
| --- | --- | --- |
| T26 | Transformer 26層 | KDAを使わないbaseline。family precisionとintegrityが強い |
| K1G1 | KDA 13層 / Gated MLA 13層 | 現時点のcoverage/integrity anchor |
| K3G1 | KDA 19層 / Gated MLA 7層 | main block完全置換・高KDA dosage baseline |

既存step 220結果では、T26はfamily Pとintegrity、K1G1はcategory/family coverage、K3G1はcategory
P/Cに特徴がある。一方、K3G1はfracture、family、landmark、文節で退行が大きかった。このため、
K3G1を単独の採用候補とはせず、高KDA量と長文計算の対照として残す。

次の既存構成は全workstreamで再実行するanchorにはせず、historical referenceとして比較表に残す。

- K14-middle: terminalでcategory precisionと低fractureが強いが、family C/landmarkが後半に低下した。
- K14-late: category coverageとlandmarkを広く拾うが、family integrityの退行が残った。
- K15-split: 文節P/Cが強いが、family C/landmarkが低い文節specialistだった。

新variantは自分のanchor controlに勝つだけでなく、目的軸でこれら既存specialistへ近づいたかも確認する。ただし、
異なるrun条件のhistorical値をpaired差や統計的反復として扱わない。

## 3. これまでの結果から固定する判断原則

1. 単一checkpointの値で選ばない。境界は学習中に形成・崩壊・回復し、構成ごとに形成順序も異なる。
2. 原則step 100までを早期gate、step 200または220までを持続性判定とする。10 step間隔の軌跡を保存する。
3. 同一step比較は「同じoptimizer update数・raw batch列に対する品質」であり、同じ境界成熟度を意味しない。
   同一step、window、matched-coverageの三つを分けて報告する。
4. 順位平均だけで決めない。fractureとintegrityの許容退行を先に適用し、他指標で相殺しない。
5. transitionが小さいだけでは良い分割とは限らない。低品質な境界が固定された可能性を確認する。
6. KDA量だけでなく配置が強く効く。architecture効果を「KDAが多いほど良い」と解釈しない。
7. stage 1を主判定、stage 0を原因診断に使う。UTF-8途中境界率はconstraint監査であり順位に使わない。
8. loss/BPBは学習破綻・重大退行の検出に使うが、小規模値の僅差を候補順位に使わない。
9. full112は有限のproxyである。category別実値、分割gallery、長文での位置別挙動も必ず残す。

## 4. 要因分離の原則

各実験では、対象要因以外を固定する。

- architecture、dataset内容と順序、model/init/runtime seed、context length、raw byte数、optimizer、LR、
  compression target、ratio loss、boundary constraintをmanifestに記録する。
- 新要因は、まず各anchorの無変更controlと一対一で比較する。異なるworkstreamの変更を最初から重ねない。
- 既存artifactをcontrolとして再利用するのは、commit差が数値挙動を変えず、config hash、data manifest、
  seed三因子、LR horizon、raw batch列が一致すると監査できる場合だけとする。それ以外はcontrolを再実行する。
- 新しいlossやwarmup用データをfull112から作らない。評価probeへの漏洩を防ぐ。
- 2つの要因が単独で通過した後だけ2 x 2 factorialで加法性とinteractionを確認する。
- context curriculumはsequence lengthとpackingを変える独立workstreamとし、短contextの構成探索へ混ぜない。

## 5. 優先順位

| 優先度 | workstream | 理由 | 初期対象 |
| ---: | --- | --- | --- |
| P0 | 共通評価・長文評価基盤 | すべての比較の前提。no-op再現とraw-byte監査が必要 | 3 anchors |
| P1 | Encoder複数層の境界特徴融合 | 現在のboundary routerへ最も直接作用し、parameter/compute差が小さい | 3 anchors |
| P2 | Family consistency loss | 現在弱いfamily C/landmark/integrityを直接狙える | 3 anchors |
| P3 | encoder/decoder warmup | 実装負担は小さいが、初期データ消費と追加学習量の対照が必要 | 3 anchors |
| P4 | FFN-MoE | mixerを変えず容量・専門化だけを検証しやすい | T26 pilot後3 anchors |
| P5 | Mixer-MoE | mixer選択と境界形成のinteractionが大きく、実装・解釈リスクが高い | T26 pilot後絞る |
| P6 | 2K→8K→32K context curriculum | KDAの目的を確認する必須段階だが、最も高価で交絡要因が多い | 3 anchors＋短期上位 |

P0の長文probe・memory/time計測の準備はP1と並行してよいが、curriculum学習はP1--P5で候補を絞った後に行う。

## 6. 実行ロードマップ

### Gate 0: baselineとno-op監査

- 3 anchorsについて既存step 220 artifactのhash、seed、data順、22時点full112を再監査する。
- 新実装の全featureをoffにしたconfigで短い再現smokeを行い、旧挙動との差を確認する。
- step軌跡集計、profile評価、category別表、分割galleryを一つのrunnerで再生成できるようにする。
- 長文評価用に位置別boundary drift、遠距離family pair、長文計算量計測のschemaだけ先に固定する。

### Gate 1: P1--P3の単独screening

各要因を別々に、まず`i42/d42/r42`、2K contextで実施する。

1. 3 anchors x control/variantをstep 100まで学習する。
2. step 10--100を10 step間隔でfull112 native評価する。
3. 明確な学習破綻、継続的なfracture悪化、integrity退行があるvariantは停止する。
4. 通過variantだけstep 220へ延長し、step 110--220も10 step間隔で評価する。
5. step 110/165/220でlow/central/high/native profileを保存する。

### Gate 2: P1--P3の再現性とinteraction

- 単独改善がterminal windowに残る要因だけ、`init=43,data=42`と`init=42,data=43`を分離して確認する。
- runtime seedは同一triple反復noiseが問題になった場合だけ増やす。
- P1とP2が両方通過した場合、代表anchorで`fusion on/off x consistency on/off`の2 x 2を実施する。
- warmupは独立効果を確認後、P1/P2 winnerとの組合せを1つだけ検証する。

### Gate 3: P4--P5 MoE

- まずT26でrouting、parameter、active FLOPs、学習安定性を確認する。
- pilotを通過した一つの仕様だけをK1G1/K3G1にも適用する。
- FFN-MoEとMixer-MoEを同時に有効化しない。
- 境界改善がrouter collapseや一部expert独占によるものでないことを利用率で監査する。

### Gate 4: P6 context curriculum

- 3 anchorsと、P1--P5から最大2候補を対象とする。
- fixed 2K、direct long-context、2K→8K→32Kを別runとして比較する。
- sequence length間でstepを直接同一視せず、累積raw bytesとoptimizer updatesを併記する。
- 32Kではfull112だけでなく、長文位置別境界、遠距離family consistency、quality/peak-memory/timeを評価する。

### Gate 5: 長期間学習候補の決定

1構成へ無理に絞らず、次の役割で最大2構成を残してよい。

- quality candidate: fracture/integrity制約を満たし、複数categoryの軌跡が安定している構成。
- long-context efficiency candidate: T26に対する長文品質の許容退行内で、KDAの計算・memory上の利点を示す構成。

## 7. 成果物

| artifact | 保存先 |
| --- | --- |
| master plan | `plan/20260826_factorized_200m_base_model_search/00_master_plan.md` |
| 共通評価仕様 | 同directory `01_shared_protocol.md` |
| 境界・schedule要因 | 同directory `02_boundary_and_schedule_factors.md` |
| MoE要因 | 同directory `03_moe_factors.md` |
| context curriculum | 同directory `04_context_curriculum.md` |
| 一時memo | `plan/mid/` |
| 実験report、CSV | `results/main_200m/` |
| checkpoint/raw eval/manifest | Drive `/content/drive/MyDrive/hnet_agent_200m_main/` |

## 8. 実行規則

- branchは`200m_main`だけを使用する。
- コード修正はローカルで実施し、focused test、commit、push後にColabでpullする。
- GPU学習とGoogle DriveアクセスはColab経由で行う。
- Colab再起動時はclone、Drive mount、Prepare、GPU/import smokeを最初から行う。
- 長い処理はforeground cellで実行し、約10分以内の間隔でstatusまたはcellを確認する。
- 既存artifact、checkpoint、ユーザー所有の変更を上書きしない。
