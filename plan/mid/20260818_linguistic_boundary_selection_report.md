# 言語的境界によるmain network中規模選定レポート

作成日: 2026-08-18

## 1. 結論

長期間学習へ進める第一候補を **T26** とする。K1G1を第二候補として残す。

この判断は速度差ではなく、共通境界予算下のstage 1分割について、3 seed・4 checkpointで
確認した説明可能性、病的断片化、seed安定性に基づく。T26とK1G1のvalidation BPBは
実用同等であり、BPB差を決定理由にはしない。

本選定は長期性能の証明ではない。「明らかに不自然な断片化が少なく、言語的に説明可能な
分割を安定して形成する構成は、長期学習でmain networkが利用しやすい可能性が高い」という
計画上の仮説に基づく候補選抜である。

## 2. 実験条件

- 候補: T26、K1G1、K3T1
- seed: 42、43、44
- checkpoint: step 55、110、165、220
- 学習量: step 220で230,686,720 raw bytes
- probe: 人手設計した日本語24文章
- constraint: `utf8-hard`
- 主比較: `central`共通境界予算
  - stage 0: 3.0 byte/chunk
  - stage 1: 3.0 stage-0 unit/chunk
- 主対象: stage 1
- UTF-8途中境界率と実行速度は順位に使用しない

## 3. 3 seed集約結果

### 3.1 Step 220

| model | precision平均 | seed範囲 | coverage平均 | seed範囲 | pathological平均 | seed範囲 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T26 | 0.478 | 0.444–0.500 | 0.282 | 0.250–0.319 | 0.000 | 0.000–0.000 |
| K1G1 | 0.465 | 0.383–0.560 | 0.301 | 0.250–0.389 | 0.028 | 0.000–0.042 |
| K3T1 | 0.427 | 0.351–0.500 | 0.269 | 0.181–0.333 | 0.028 | 0.000–0.042 |

T26はcoverageではK1G1より低いが、precisionのseed範囲が狭く、step 220の全seedで
病的断片化がなかった。K1G1はcoverage最大だが、seed間変動と短断片が残った。

### 3.2 Checkpoint推移

| model | step | precision平均 | coverage平均 | pathological平均 |
| --- | ---: | ---: | ---: | ---: |
| T26 | 55 | 0.485 | 0.296 | 0.028 |
| T26 | 110 | 0.501 | 0.282 | 0.014 |
| T26 | 165 | 0.503 | 0.287 | 0.000 |
| T26 | 220 | 0.478 | 0.282 | 0.000 |
| K1G1 | 55 | 0.434 | 0.269 | 0.014 |
| K1G1 | 110 | 0.459 | 0.310 | 0.028 |
| K1G1 | 165 | 0.435 | 0.287 | 0.014 |
| K1G1 | 220 | 0.465 | 0.301 | 0.028 |
| K3T1 | 55 | 0.457 | 0.269 | 0.014 |
| K3T1 | 110 | 0.406 | 0.231 | 0.014 |
| K3T1 | 165 | 0.433 | 0.264 | 0.014 |
| K3T1 | 220 | 0.427 | 0.269 | 0.028 |

T26はstep 165以降で病的断片化が消えた。K1G1とK3T1では、頻度は小さいものの
特定seedでstep 220まで残った。precision/coverageはいずれも単調改善ではなく、
小規模学習内で境界順位が継続的に変化している。

## 4. Category別所見

step 220の3 seedを合算すると、K1G1は補助動詞、活用、複合語、助詞、structuredで
T26より高いcoverageを示した。

| category | T26 P/C | K1G1 P/C | K3T1 P/C |
| --- | ---: | ---: | ---: |
| auxiliary | 0.333 / 0.083 | 1.000 / 0.333 | 0.667 / 0.167 |
| inflection | 0.250 / 0.111 | 0.444 / 0.222 | 0.364 / 0.222 |
| particle | 0.750 / 0.333 | 0.692 / 0.500 | 0.727 / 0.444 |
| compound | 0.571 / 0.333 | 0.640 / 0.444 | 0.600 / 0.417 |
| bunsetsu | 0.500 / 0.429 | 0.190 / 0.190 | 0.400 / 0.381 |
| structured | 0.429 / 0.167 | 0.471 / 0.222 | 0.286 / 0.111 |

K1G1には`変更/していない`、`書/いている`、`分割/の結果/は人/が確認`など、
仮説に合う分割が多い。一方、同じstep 220でもseedによって`[値][が][正]`、
`[処][理][が完]`のような連続短断片が出た。

T26にも`笑/っ/ている`や単漢字chunkなどの説明困難な例はあり、分割が常に自然という
意味ではない。ただし、step 220では3 seedすべてで3連続の短断片はなく、aggregate
precisionとseed安定性が候補中で最も良かった。

K3T1はseed 42では比較的文節的なchunkを形成したが、seed 44では`書/い/ている`、
`してい/ない`、`文/章`などの細分化が増えた。seed再現性を考えると第一候補にはしにくい。

## 5. 文脈pair

`笑っている`の通常用法と文字列用法を変えるdynamic pairは、全model・全seed・全checkpointで
厳格なpass条件を満たさなかった。異なるsignatureになる例はあったが、説明困難境界を伴うか、
期待する許容位置を捉えなかった。

control pairのpass数も学習量とともに単調には増えなかった。step 220では各modelとも
6 control判定中3–4件のpassに留まった。このprobeでは、文脈適応性をmodel選定の肯定材料に
できない。これは全候補共通の未解決事項として長期評価へ引き継ぐ。

## 6. BPBとnative圧縮率

| model | step 220 BPB平均 | seed範囲 | native L2/L0平均 |
| --- | ---: | ---: | ---: |
| T26 | 1.6767 | 1.6721–1.6816 | 9.55 |
| K1G1 | 1.6773 | 1.6714–1.6823 | 9.43 |
| K3T1 | 1.6840 | 1.6800–1.6874 | 11.20 |

T26とK1G1のBPB差は0.0006であり、優劣を主張できる幅ではない。K3T1も重大な退行では
ないが、native圧縮率が他候補より高い。言語評価では共通境界予算を強制しているため、
native圧縮率の差をprecision順位へ直接混ぜていない。

## 7. 選定理由

T26を第一候補とする理由は次のとおりである。

1. step 220のaggregate precisionが3候補中で最大。
2. precisionとcoverageのseed範囲がK1G1、K3T1より狭い。
3. step 165と220で全seedの病的断片化が0。
4. 文節、助詞、複合語、数値単位など複数categoryを捉え、単一categoryだけへの偏りではない。
5. K1G1とのBPB差は実用同等で、速度差を使わず境界proxyで選べる。

ただしK1G1は補助動詞、活用語尾、助詞、複合語のcoverageでT26を上回る。長期学習予算が
2構成分ある場合は、T26とK1G1の並行比較が最も情報価値が高い。1構成だけなら、現時点では
T26を選ぶ。

## 8. Artifact

- raw評価: Drive `hnet_agent_200m_main/evals/linguistic_boundary_phase2/`
- 集約: Drive `hnet_agent_200m_main/reports/linguistic_boundary_selection/phase2_three_seed_trajectory/`
- 学習status: Drive `hnet_agent_200m_main/manifests/linguistic_boundary_phase2_k3t1_training_status.json`
- 評価status: Drive `hnet_agent_200m_main/manifests/linguistic_boundary_phase2_evaluation_status.json`

