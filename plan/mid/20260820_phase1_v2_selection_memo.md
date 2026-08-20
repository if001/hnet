# 拡張日本語probe Phase 1選定メモ

作成日: 2026-08-20

## 条件

- probe: v2、88文章、11 category各8件
- checkpoint: seed 42、step 55
- 主比較: `central`固定境界予算、stage 1
- UTF-8途中境界率と速度は順位に使用しない
- 旧short fragmentationは記述値とし、単独の除外条件にしない

## 全体結果

| model | precision | coverage | best segmentation F1 | lexeme fracture rate | short records | severe records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K3T1 | 0.471 | 0.239 | 0.465 | 0.218 | 0/88 | 0/88 |
| T26 | 0.443 | 0.248 | 0.425 | 0.191 | 0/88 | 0/88 |
| K1G1 | 0.427 | 0.239 | 0.464 | 0.206 | 2/88 | 0/88 |
| K1T1 | 0.416 | 0.222 | 0.438 | 0.240 | 0/88 | 0/88 |
| K3G1 | 0.414 | 0.256 | 0.383 | 0.359 | 4/88 | 0/88 |
| M3T1 | 0.293 | 0.197 | 0.303 | 0.478 | 7/88 | 2/88 |

## 判断

Phase 2へT26、K1G1、K3G1、K3T1を進める。

- K3T1はprecisionとbest segmentation F1が最大級で、severe fragmentationがない。
- T26はlexeme fractureが最少で、precision/coverageも上位。
- K1G1は活用・補助動詞・助詞のF1が高く、完全分割F1がK3T1と同等。
- K3G1はlexeme fractureが多いがcoverageが最大で、compound、proper noun、bunsetsuの
  coverageが高い。severe fragmentationは0であり、step 55だけでは除外しない。
- K1T1は重大な異常はないが、K3T1などにprecision、coverage、F1、lexeme fractureで
  概ねPareto支配される。
- M3T1はprecision、coverage、F1が最低で、lexeme fractureとsevere fragmentationが最大。

K3G1以外のPhase 2候補は既にseed 42/43/44、step 55/110/165/220のcheckpointがある。
K3G1はnative圧縮率を再キャリブレーション後、同じ3 seed・220 stepを追加学習する。

## 注意

v2の`protected_substrings`も人手annotationであり、lexeme fractureは絶対的なgoldではない。
特に複合語は複数粒度が妥当なため、数値とblind galleryを併用する。Phase 2では、K3G1の
lexeme fractureが学習量とともに減るか、compound coverageを維持するかを主要確認事項とする。

