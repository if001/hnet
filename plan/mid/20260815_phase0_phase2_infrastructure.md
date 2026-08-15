# Phase 0・Phase 2比較基盤メモ

作成日: 2026-08-15

## 中規模snapshot

Phase 2には既存の次のsnapshotを使う。

- train: `/content/drive/MyDrive/hnet_agent/datasets/screening/ja8_en1_code1_0p5b_ctx2k`
- 100,000 documents、518,981,601 raw bytes、8 packed shards。
- document比率: Japanese web 0.68、Japanese Wikipedia 0.12、English web 0.08、
  English Wikipedia 0.02、code 0.10。
- validation: `/content/drive/MyDrive/hnet_agent/datasets/validation/ja8_en1_code1_12m_ctx2k`
- Phase 2の共通学習予算は500,000,000 raw bytesとする。
- 125,000,000 raw bytesごとにvalidationとcheckpointを実行し、4点のcurveを得る。

step数ではなくraw byte累積量で停止する。128k tokenizerではtokenごとの元byte長を
packed manifestへ保存し、同じBPB定義、raw-byte予算、checkpoint位置を使う。
1 optimizer stepぶんのovershootは記録し、比較時は各checkpointの
`cumulative_input_bytes`を用いる。

## 95–100M構成のparameter内訳

Colab A100環境で実モデルをinstantiateして集計した。
JSONは
`/content/drive/MyDrive/hnet_agent_200m_main/configs/phase0/parameter_accounting.json`
へ保存した。

| 構成 | 総parameter | main network | embedding/head等 |
|---|---:|---:|---:|
| H-Net T26 | 100,071,728 | 83,520,000 | hierarchy 15,765,296、router 589,824、embedding+head 196,608 |
| H-Net K1G1 x12 | 95,059,600 | 78,507,872 | hierarchy 15,765,296、router 589,824、embedding+head 196,608 |
| H-Net K1T1 x12 | 97,709,200 | 81,157,472 | hierarchy 15,765,296、router 589,824、embedding+head 196,608 |
| tokenizer T10、128k vocab | 97,659,392 | 32,123,392 | shared embedding/head 65,536,000 |

全構成を約95–100Mへ収めた。tokenizer baselineでは67.1%がshared embedding/headを
占める。この差はparameter-matched条件でtokenizer方式が負担する実コストとして扱う。

## tokenizer作成規則

- training splitのbyte-packed文書だけから128k ByteLevel BPEを学習する。
- validationをtokenizer学習へ混ぜない。
- BOS/EOS/PAD/UNKを明示し、同一文書を`uint32` token IDへ変換する。
- trainとvalidationを同じtokenizerでpackする。
- tokenizerとtokenized snapshotは
  `/content/drive/MyDrive/hnet_agent_200m_main/tokenizers/128k`および
  `data/tokenized/`へ保存する。

### Colab作成・監査結果

- actual vocab: 128,000、tokenizer SHA256:
  `9e003d8ccfdba4bf0ce2c225a85c7614ef031c77034061a34a430e95c9befc75`
- train: 100,000 documents、99,475,679 tokens、518,781,601 raw bytes、
  48,546 samples at 2,048 tokens。
- validation: 3,103 documents、1,925,680 tokens、10,976,867 raw bytes、
  938 samples at 2,048 tokens。
- train全体は特殊token込みで5.215 raw bytes/token。
- 256文書を再encodeし、token byte-length lookupの合計と元UTF-8 byte長が
  全件一致した。
- 詳細はDriveの`tokenizers/128k/audit.json`へ保存した。

raw bytes/updateを約262kへ揃える初期値として、H-Netはbatch 4・grad accumulation
32、tokenizerはbatch 4・grad accumulation 6を使う。最終値は1-step memory
calibration後に固定する。

## 実装と検証

- `train.py`とtrainerへ`max_train_bytes`、byte間隔のsave/validation、
  cumulative raw-byte logger、resume時のbyte位置復元を追加した。
- `scripts/run_phase2_screening.py`でbranch、data manifest、commit、実行commandを
  run manifestへ固定する。
- Colabで関連unit test 12件を実行し、全件passした。

## 開始前に残る項目

- Phase 1の3 seed結果からT26以外の対抗候補を1構成に絞る。
- T26、対抗候補、tokenizerの短いmemory calibrationを行い、raw bytes/updateが
  近くなるようbatch sizeとgradient accumulationを固定する。
- tokenizer作成後にactual vocab、bytes/token、coverage、replacement/parse失敗を監査する。
