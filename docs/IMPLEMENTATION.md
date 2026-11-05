# RT-DETRv2 OWOD 実装メモ（Part I / Week 1–3 範囲）

計画書: `docs/RT-DETRv2_OWOD.md`

## 環境

```bash
cd /data1/AnoDet/owod
uv sync                       # .venv (Python 3.11, torch 2.8.0+cu128)
```

- `third_party/RT-DETR` … lyuwenyu/RT-DETR（commit df2a96c, **無改変**。`owod/` からサブクラスで拡張）
- `third_party/PROB` … M-OWODB split ファイルと評価コードの参照元

## データ準備（済）

```bash
uv run python -m scripts.prepare_mowodb      # -> data/mowodb/{t1_train,test}.json
```

| ファイル | 画像 | アノテーション | 備考 |
|---|---:|---:|---|
| `t1_train.json` | 16,551 | 47,223 | VOC07+12 trainval（既知20クラスのみ）。`info.split=train` |
| `test.json` | 10,246 | 61,707 | VOC07 test 4,952 + COCO val2017 5,294。全80クラス + `difficult` + `bbox_voc` |

- Task 1 の未知 GT（difficult 除く）= **23,320**。ORE/PROB の M-OWODB T1 の値と一致。
- VOC07 test の `difficult` フラグは既存の変換済 JSON に無かったため、VOC2007 test の XML を
  `data/raw/voc07test/` に取得して使用。COCO 画像は ORE の `coco2voc.py` と同じ規則（全 difficult=0、crowd 含む）。
- **未知 GT 漏洩ガード**: `OWODDetection(split='train')` は (a) `info.split != 'train'` のファイルを拒否、
  (b) `source=='gt'` かつ `category_id >= 20` の注釈があれば `UnknownLeakError`。構築時に全件、`__getitem__` で毎回検査。

## 構成要素の実装対応

| 計画 | 実装 | 設定キー |
|---|---|---|
| C1 objectness ヘッド | `owod/zoo/ow_decoder.py` 各デコーダ層の query 埋め込みに Gaussian objectness、および学習される「既知以外」logit（`dec_unk_head`, `OWCriterion.loss_unk`） | `OWRTDETRTransformerv2.use_objectness/use_unk_logit`, `loss_unk` 重み |
| C2 確率的 objectness 損失 | `owod/zoo/gaussian_objectness.py`（μ,Σ は buffer, EMA 更新）、`OWCriterion.loss_obj`。E2〜E4 は PROB と同じく全クエリ・対角・ε=1e-5・毎イテレーション更新（`docs/E2_SATURATION.md`） | `OWCriterion.obj_stats`, `obj_cov_type/obj_momentum/obj_shrinkage/obj_interval/obj_temperature`, `loss_obj` 重み |
| C3 背景損失マスキング | `OWCriterion._bg_ignore_mask`（未マッチ中 objectness 上位 k_bg を VFL から除外。enc 出力にも適用） | `OWCriterion.bg_ignore_topk` |
| C4 objectness query selection | `OWRTDETRTransformerv2._get_decoder_input`（enc 用 Gaussian の d² 最小 top-300） | `query_select_method: objectness` |
| C6 CDN 除外 | targets に `cdn_mask` があれば該当 GT を CDN から除外（Part II / C5 用のフック） | – |
| 推論スコア | `owod/zoo/ow_postprocessor.py`：known `p(c)`（`cls`, E2〜E4 の既定）/ `o·p(c)`（`obj_cls`）、unknown `learned`（o·σ(u), 既定）/ `c1`（当初式 o·(1−max p)）/ `maha/msp/energy`、top-100 | `OWPostProcessor.unk_score/known_score` |
| 評価 | `owod/eval/owod_eval.py`：PROB `open_world_eval.py` の移植 | `owod_eval.gt_json` |

評価器は `tests/test_owod_eval_vs_prob.py` で PROB 原実装と Known AP50 / U-Recall / WI / A-OSE / U-AP50 が一致することを確認済み。
さらに PROB 公式 Task 1 重みの検出結果で、PROB 公式評価と Known AP50 / U-Recall / A-OSE が一致することを確認済み（重複テスト画像の扱い等を修正、`docs/EVAL_VALIDATION.md`）。

## 実験の実行

```bash
# 学習（GPU 1枚、AMP + EMA、50 ep、5 ep ごとに OWOD 評価）
CUDA_VISIBLE_DEVICES=0 uv run python tools/train.py -c configs/e0_closed.yml      # E0
CUDA_VISIBLE_DEVICES=0 uv run python tools/train.py -c configs/e2_obj.yml         # E2: C1+C2
CUDA_VISIBLE_DEVICES=0 uv run python tools/train.py -c configs/e3_obj_bgmask.yml  # E3: +C3
CUDA_VISIBLE_DEVICES=0 uv run python tools/train.py -c configs/e4_m0.yml          # E4 = M0: +C4

# E1（学習不要）: E0 に post-hoc で Gaussian をフィットし MSP / Energy / Mahalanobis / C1 を一括評価
uv run python tools/e1_posthoc.py -c configs/e0_closed.yml -r outputs/e0_closed/best.pth --fit-gaussian
# E-score（学習不要）: E4 の未知スコア関数比較
uv run python tools/e1_posthoc.py -c configs/e4_m0.yml -r outputs/e4_m0/best.pth

# 評価のみ
uv run python tools/train.py -c configs/e4_m0.yml --test-only -r outputs/e4_m0/best.pth
```

`-u key=value` で YAML を上書き可能（例: `-u train_dataloader.total_batch_size=8 OWCriterion.bg_ignore_topk=10`）。

## 計画書からの逸脱・解釈（要確認）

1. **objectness の温度**: 計画の `o(q)=exp(-½ d²)` は D=256 で d²≈χ²(256)（平均256）となり全クエリで 0 にアンダーフローする。
   PROB と同じ `o(q)=exp(-τ d²/D)`（τ=1.3）を採用。損失も `d²/D` に正規化し、λ_obj=0.3 は PROB の 8e-4（生 d²）とほぼ同スケール。
   PROB と同様に `d²/D ≥ -log 0.9` でクランプ。
2. **Σ の shrinkage**: 計画式 `Σ ← (1−m)Σ + m·cov + λI` は毎回 λI が加算され床が λ/m に漂流するため、λI を cov 側に入れて EMA（床は常に λ）。
3. **Gaussian 初期化**: 最初の推定は matched 埋め込みが 4·D=1024 個たまった時点（それ以前は μ=0, Σ=I）。以降 50 iter ごと。
4. **飽和判定（§9.2）**: Σ を matched 埋め込み自身から推定するため matched の d²/D は常に ≈1、o(q) は最大でも ≈exp(−1.3)≈0.27。
   「全クエリで o(q)>0.9」は原理的に起きないので、**評価時の全クエリのうち d²/D≤1（前景シェル内）の割合** を飽和指標として記録（→1.0 で飽和）。
5. **C4 のエンコーダ objectness**: 計画の「1層 MLP」は採らず、enc_output（cls/box ヘッドと共有の特徴）上に直接 Gaussian を置いた。
   正例のみの損失で独立した射影を学習すると全トークンを μ に潰す自明解があるため。必要なら射影版を追加する。
6. **バックボーン**: ImageNet 事前学習（PaddlePaddle ResNet50-vd SSLD）のみ。**COCO 事前学習済み RT-DETR 重みは未知クラスを見ているため使用禁止**。
7. **スケジュール**: 50 ep、強い augmentation は最後の 2 ep で停止（上流 72 ep 設定の 71 に相当）。
8. **閾値調整用の検証セット**は未作成（計画の「検証セットで決定」）。test を使うとリークになるため、別途 split を決める必要あり。

## 未実装（次段）

- FPS 計測（TensorRT FP16）。推論時の Cholesky は deploy 時に whitening 行列を前計算する必要あり
- C5 (SAM2 疑似未知ラベル) フォールバック
- AURC 指標
- Part II（Stage A–E ループ、語彙管理、DAM/Qwen 連携）
- W&B / Hydra（現状は RT-DETR の YAML + TensorBoard + `log.txt`）
