<p align="center">
  <img src="figs/logo.png" alt="UniQ-DETR" width="600">
</p>


## Highlights

This is the official implementation of UniQ-DETR: **Real-Time Open-World Object Detection with Unique Queries for Unknown Objects**

- **NMS Free：** 推論は元の RT-DETRv2 と同じく topk だけです。NMS も重複除去も使いません。
- **Unique Queries：** クラスによらない 1 対 1 ヘッド r を持ちます。r は、物体ごとに代表のクエリを 1 個だけ選ぶように学習するので、未知物体の重複も NMS なしで抑えられます。
- **Self-Supervised Pseudo-Labels：** 学習初期の自分自身を教師にして、未知物体の疑似ラベルを作ります。外部のモデル（SAM など）は使いません。
- **Performance：** M-OWODB Task 1 で **U-Recall 28.69**（Known AP50 63.53）、**約 107 FPS** です（RTX 4090、fp32、バッチ 1）。U-Recall は、NMS ありの比較手法も含めて最高です（OW-DETR-RT は NMS 0.5 で 27.83）。

## Records

M-OWODB Task 1 の値です（report 側、推論はすべて topk のみ）。詳細は `analysis/G4_PLAN.md` にあります。

| 日付 | 段階 | 変更点 | Known AP50 | U-Recall |
|---|---|---|---:|---:|
| 2026-10-28 | 案 G3 | 1 対 1 ヘッド r ＋ OW-DETR の疑似ラベル（300 クエリ） | 62.61 | 23.92 |
| 2026-11-02 | G4-1 | 疑似ラベルを学習初期の自分自身を教師にしたものに置き換え | 62.17 | 25.99 |
| 2026-11-04 | G4-4 | 600 クエリで学習 | 62.87 | 27.24 |
| 2026-11-05 | **UniQ-DETR** | 900 クエリで推論、未知スコアに箱の面積の項を追加 | **63.53** | **28.69** |

## Results (WIP)

評価の条件：
- 1 画像 top-100 です。
- テスト画像を名前の偶奇で 2 分割します。片方（tune 側）で推論設定を選び、もう片方（report 側）の値を示します。
- 選ぶ基準は「Known AP50 ≥ 61.51 のもとで U-Recall 最大」です。
- 比較手法は、同じ RT-DETRv2・同じ学習レシピに移植したもの（`-RT`）です。値はそれぞれの最良の後処理でのものです（`analysis/G3_SUMMARY.md`）。

| 手法 | 後処理 | Known AP50 | U-Recall | WI ↓ | A-OSE ↓ | FPS（fp32） |
|---|---|---:|---:|---:|---:|---:|
| **UniQ-DETR** | **topk のみ** | 63.53 | **28.69** | 0.047 | 2,555 | 107 |
| OW-DETR-RT | NMS 0.5 | 62.30 | 27.83 | 0.055 | 7,295 | 95〜98 |
| CAT-RT | NMS 0.7 | 62.50 | 26.28 | 0.057 | 11,327 | 70 |
| PROB-RT | NMS 0.7 | 62.26 | 24.66 | 0.070 | 4,812 | 93〜95 |
| RandBox-RT | NMS 0.5 | 61.85 | 24.03 | 0.036 | 5,336 | 13.5※ |
| OrthogonalDet-RT | NMS 0.5 | **65.22** | 19.45 | **0.029** | **1,993** | 12.0※ |
| OW-DETR-RT | topk のみ | 62.52 | 23.74 | 0.055 | 3,165 | 109〜114 |
| PROB-RT | topk のみ | 61.57 | 21.55 | 0.055 | 2,071 | 104〜109 |
| CAT-RT | topk のみ | 61.92 | 15.11 | 0.048 | 2,168 | 91 |

※ RandBox 系は元の推論（格子状の箱 10,000 個、NMS 0.6）での値です。

注意点：
- **w の選び方：** UniQ-DETR の w 8 は、Known AP50 に余裕を持たせて選んだ値です。tune 側の基準どおりに選ぶと w 12 になり、Known 61.94・U-Recall 29.16 です。
- **面積の項の公平性：** 未知スコアの面積の項を、比較手法にも掛けて確かめました。U-Recall の変化は −0.8〜+1.0pt で、最高が UniQ-DETR であることは変わりません。ただし、OrthogonalDet-RT の Known AP50 は 65.50 に上がります（`analysis/G4_PLAN.md` §6.6）。
- **未達の目標：** Known AP50 は、OrthogonalDet-RT に及びません。

## Methods

### Base model

| 検出器 | バックボーン | 状態 |
|---|---|---|
| RT-DETRv2 | R50 | ✅ (WIP) |
| RT-DETRv2 | R18 | 予定 |
| RT-DETRv2 | R101 | 予定 |
| RT-DETRv4 | R18 | 予定 |
| RT-DETRv4 | R50 | 予定 |
| RT-DETRv4 | R101 | 予定 |
| RF-DETR | Small | 予定 |
| RF-DETR | Medium | 予定 |
| RF-DETR | Large | 予定 |
| DEIMv2 | S | 予定 |
| DEIMv2 | M | 予定 |
| DEIMv2 | L | 予定 |

現在の構成は RT-DETRv2（PResNet-50、640×640、デコーダ 6 層）です。そこに次の要素を足しています。

### 1. 4 つの軽いヘッド（デコーダの各層、各 Linear(256→1) 程度）

| ヘッド | 意味 | 学習 |
|---|---|---|
| 物体らしさ o | 全クエリの分布からの距離（BatchNorm 後の二乗ノルム、PROB と同じ） | 既知 GT に対応付いたクエリの d² を小さくする |
| 既知以外 u | 既知クラスに対応付かないクエリか | 未マッチのクエリ = 1、対応付いたクエリ = 0（focal） |
| 位置の当てはまり具合 q | 箱が物体にぴったり合っているか（クラス非依存の IoU 予測） | 既知 GT と IoU ≥ 0.3 のクエリだけ、IoU を目標に学習する |
| **1 対 1 ヘッド r** | そのクエリが物体を代表する 1 個か（unique query） | 既知 GT に対応付いたクエリ = 1、同じ GT と IoU ≥ 0.5 の他のクエリ = 0、背景は損失なし |

- **r がこの手法の中心です。**
  - DETR の既知クラスが NMS なしで重複しないのは、1 対 1 マッチングのおかげです。
  - r はこの性質をクラスによらない形で学ぶので、未知物体の重複も抑えられます。

### 2. 学習初期の自分自身を教師にした疑似ラベル（学習前にオフラインで一度だけ作る）

- **教師：** 同じ構成の検出器を 2 エポックだけ学習した重みを使います。学習初期のモデルは、未知物体をまだ背景として抑え込んでいないためです。
- **作り方：**
  1. 学習画像ごとに、教師の予測を元画像と左右反転画像で得ます。
  2. 既知 GT と重ならない箱を候補にします。
  3. 反転一致度（反転画像の予測との最大 IoU）で順位を付け、上位 5 個を unknown の疑似ラベルにします。
- **学習での使い方：** 疑似ラベル 1 個につき、最も重なる未マッチのクエリ 1 個を unknown クラスの正例にします。使い始めるのはエポック 9 からです。

### 3. 600 クエリで学習し、推論は 900 クエリ

- クエリを増やすと、未知物体を覆う候補が増えます。デコーダ候補の網羅率は、300 クエリの 47% から、900 クエリでは 65% になります。
- 重複は r が抑えるので、topk のままでも U-Recall が上がります。速度はほぼ変わりません。

### 4. 推論のスコア

- **既知の 20 列：** p(c)
- **未知の 1 列：** w · o · σ(u) · q · σ(r)^0.5 · (箱の面積)^0.25
  - 面積は 0〜1 に正規化した値で、w = 8 です。
- スコア行列 [900 × 21] 全体から、上位 100 件を出力します。
- 追加の計算は、要素ごとの積だけです。

面積の項は、r で重複を抑えた後に残る誤検出（小さな背景の断片）を押し下げます。この項で、U-Recall は 27.50 から 28.69〜29.16 に上がります。

## Getting Started

### Installation

```bash
uv sync          # .venv（Python 3.11, torch 2.8.0+cu128）
```

- `third_party/RT-DETR` は、lyuwenyu/RT-DETR を無改変で使っています。拡張はすべて `owod/` のサブクラスです。
- バックボーンは、ImageNet 事前学習の重みだけを使います。COCO 事前学習の RT-DETR 重みは、未知クラスを見ているので使いません。

### Training and Evaluation

GPU は `CUDA_VISIBLE_DEVICES` で指定します。学習は GPU 1 枚で、50 エポックに約 8 時間かかります。

```bash
# 1. データ（M-OWODB Task 1）-> data/mowodb/{t1_train,test}.json
uv run python -m scripts.prepare_mowodb

# 2. 疑似ラベルの教師：E2 と同じ構成を 2 エポックだけ学習（outputs/e2_pilot_bn/last.pth）
uv run python tools/train.py -c configs/e2_obj.yml -u epoches=2 --output-dir outputs/e2_pilot_bn

# 3. 疑似ラベル（反転一致度の上位 5 個 / 画像）-> data/pseudo_unk/e2ep2_cons_k5.json
uv run python tools/make_pseudo_unk.py -c configs/e2_obj.yml -r outputs/e2_pilot_bn/last.pth \
    --ann data/mowodb/t1_train.json --out-prefix data/pseudo_unk/e2ep2_cons --ks 5

# 4. 学習（600 クエリ）-> outputs/g4_4/best.pth
uv run python tools/train.py -c configs/g4_4.yml

# 5. 評価（900 クエリ、tune / report の 2 分割、report 側の値を表示）
uv run python tools/sweep_pp.py -c configs/uniq_detr.yml -r outputs/g4_4/best.pth --grid unk_weight=8,12

# 6. 速度（バッチ 1、モデル＋後処理、fp32 と AMP）
uv run python tools/bench_fps.py -c configs/uniq_detr.yml -r outputs/g4_4/best.pth
```

- `scripts/run_g4_4.sh` は、手順 4 とその後の推論設定の掃引・分析をまとめて行います。
- `-u key=value` で、YAML の設定を上書きできます。
- テスト全体で評価するだけなら、`tools/train.py -c configs/uniq_detr.yml --test-only -r outputs/g4_4/best.pth` です。2 分割しないので、上の表の値とは少し違います。

### Configs

`configs/uniq_detr.yml`（推論）
→ `g4_4.yml`（600 クエリ）
→ `g4_1.yml`（疑似ラベル）
→ `m0_primary_owdetr.yml`（r、unknown クラス）
→ `m0_owdetr.yml` → … → `include/rtdetrv2_r50_ow.yml`

| 段階 | 設定 | 内容 |
|---|---|---|
| M₀ | `configs/f1_quality_50ep.yml` | o・u・q のヘッド |
| 案 G | `configs/m0_primary.yml` | + 1 対 1 ヘッド r |
| 案 G3 | `configs/m0_primary_owdetr.yml` | + unknown クラスと OW-DETR の学習信号 |
| G4-1 | `configs/g4_1.yml` | 疑似ラベルを、学習初期の自分自身を教師にしたものに置き換え |
| G4-4 | `configs/g4_4.yml` | 600 クエリで学習（UniQ-DETR の学習設定） |
| **UniQ-DETR** | `configs/uniq_detr.yml` | G4-4 の重みを 900 クエリで推論、面積の項を含む未知スコア |

## Project Structure

| パス | 中身 |
|---|---|
| `owod/zoo/ow_decoder.py` | デコーダとヘッド（o・u・q・r）、疑似ラベルの DN 除外 |
| `owod/zoo/ow_criterion.py` | 損失（`loss_primary` = r、`_ext_pseudo` = 疑似ラベルの割り当て） |
| `owod/zoo/ow_postprocessor.py` | 推論のスコア（`unk_score: learned_qr`、`unk_r_gamma`、`unk_q_exp`、`unk_area_exp`）と topk |
| `owod/eval/owod_eval.py` | 評価器。PROB の `open_world_eval.py` の移植で、PROB 公式重みで論文値を再現済み（`docs/EVAL_VALIDATION.md`） |
| `tools/` | 学習、疑似ラベル作成（`make_pseudo_unk.py`）、推論設定の掃引（`sweep_pp.py`）、分析（`diag_unk_*.py`）、速度（`bench_fps.py`） |
| `configs/baselines/` | 比較手法の RT-DETRv2 への移植（OW-DETR、CAT、RandBox、OrthogonalDet）。PROB は `configs/b_prob_rt.yml` |
| `analysis/` | 最新の実験の記録。`G4_PLAN.md` は UniQ-DETR に至る経緯と全結果、`G3_SUMMARY.md` は前段のまとめ |
| `docs/` | それ以前の記録（実装メモ、比較手法の移植、評価器の検証） |
| `figs/` | ロゴなどの図 |

## TODO

- [ ] **Known AP50 の向上：** OrthogonalDet-RT の 65.22 にはまだ届いていません。閉集合でも 64.06 なので、検出器そのものを強くする必要があります。
- [ ] **G4-6（学習中）：** r の学習に、疑似ラベルの箱も入れます。候補があるのに取りこぼす未知物体の多くで、σ(r) が低いことが分かっているためです（`tools/diag_unk_miss.py`）。
- [ ] **疑似ラベルの更新方法：** 学習中に疑似ラベルを更新する方法を、手元のチェックポイントで模擬します（`tools/sim_pseudo_update.py`）。
