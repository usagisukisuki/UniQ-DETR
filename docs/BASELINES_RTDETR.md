# 比較手法の RT-DETRv2 への移植（設計）

Part I の M₀ と、検出器を RT-DETRv2 にそろえた PROB / OW-DETR / CAT / RandBox / OrthogonalDet を、同じ条件で比べる。
各手法の参照コードの調査記録は `docs/baselines/*_SPEC.md`。

方針（2026-10-21 決定）：
- 各手法は公開コードにできる限り忠実に移植する。
- CAT の selective search（学習なしの古典的な領域提案）は使う（比較対象の再現であり、Part I の「外部の信号を使わない」方針とは切り分ける）。
- 推論設定は全手法で同じ手順で決める（未知スコアの重み w をテスト画像の tune 側で選び、report 側で報告）。

## 1. 全手法で共通にそろえる条件

| 項目 | 設定 | 元の手法との違い |
|---|---|---|
| 検出器 | RT-DETRv2、PResNet-50（ImageNet 事前学習）、640×640、HybridEncoder、デコーダ 6 層 | 元は Deformable DETR（DINO ResNet-50）や Sparse R-CNN（R-50 + FPN）。backbone の事前学習も DINO ではない |
| 学習 | 50 エポック、AdamW lr 1e-4（backbone 1e-5）、バッチ 16、EMA、RT-DETRv2 のデータ拡張、fp32 | 各手法のスケジュール・拡張ではなく、検出器側のレシピにそろえる |
| データ | M-OWODB Task 1（`t1_train.json` 16,551 枚）、既知 20 クラス | PROB の TOWOD と同じ分割 |
| 評価 | 検証済みの評価器（`docs/EVAL_VALIDATION.md`）、1 画像 top-100 | RandBox / OrthogonalDet は元は上限なし＋スコア閾値（U-Recall が甘くなる）。ここでは top-100 にそろえる |
| 推論設定の選択 | w（未知スコアの倍率）を tune 側で「Known AP50 ≥ 61.5 で U-Recall 最大」に選び、report 側で報告 | 各手法の元の推論式はそのまま使い、w だけを共通の手順で調整する |
| 未出現クラス | クラス数は 20 + unknown（OrthogonalDet のみ 80 + unknown） | 元は 81 で 20..79 を mask。等価 |

## 2. 手法ごとの移植

### 2.1 PROB-RT（学習済み）
- 学習は E2（`configs/e2_obj.yml`）：全クエリの BatchNorm による物体らしさ（正例のみの尤度損失）、unknown クラス（未マッチを正例とする focal, 重み 0.1）。
- 推論は PROB と同じ：既知 o·p(c)、未知 o·σ(u)。`configs/b_prob_rt.yml`。
- 違い：分類損失は RT-DETRv2 の VFL（PROB は focal）。exemplar replay は Task 1 では不要。

### 2.2 OW-DETR-RT
- attention による疑似ラベル：backbone C4（1024 ch, stride 16）のチャネル平均を 640×640 に拡大し、未マッチのクエリの箱内平均の top-5 を unknown に。デコーダ層ごとに再計算。
- 疑似ラベルの箱は予測箱（detach、VFL の IoU 目標 = 1）。クラス損失と物体らしさ損失のみ、箱の回帰なし、num_boxes は GT のみ。
- 物体らしさの枝：各層 Linear(256,1)、focal、重み 0.05（OW-DETR の cls 2 : NC 0.1 の比を VFL 重み 1 に合わせる）。推論では使わない。
- 疑似ラベル・物体らしさ損失は epoch 9（50 エポック中）から。
- 推論：sigmoid(300 × 21)、top-100。

### 2.3 CAT-RT
- カスケード復号：パス 1（位置推定）= RT-DETRv2 のデコーダ（box refine あり）で箱と物体らしさ。パス 2（識別）= **別の重みの**デコーダ（SKDF のコードに従う。論文は共有）に、
  query_pos = パス 1 最終層の埋め込み、content = 0、参照箱 = パス 1 の最終の箱（detach。SKDF は box refine がないため初期参照点 = 最終参照点）。クラス（21）はパス 2。
- 補助損失：パス 2 の i 層とパス 1 の i 層を組にする。マッチングはパス 2 のクラスとパス 1 の箱。
- 疑似ラベル（論文 Eq.3-8 を実装。SKDF では未稼働のため）：S_i = norm(s_o)^{W_m} · (max_j IoU(SS_j, box_i))^{W_I}。
  s_o は OW-DETR と同じ attention スコア、SS は selective search（`scale 500, sigma 0.9, min_size 200`、領域 < 2000 画素とアスペクト比 > 2 を除外, 学習画像について事前計算）。
  W_m = 0.8, W_I = 0.2 から、損失の推移（直近 20 回 / その前 20 回の重み付き平均の比）で更新（π_nma 0.5, π_pma 1/3）。T_b = 40 it ごと、疑似ラベル開始（epoch 5）以降。
  top-5 を unknown（クラス損失と物体らしさ損失のみ）。
- 損失：クラス VFL 1、物体らしさ focal 0.5（CAT の cls 2 : obj 1 の比）、L1 5、GIoU 2。
- 推論：sigmoid(21)、物体らしさは使わない、top-100。

### 2.4 RandBox-RT
- エンコーダの top-k クエリ選択を**ランダムな箱**に置き換える（エンコーダの選択用 head の補助損失は外す）。学習時 500 個/画像/イテレーション、
  切り詰め正規分布（N(0,1) を [−2,2] に切り詰め [0,1] へ）、w, h ≥ 0.02。内容クエリ = 箱内のエンコーダ特徴の多段平均 → Linear + LN。
- デノイジング（CDN）は切る。
- マッチング：SimOTA dynamic-k（コスト 5·L1 + 2·focal + 2·(−GIoU) + 100·中心外 + 1e4·候補外、k = Σ top-5 IoU）を各デコーダ層で。
- 損失：sigmoid focal（重み 2, VFL ではない）、L1 5、GIoU 2。unknown：Σ sigmoid(21) の top-10（Known-FG の最良クエリを除く）を label 20 として 2 本目の focal（β 0.1）、500 it 遅らせる。
- 推論：格子 10,000 個（中心 {0.05..0.95}, w,h {0.1..1.0}）を 500 個 × 20 回、最終層、sigmoid、クラスごとの NMS 0.6 → top-100。

### 2.5 OrthogonalDet-RT（RandBox-RT の上に構築）
- 各デコーダ層の分類 head を極座標 head に：`f = LN(Linear_nobias(q))`、`p_obj = sigmoid(BN1d(||f||))`、`logits = Linear(LN(f))`（81 スロット: 80 + unknown）。
- 合成 unknown（Eq.3, logsumexp で安定化）、p_joint = softmax · p_obj を分類損失（focal on logit(p_joint)）・マッチング・NC 選択・推論に使う。
- 非相関化損失（クラス確率は detach、重み 1、全層、eps）。NC：p_obj の top-10、β 0.1、イテレーション 0 から。
- 推論：p_joint、クラスごとの NMS 0.6、既知 ×0.75・unknown ×2.0、top-100。

## 3. 進め方
1. 実装順：OW-DETR-RT → CAT-RT → RandBox-RT → OrthogonalDet-RT（依存関係と難度の順）。
2. 各手法：単体テスト → 2 エポックの試し学習（損失・検出が壊れていないか、閉集合の Known AP50 が E0 の同時点に近いか）→ 50 エポック。
3. 評価：`tools/sweep_pp.py` で w を選び report 側を報告。内訳は `tools/diag_unk_dets.py`。FPS も測る。
4. 1 本あたり学習 7〜9 時間。RandBox/OrthogonalDet は推論で 10,000 個の箱を流すため評価が重い。
