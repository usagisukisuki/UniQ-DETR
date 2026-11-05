# RandBox 移植仕様（参照コード調査の記録）

出典：RandBox 論文（ICCV 2023）と公式実装（`third_party/baselines_src/RandBox`, commit 9b1f7d8）。パスはリポジトリ相対。

拡散モデル由来の仕組み（`detector.py:76-119`）は使われていない（`SAMPLE_STEP=1`、t 不使用）。実態は
**ランダムな箱を候補にした Sparse R-CNN ＋ unknown logit ＋ マッチングスコア上位 N 個への追加損失**。

## 1. 候補と head

- 学習時：1 画像 N = 500（`t1.yaml`）、毎イテレーション新たに生成、GT は混ぜない。
  コードは `clamp(N(0,1), 0, 1)` 相当で、約 75% が幅か高さ 0 の退化した箱（**バグ**）。論文は「N(0,1) を [−2,2] に切り詰めて [0,1] に線形変換、画像外は縮める」。
- 推論時：固定の 10×10×10×10 = 10,000 個の格子（cx, cy, w, h ∈ {0.0, …, 0.9}）を 500 個ずつ 20 回に分けて head に通す（`detector.py:120-127, 171-183, 217-230`）。
- head：6 段（重み共有なし）、段間で箱を detach、RoIAlign 7×7（FPN p2–p5）、初段の特徴は箱内 RoI 特徴の平均、
  self-attention → DynamicConv → FFN → cls/reg、sigmoid の class logit（81 個, index 80 = unknown）、Faster R-CNN の delta 表現。

## 2. マッチング（`loss.py:355-578`）

- 6 段それぞれで独立にマッチング。
- **Known-FG**：SimOTA（DiffusionDet 由来）。コスト = 5·L1 + 2·focal_cls + 2·(−GIoU) + 100·(中心領域外) + 1e4·(fg_mask 外)。
  k_g = clamp(int(Σ top-5 IoU), min 1)。重複は最小コストの GT に寄せる。
- **Unknown-FG**：マッチングスコア = Σ sigmoid(全 81 logit)、1 画像 top-10（`FORWARD_K=10`）、各 GT の最良クエリを除外
  （論文は「Known-FG を除外」、コードは最良クエリのみ除外）。label 80。
- **BG**：それ以外すべて。
- バグ：GT 0 個の画像でバッチ index がずれる。

## 3. 損失

- `loss_labels`：sigmoid focal（α 0.25, γ 2）、Known-FG は one-hot、それ以外（Unknown-FG と unknown logit を含む）は全 0。重み 2。
- `loss_boxes`：Known-FG のみ、L1 ×5（正規化 xyxy）+ GIoU ×2。
- `loss_nc_labels`：2 本目の focal、Unknown-FG を index 80 の one-hot、他は全 0、unknown の数で正規化、**β = 0.1**。unknown の箱の回帰はしない。
- 論文は L_U を 500 イテレーション遅らせる（コードは 0 で無効）。全 6 段に同じ重みで深い監督。物体らしさ head はない。

## 4. 推論

- 最終段のみ。sigmoid スコアを (10,000 × 81) で平坦化し top-10,000、unknown は index 80 の普通のクラス。
- クラスごとの NMS（IoU 0.6）、**上位件数の上限なし**、評価器でスコア < 0.15 を除外（論文は 0.1）。
  **上限なしは U-Recall を甘くする**（他手法は top-100）。本比較では全手法 top-100 にそろえる。
- テスト時のリサイズは短辺 800 / 長辺 1333。

## 5. ハイパーパラメータ

AdamW lr 2.5e-5、wd 1e-4、勾配クリップ 1.0、warm-up 1000 it、バッチ 12、T1 は 20,000 it（約 14.5 エポック, step 15k/18k）、
R-50 + FPN、DETR 型の multi-scale + crop 拡張。

## 6. 本質と付随物

- 本質：学習データに依存しないランダムな候補（論文 Table 2a：RPN 59.0/8.6、クエリ 59.7/9.1、ランダム 61.8/10.6）、
  マッチングスコアによる Unknown-FG（top-10, β 0.1）、unknown logit を含む独立 sigmoid の分類、残りは BG。
- 付随物：RoI pooling、DynamicConv、delta 表現、SimOTA の細部、損失の重み 2/5/2、拡散のバッファ。NMS は 1 対多のマッチングに伴って必要。

## 7. RT-DETRv2 への対応づけ（調査からの提案）

1. エンコーダの top-k クエリ選択をランダムな箱に置き換える（エンコーダの選択用 head と補助損失は外す）。論文どおりの切り詰め正規分布を使い、w, h の下限を設ける（退化した箱は inverse_sigmoid で発散する）。
2. 内容クエリ：箱内のエンコーダ特徴の平均（RoIAlign 相当）＋ Linear + LN。
3. クエリ数：学習 500（メモリが厳しければ 300）。
4. デコーダ 6 層を 6 段とみなし、各層に RandBox の損失（層ごとのマッチングと Unknown-FG 選択）。
5. マッチング：SimOTA の dynamic-k に置き換え、推論にクラスごとの NMS（0.6）を加える。
6. デノイジング（CDN）は切る。
7. 分類：21 個の sigmoid logit（index 20 = unknown）、VFL ではなく sigmoid focal（重み 2）、unknown 損失 β 0.1、500 it 遅らせる。
8. 推論：格子 10,000 個を 500 個ずつ、最終層のみ、クラスごと NMS 0.6、top-100（本比較の共通条件）。
