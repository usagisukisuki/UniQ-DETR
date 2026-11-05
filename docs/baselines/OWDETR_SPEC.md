# OW-DETR 移植仕様（参照コード調査の記録）

出典：OW-DETR 論文（CVPR 2022）と公式実装（`third_party/baselines_src/OW-DETR`, commit 3515ff8）。手法は `models/deformable_detr.py` に集約。
名前の注意：論文の「novelty classification」はクラス分類器の unknown クラス（index 80）。コードの `NC_branch` / `nc_class_embed` / `loss_NC` は
論文の**前景の物体らしさ**の枝。

## 1. attention による疑似ラベル（`SetCriterion.forward`, :452-513、補助層 :529-578）
- 特徴：backbone **C4（ResNet layer3, 1024 ch, stride 16）**の `input_proj` 前の生出力（`--featdim 1024`）。DINO ResNet-50、FrozenBN。
- 活性マップ：チャネル平均（正規化なし）を、パディング込みのバッチ画像サイズに bilinear で拡大。
- 各予測箱の箱内平均をスコアに（空の箱は −1e10）。
- 候補：Hungarian でマッチしなかったクエリのみ。IoU・大きさ・閾値の絞り込みなし。
- **top-5**（1 画像）を unknown（label 80）として、既存のマッチ index に追記。**クラス損失と物体らしさ損失のみ**、箱の回帰なし、num_boxes は GT のみ。
- 開始：`epoch ≥ 9`（50 エポック中。ハードコード）。現在のモデル（オンライン）を使用。補助層ごとに再計算。
- 公式コードのバグ：補助層ループの変数の上書きで、epoch 9 以降は補助層の損失が最後の 1 層分しか残らない。本比較では再現しない。

## 2. novelty 分類（unknown クラス）
Linear(256, 81)、index 80 = unknown、未出現クラス 20..79 は損失で −1e10。sigmoid focal（α 0.25, γ 2）、重み 2。

## 3. 物体らしさの枝
Linear(256, 1)（prior 0.01）。focal で、マッチした GT と疑似 unknown が 1、他は 0。重み 0.1（epoch < 9 は 0）。**推論では使わない**。

## 4. ハイパーパラメータ（M-OWODB = `configs/OWOD_split.sh`）
クエリ 100、50 エポック、top_unk 5、nc_loss_coef 0.1、nc_epoch 9、DINO ResNet-50、AdamW lr 2e-4（backbone 2e-5）、wd 1e-4、clip 0.1、
StepLR 40、バッチ 16、4 レベル、エンコーダ / デコーダ 6/6、box refine・two-stage なし、マッチャ class 2 / L1 5 / GIoU 2。
分割ファイルは PROB の TOWOD と同一（cmp で確認）。**公式リポジトリのクラス順は OW-DETR 独自分割のもの**で、M-OWODB では PROB の TOWOD の順序を使う必要がある。

## 5. 推論
sigmoid(100 × 81) を平坦化して top-100、unknown は index 80 の普通のクラス。物体らしさの掛け算なし、NMS なし。
公式は未出現クラスを mask しない（PROB の版は mask する）。

## 6. 公式と PROB 内のコピーの違い
PROB の `models/deformable_detr.py` は後処理の mask と build の戻り値以外は同一。PROB のコピーは評価のみで学習はできない
（criterion の引数が違う）。PROB の README の OW-DETR の M-OWODB 値（U-Recall 7.5, mAP 59.2）は外部で学習した重みによる。

## 7. RT-DETRv2 への対応（調査からの提案）
1. 注意の特徴：RT-DETRv2 の backbone（PResNet-50, return_idx [1,2,3]）の C4（1024 ch, stride 16）を HybridEncoder の前で取り出し、detach してチャネル平均。
   640×640 に拡大し箱内平均（パディングなし）。積分画像で高速化可。
2. 候補：マッチング用の 300 クエリのみ（デノイジングのクエリは除く）、Hungarian でマッチしたものを除外、top-5、デコーダ層ごとに再計算。エンコーダの補助損失は GT のみ。
3. unknown クラスと VFL：疑似ラベルの箱は予測箱（detach）とし IoU = 1 → VFL の目標 1.0（OW-DETR の focal の正例と等価）。L1/GIoU と num_boxes からは除外。
4. 物体らしさの枝：各デコーダ層に Linear(256,1)。重み：OW-DETR は cls 2 に対し 0.1、RT-DETRv2 の VFL 重み 1 に合わせると 0.05。推論では使わない。
5. 疑似ラベルと物体らしさ損失は学習の 18%（50 エポック中 epoch 9）から。オンラインのモデル（EMA ではない）。
6. 推論：未出現クラスを mask、sigmoid(300 × 21)、top-100、NMS なし。
