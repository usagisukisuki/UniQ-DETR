# OrthogonalDet 移植仕様（参照コード調査の記録）

出典：OrthogonalDet 論文（CVPR 2024）と公式実装（`third_party/baselines_src/OrthogonalDet`, commit d36b8e5）。RandBox の上に構築。
新しい部分は `cfg.MODEL.DISENTANGLED`（0 = RandBox、1 = 物体らしさ head を分離、**2 = 本手法（既定）**）と `DECORR_WEIGHT > 0` の裏にある。
**Task 1 ではタスク間の較正層（Sec. 3.5）は使われない**（`TEST.PREV_CLASSES` が空）。

## 1. RandBox からの変更

### 1a. head（`core/head.py:149-185`）
- `cls_module`：Linear(256,256, bias なし) → LayerNorm(256)（ReLU は外す, "avoid partial sphere"）。
- `norm4` = LayerNorm(256)（方向の正規化）。
- `object_logit` = BatchNorm1d(1)（物体らしさ head はこれだけ。Linear なし）。
- `class_logits` = Linear(256, 81)。0..79 が既知スロット（Task 1 では未出現の 60 クラスを含む）、80 が生の unknown logit。

### 1b. 極座標の分離（`head.py:226-256`）
```
f      = cls_module(x)            # LN 出力（affine あり）, ReLU なし
p_obj  = sigmoid(BN1d(||f||_2))   # 半径
logits = Linear(LN_norm4(f))      # 方向 → 81 logits
```
論文 Eq.2 は p_cls = h_cls(f/||f||) だが、コードは L2 除算ではなく LayerNorm。

### 1c. 合成 unknown 確率（論文 Eq.3, `head.py:249-255`）
```
m = max_{k<80} l_k ; z = l − m ; S = Σ_{k<80} exp(z_k)
l_unk' = log(S−1) + log S − log(1 + exp(z_80))
```
softmax 後 p(unk) = (S−1)/(S + e^{z_80})。S = 1 で −inf になるので logsumexp で安定化する。

### 1d. 同時確率 p_joint = softmax(logits)[c] · p_obj
分類損失（logit(p_joint) に sigmoid focal）、NC 損失、マッチングのコスト、推論のすべてで使う（RandBox は sigmoid(logits)）。

### 1e. 予測の非相関化損失（論文 Eq.4, `core/loss.py:189-199`）
クラス確率（detach）と物体らしさのピアソン相関の 2 乗を、クラス平均。全提案（バッチ × 500）で計算。勾配は物体らしさ側のみ。重み 1.0、全段。eps を足す。

### 1f. unknown の疑似ラベル（NC 損失）
選択スコア = Σ_c p_joint = **p_obj**。top-10（`FORWARD_K`）、各 GT の最良クエリを除外、label 80、箱の回帰なし、
他のクエリはすべて全 0、疑似 unknown 数で正規化、重み 0.1、全段、イテレーション 0 から。GT なし画像でも動く。

### 1h. 変更なし
学習時 500 提案、SimOTA dynamic-k マッチャ、L1/GIoU、深い監督、ランダムな箱の学習。

## 2. 損失・ハイパーパラメータ
L = 2·focal(p_joint) + 5·L1 + 2·GIoU + 0.1·L_nc + 1.0·L_decorr（6 段の和）。focal α 0.25, γ 2。
AdamW lr 2.5e-5、wd 1e-4、バッチ 12、T1 20,000 it（step 15k/18k）、warm-up 1000、勾配クリップ 1.0、EMA なし、RandBox と同じ拡張。

## 3. 推論（`core/detector.py:194-243, 362-418`）
1. 格子 10,000 個を 500 個 × 20 回、最終段のみ。
2. スコア = softmax(logits_out) · p_obj（合成 unknown を含む 81）。
3. top-10,000 → クラスごとの NMS 0.6。
4. **再スケール**：既知 ×0.75、unknown ×2.0（DISENTANGLED=2 のみ）。
5. 評価器でスコア < 0.15 を除外。上位件数の上限なし。

## 4. 本質と RT-DETRv2 への対応
論文 Table 4（Task 1, U-Recall / K-mAP）：base 8.4/59.8、+特徴の直交化 18.2/61.3、+非相関化のみ 15.2/59.9、全部 24.6/61.3。
- 本質：極座標 head（デコーダ各層、クエリ埋め込みに適用。BN 統計はデノイジングのクエリを除く）、合成 unknown、p_joint、非相関化損失、NC 損失、推論の再スケール。
- 付随：ランダムな箱・格子推論、RoI/DynamicConv、SimOTA、NMS、スケジュール。
- 本比較では OrthogonalDet を RandBox-RT の上に構築する（元の実装が RandBox ベースのため）。
