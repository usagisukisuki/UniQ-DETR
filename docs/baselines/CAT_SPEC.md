# CAT 移植仕様（参照コード調査の記録）

出典：CAT 論文（arXiv 2301.01970）と、同じ著者の SKDF（`third_party/baselines_src/SKDF`）のコード。CAT の公式リポジトリ
（github.com/xiaomabufei/CAT）は 404。SKDF は CAT の部品を含むが、GLIP 知識蒸留など SKDF 固有の部分は除外する。
略記：DD = `SKDF/models/deformable_detr.py`、DT = `SKDF/models/deformable_transformer.py`。

## 1. カスケード復号

- Deformable DETR（DINO ResNet-50、4 レベル、エンコーダ 6 層、d 256）。クエリ 100、`query_embed` Embedding(100, 512) を位置 256 + 内容 256 に分割。参照点は 2 次元、box refine なし。
- **パス 1（位置推定）**：`local_hs = local_decoder(tgt, ref, memory, query_pos)`、6 層、中間出力すべて（DT:159）。
- **パス 2（識別）**：`class_query_pos = local_hs[-1]`、`class_tgt = 0`、参照点はパス 1 の**初期**参照点（DT:163-169）。論文 Eq.(2) E_Class = F_s(M, E_Location, R) と一致。
- **重み共有**：論文は 1 つのデコーダ F_s を 2 回使う。SKDF は独立した 2 つのデコーダ（DT:43-45）。**食い違い。**
- ヘッド：box（MLP 3 層）と物体らしさ（Linear 256→1, prior bias −log 99）はパス 1、クラス（Linear 256→81, index 80 = unknown）はパス 2（DD:216-246）。
  論文本文は「E_class を物体らしさと新規性分類へ」と書く（食い違い）。
- 補助出力：パス 2 の i 層とパス 1 の i 層を組にした 5 組（DD:249-277）。マッチングはパス 2 のクラスとパス 1 の箱。
- 推論：箱はパス 1 最終層、クラスはパス 2 最終層。

## 2. 疑似ラベル

- **SKDF で実際に動くもの**（DD:686-710）：epoch ≥ nc_epoch(5) から、未マッチのクエリを物体らしさ logit のクエリ間 softmax で並べ top-5 を unknown（label 80, 信頼度 = softmax 値）。CAT 論文の方法ではない。
- **論文のモデル由来 PLM**（OW-DETR と同じ）：backbone C4 のチャネル平均を画像サイズに拡大し、未マッチのクエリの箱内の平均活性を s_o とする（参照実装 PROB/models/deformable_detr.py:474-513）。
- **入力由来 PLM（selective search）**：SKDF/models/test_selective_search.py:27-60 のコメントアウトされた断片のみ。
  `selective_search(img, scale=500, sigma=0.9, min_size=200)`、重複除去、領域サイズ < 2000 画素を除去、アスペクト比 > 2 を除去。CPU 処理なので事前計算する。
- **自己適応的な組み合わせ**（論文 Eq.3-8, Alg.1。SKDF/util/store.py:51-103 の `Memory_loss` は未使用でバグあり）：
  - S_i = norm(s_o,i)^{W_m} · (max_j IoU(P^I_j, P^m_i))^{W_I}。未マッチのクエリから S_i 上位 5 個。norm は画像内の min-max。
  - 初期 W_m = 0.8、W_I = 0.2。
  - 損失の記録 L_m（deque）。Δl = 直近窓の重み付き平均 / 古い窓の重み付き平均（重みは線形減衰で和 1）。
  - Δl > 1：Δw = π_nma · sigmoid(Δl − 1)（π_nma = 0.5）。Δl ≤ 1：Δw = −π_pma · Δl（π_pma = 1/3）。
  - W_m ← W_m + Δw·W_m、W_I ← W_I − Δw·W_I、和 1 に正規化。T_b ごとに更新、T_start 以降。T_b・T_start は論文・コードとも未記載。
- 疑似ラベルの使い方：クラス損失（unknown index 80, hard target 1, OW-DETR 方式）と物体らしさ損失の正例。**箱の回帰はしない**。num_boxes は GT のみ。

## 3. 損失・マッチャ・ハイパーパラメータ

- L = 2·focal(cls, α 0.25, γ 2) + 1·focal(物体らしさ) + 5·L1 + 2·GIoU。最終層 + 補助 5 層。
- 未出現クラスの logit は −1e11（invalid_cls_logits）。
- マッチャ：class 2（focal 型）、L1 5、GIoU 2。
- クエリ 100、AdamW lr 2e-4（backbone 2e-5）、wd 1e-4、clip 0.1、StepLR 40、T1 は 50 エポック（旧分割）。

## 4. 推論

- sigmoid(全 81 クラス)、unknown は index 80 の普通のクラス。物体らしさは推論で使わない。top-100（論文は top-50）。未出現クラスを mask する。

## 5. 除外する SKDF 固有部分

GLIP の unknown_boxes 読み込みと targets への統合、down-weight loss（信頼度で重み付けした focal・L1・GIoU）、マッチャの信頼度コスト、`--use_sam` 等のフラグ、first/second stage モード。
