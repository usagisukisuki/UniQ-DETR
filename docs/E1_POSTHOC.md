# E1: post-hoc 未知スコアの比較

E0（既知20クラスで閉集合学習した RT-DETRv2）の重みを**一切再学習せず**、推論時の未知スコア関数だけを差し替えて
U-Recall などを測る実験。後続の E2〜E4（学習改造あり）が超えるべき下限を確定するのが目的。

- 重み: `outputs/e0_closed/best.pth`（epoch 50, Known AP50 64.51）
- 数値はすべて検証・修正後の評価器による（`docs/EVAL_VALIDATION.md`）。修正前の値は Known AP50 で約 +6pt、U-Recall で約 +6pt 甘かった。
- 実装: `tools/e1_posthoc.py`（評価）、`owod/zoo/ow_postprocessor.py`（スコア関数）、`tools/vis_e1.py`（可視化）
- 結果: `outputs/e0_closed/posthoc_scores.json`、ログ `outputs/logs/e1_posthoc.log`

```bash
uv run python tools/e1_posthoc.py -c configs/e0_closed.yml -r outputs/e0_closed/best.pth --fit-gaussian
```

## 推論時の出力の作り方（4手法共通）

デコーダ最終層の各クエリ q（300個）について、

- 既知クラス c のスコア: `s_c(q) = p(c|q) = sigmoid(logit_c)`（E1 では objectness を学習していないので `known_score=cls`）
- 未知スコア: `s_unk(q)`（下記4手法で異なる）

を並べた `[Q, 21]` のスコア表を flatten し、**上位 100 件**を検出結果とする（PROB と同じ）。
既知と未知が同じ top-100 枠を取り合うため、**未知スコアの値の大きさ（スケール）が既知の検出を押し出すかどうか**を直接左右する。
この点が結果を読むうえで最も重要。

## 4手法

| 名前 | 式 | 使う情報 | 追加の準備 |
|---|---|---|---|
| **MSP** | `1 − max_c p(c\|q)` | 分類ヘッドの出力のみ | なし |
| **Energy** | `sigmoid(−T · logsumexp_c(logit_c / T))`, T=1 | 分類ヘッドの logit のみ | なし |
| **Mahalanobis (maha)** | `o(q) = exp(−τ · d²(q) / D)` | クエリ埋め込み | 学習セットでガウスをフィット |
| **C1** | `o(q) · (1 − max_c p(c\|q))` | 埋め込み + 分類 | 同上 |

### MSP（Maximum Softmax Probability）
OOD 検出の古典的ベースライン（Hendrycks & Gimpel, 2017）。「どの既知クラスにも自信がない」クエリほど未知らしいとみなす。
RT-DETR は softmax ではなく sigmoid（focal loss）で学習しているため、ここでは max sigmoid を使う。

**弱点**: 「既知クラスに自信がない」のは未知物体だけでなく**背景**も同じ。DETR 系の 300 クエリの大半は背景なので、
背景クエリのスコアがほぼ 1.0 になり top-100 を独占する。

### Energy
Energy-based OOD（Liu et al., 2020）。全クラスの logit をまとめた `logsumexp` を「既知らしさ」とし、その逆を未知スコアにする。
MSP が最大クラスだけを見るのに対し、全クラスの logit を使う点が違う。0〜1 に揃えるため sigmoid に通している。

**弱点**: MSP と同じく分類ヘッドの出力だけに依存するので、背景と未知物体を区別できない。

### Mahalanobis（maha）
「既知物体にマッチしたクエリ埋め込みの分布」からの距離で**物体らしさ（objectness）**を測る。

1. 学習セット（16,551枚、拡張なし）で E0 を推論し、既知 GT とハンガリアンマッチしたクエリの最終層埋め込み（D=256）を集める（47,223個）。
2. その平均 μ と共分散 Σ（+ 1e-3·I の shrinkage）を求める。**使うのは既知 GT のみ**で、未知の情報は入らない。
3. テスト時、各クエリの埋め込み z について `d² = (z−μ)ᵀ Σ⁻¹ (z−μ)`、`o(q) = exp(−τ d²/D)`（τ=1.3）。

既知物体に近い埋め込みほど o が大きい。背景クエリは既知物体の分布から遠いので o が小さくなり、
「物体っぽいが既知クラスとは限らない」クエリを拾うことを狙う。温度 τ と D による正規化は計画書からの逸脱（`docs/IMPLEMENTATION.md` 参照）。

**弱点**: 既知物体そのものも o が高いので、未知スコアとしては既知物体と重なりやすい。
ただし同じクエリでは既知クラスのスコア p(c|q) の方が高くなりやすく、top-100 の取り合いで既知が残る。

### C1
計画の C1（objectness × 未知らしさ）を E0 に post-hoc で当てはめたもの。
`o(q)`（物体らしさ）と `1 − max p`（既知らしくなさ）の積で、「物体らしい かつ 既知クラスではない」クエリを高くする。
MSP の「背景も拾う」問題を o で抑え、Mahalanobis の「既知物体も拾う」問題を `1 − max p` で抑える設計。
E2 以降では o を学習で作る（C2）が、E1 では上の post-hoc フィットの o を使う。

## 結果（M-OWODB Task 1 test, 10,246枚, 未知 GT 23,320個）

| 未知スコア | Known AP50 | U-Recall | WI | A-OSE | U-AP50 |
|---|---|---|---|---|---|
| なし（E0, 閉集合） | 64.51 | 0.00 | 0.0511 | 56,433 | 0.00 |
| MSP | 7.73 | 16.39 | 0.0158 | 18 | 0.07 |
| Energy | 41.15 | 15.00 | 0.0383 | 577 | 0.06 |
| **Mahalanobis** | **64.41** | **16.81** | 0.0445 | 11,220 | **0.16** |
| C1 | 64.43 | 15.08 | 0.0425 | 12,200 | 0.09 |

### 文献値との比較（M-OWODB Task 1, `docs/RT-DETRv2_OWOD.md` §5.3）

| 手法 | U-Recall | Known mAP | 学習改造 |
|---|---:|---:|---|
| PROB | 19.4 | 59.5 | あり |
| Decoupled PROB | 20.3 | 59.8 | あり |
| CAT | 23.7 | 60.0 | あり |
| RandBox | 10.6 | 61.8 | あり |
| OrthogonalDet | 24.6 | 61.3 | あり |
| **E1: Mahalanobis（本実験）** | **16.8** | **64.4** | **なし**（post-hoc） |
| E1: C1（本実験） | 15.1 | 64.4 | なし（post-hoc） |

評価器は PROB 公式重みで PROB 公式評価と一致することを確認済み（`docs/EVAL_VALIDATION.md`）なので、この比較は同じ物差しによる。
学習改造なしの post-hoc で、U-Recall は RandBox（10.6）を上回るが PROB（19.4）には届かない。
Known mAP は文献値より約 4〜5pt 高い。RT-DETRv2 は閉集合検出器として強く、未知スコアが既知の枠をほとんど奪わないためと考えられる。

## 考察

1. **MSP / Energy は使えない**: Known AP50 が 7.7 / 43.8 に崩壊。背景クエリの未知スコアが ~1.0 で top-100 を埋め、既知の検出が押し出されている。
   WI・A-OSE が良く見えるのは既知の出力が減ったためで、性能向上ではない。
2. **Mahalanobis / C1 は Known AP50 をほぼ維持**（−0.1pt）しつつ U-Recall 15〜17%。今回は Mahalanobis 単独が C1 よりわずかに良い。
   post-hoc の o では「1 − max p」を掛けても未知の選別が改善しておらず、C1 の効果は o を学習した後（E2 以降）で改めて評価する。
3. **U-AP50 はどの手法も 0.2 未満**: 未知物体の位置はある程度当たっている（Recall）が、スコアの順位がほぼ無意味。
   可視化でも maha / C1 の未知スコアは 0.2 前後に集中し、小物体や背景の端を拾うことが多い。
4. **評価基準への影響**: 計画書の「E1 で U-Recall > 10% なら後続の基準を引き上げる」に該当（16.8%）。
   M₀ の Gate（U-Recall ≥ 10%）は、E1 の Mahalanobis（16.8%）を上回ること、さらに PROB（19.4%）水準を目標に読み替えるのが妥当。

## 可視化

```bash
uv run python tools/vis_e1.py -c configs/e0_closed.yml -r outputs/e0_closed/best.pth
```

- 出力: `outputs/e0_closed/vis_e1/<image_id>.png`。1画像につき「GT | msp | energy | maha | c1」を横に並べる。
- 青 = 既知クラス、赤 = `unknown`。GT 側の未知は `unknown (truck)` のように元の COCO クラス名を併記。
- 表示条件: 既知は score ≥ 0.5（`--known-thr`）、未知は NMS(IoU 0.5) → 表示中の既知と IoU ≥ 0.5 で重なるものを除外 → 上位 5 件（`--unk-topk`）。
  評価（top-100 全件）とは異なる、見やすさのための絞り込み。
- 画像は既知・未知の GT が両方ある画像からランダムに 12 枚（`--seed 0`）。`--image-ids` で指定可。
- 学習セットでフィットしたガウスは `outputs/e0_closed/e1_gaussian.pt` にキャッシュされ、2回目以降はフィットを省略する。
- 背景クエリは画像外の箱を回帰することがあるため、描画時に画像範囲へクリップしている。

## 未知 GT について

未知の GT は新たに作ったものではなく、テストセットの COCO val2017 画像の正解アノテーション（80クラス）のうち、
タスク1で既知でない 60 クラス（`category_id ≥ 20`）を未知として扱っているだけ（`scripts/prepare_mowodb.py`）。
VOC07 test 画像には 20 クラス分の注釈しかないため未知 GT はない（ORE / PROB と同じ仕様）。
未知 GT は `test.json` にのみ存在し、学習用データセットは未知 GT を含むファイルを読み込まない（`owod/data/mowodb.py`）。
