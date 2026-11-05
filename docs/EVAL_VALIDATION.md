# 評価コードの検証（PROB 公式重みでの再現）

計画書 §10「評価コードの不整合」への対応。E0 の Known AP50（70.87）が文献値（約 60）より 10pt 以上高かったため、
PROB の公式 Task 1 重みを (a) PROB の公式評価、(b) 本リポジトリの `owod/eval/owod_eval.py` の両方で採点して突き合わせた。

## 手順

| 項目 | 内容 |
|---|---|
| 重み | PROB 公式 `MOWODB/t1.pth`（README の Google Drive `results.zip`） → `third_party/prob_download/` |
| 環境 | `third_party/PROB/.venv`（uv, Python 3.10, torch 2.5.1+cu124, numpy 1.26）。MSDeformAttn は sm_86/sm_89 向けにビルド |
| データ | `third_party/prob_data/OWOD/`：画像は symlink、COCO val の XML は PROB 同梱の `datasets/coco2voc.py` で生成、VOC07 test は元 XML。テストリストは PROB の `owod_all_task_test.txt` |
| 実行 | `tools/prob_check/run_prob_eval.py`（`main_open_world.py --eval` をそのまま実行し、評価器に渡る検出結果を全件ダンプ） |
| 再採点 | `tools/prob_check/compare_eval.py`（同じ検出結果を本評価器と `test.json` で採点） |

```bash
cd third_party/PROB
CUDA_VISIBLE_DEVICES=0 PROB_DUMP=../prob_eval_out/t1_dets.pt .venv/bin/python -u ../../tools/prob_check/run_prob_eval.py \
  --output_dir ../prob_eval_out --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_all_task_test --test_set owod_all_task_test --data_root ../prob_data/OWOD \
  --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain ../prob_download/MOWODB/t1.pth \
  --eval --wandb_project "" --batch_size 4 --num_workers 4
cd ../.. && uv run python tools/prob_check/compare_eval.py third_party/prob_eval_out/t1_dets.pt
```

## 結果

| | Known AP50 | U-Recall | WI | A-OSE | U-AP50 |
|---|---:|---:|---:|---:|---:|
| PROB 論文値 | 59.5 | 19.4 | – | – | – |
| (a) PROB 公式評価（再現） | 59.60 | 19.43 | 0.0569 | 5,219 | 1.71 |
| (b) 本評価器・**修正前** | **65.29** | **25.89** | 0.0550 | 5,219 | 1.98 |
| (b) 本評価器・修正後 | 59.60 | 19.43 | 0.0568 | 5,219 | 1.49 |

- PROB の公式重みと公式評価で論文値を再現できた → データ（テスト画像・GT）の構築は正しい。
- 修正前の本評価器は、**同じ検出結果に対して Known AP50 を +5.7pt、U-Recall を +6.5pt 甘く採点していた**。
- 修正後は Known AP50 / U-Recall / A-OSE が一致。残る差（WI 4 桁目、U-AP50）は同点スコアの並び順によるもので、後述。

## 見つかった不整合と修正（`owod/eval/owod_eval.py`）

### 1. テストリストの重複画像（主因：+5.7pt / +6.5pt）
`owod_all_task_test.txt` は 10,246 行だが異なる画像は 9,119 枚で、**1,127 枚が 2 回ずつ載っている**。
PROB / ORE の `voc_eval` は

- GT を画像**名**で引く（`class_recs[imagename]`）→ 重複画像の GT は 1 組だけ
- 正解数 npos はリストの**行**ごとに数える → 重複画像の GT は 2 回数えられる
- モデルは各行で推論する → 2 回目の検出は既にマッチ済みの GT に当たり FP になる

という挙動で、重複画像のリコール上限は実質半分になる。`test.json` は各行を別 image id にしていたため、本評価器では 2 回目も TP になっていた。
文献値はすべてこの挙動の下での値なので、本評価器を PROB に合わせた（`owod_name` をキーに GT を共有、npos は行ごとに数える）。

### 2. float32 のまま丸めていた座標
`round(x1 + 1., 1)` の `x1` が numpy float32 だったため結果も float32 になり（4.4 → 4.4000001）、IoU がちょうど閾値付近の判定が PROB とずれていた。
`float()` に変換してから丸めるよう修正。

### 3. 同点スコアの並び順（環境依存）
PROB はスコアを小数 3 桁に丸めてから `np.argsort` で並べるため、同点が大量に出る。デフォルトの argsort は同点の順序が numpy のバージョン / CPU の SIMD 経路で変わる。
**同じ PROB の採点コード・同じ検出結果でも**、numpy 1.26 では U-AP50 = 1.71、numpy 2.4 では 1.49 になることを確認した。
本評価器は再現性のため安定ソート（同点は画像順）に変更。この影響は Known AP50 で約 0.01pt、U-AP50 で約 0.2pt。U-Recall と A-OSE は順序に依存しない。
**U-AP50 は環境依存で ±0.2pt 程度揺れるため、文献値との比較には使わない**こと。

### テスト
`tests/test_owod_eval_vs_prob.py` に重複画像のペアを追加し、PROB 側も安定ソートにそろえて、5 指標すべてが一致することを確認（`ALL MATCH`）。

## 既存結果への影響

修正前の評価器で出した E0 / E1 の数値はすべて甘い。再採点した値は `docs/E1_POSTHOC.md` を参照。
学習中の評価（5 エポックごと）と `best.pth` の選択も修正前の評価器による。E0 では修正前の評価で最終エポックが best だった。
修正後の評価器でもそうなるかは確かめていないが、エポック間の差は小さく、修正の影響は全エポックに同じようにかかるため、結論に関わる可能性は低い。
