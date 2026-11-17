# UniQ-DETR の M-OWODB Task 2〜4（2026-11-10 開始）

UniQ-DETR（Task 1 = G4-8、`outputs/g4_8/best.pth`）を、PROB の手順で Task 2〜4 に進める。GPU0 の 1 枚で回す（`scripts/run_uniq_tasks.sh`）。

## 手順（PROB `configs/M_OWOD_BENCHMARK.sh` を RT-DETRv2 のレシピに写したもの）

| 段階 | データ | ラベル | エポック | lr（backbone） | PROB |
|---|---|---|---:|---|---|
| Task t 段階 1 | owod_t{t}_train（4.0〜4.6 万枚） | Task t の 20 クラスだけ（前のクラスと未知は消す） | 10 | 1e-5（1e-6） | 10 エポック、lr 2e-5（基準の 0.1 倍） |
| Task t 段階 2 | owod_t{t}_ft（ORE の固定の例示セット、1,743 / 2,361 / 2,749 枚） | 既知 20t クラスすべて | 60 / 60 / 70 | 1e-4（1e-5）、40 / 35 / 50 エポック後に 0.1 倍 | t{t}_ft と同じエポック数と lr_drop |

- 例示セットは ORE の固定リストを使う。OW-DETR / CAT / OrthogonalDet と同じで、PROB の学習による例示選択は移植しない。
- 各段階は前の段階の最後の重み（`last.pth`）から始める。テストでの最良選択（`best.pth`）は使わない（Task 1 の G4-8 は最終エポックが best）。
- 強いデータ拡張は、各段階の最後の 2 エポックで止める（Task 1 と同じ）。lr のウォームアップは段階 1 が 2,000 反復、段階 2 が 500 反復。
- 推論は Task 1 の UniQ-DETR の設定そのまま（600 クエリ、w 12、面積^0.25）。Task ごとに選び直さない。参考として w の掃引も保存する。

## 分類ヘッドを広げる（`tools/expand_classes.py`）

- 分類の層（`enc_score_head`、`dec_score_head.*`、`denoising_class_embed`）の行を、20(t−1)+1 から 20t+1 に増やす。
- 既知の行はそのまま、未知の行（と DN の詰め物の行）は末尾へ移す。新しいクラスの行は新しいモデルの初期値（RT-DETR の事前確率のバイアス）にする。
- 小さいテスト画像で、広げた直後のモデルの Prev（前タスクのクラス）の AP が Task 1 のモデルと一致することを確認した（8.64 = 8.64、64 枚）。

## 疑似未知の箱（`tools/make_pseudo_unk.py --exclude-*`）

- Task 1 と同じ教師（E2 のエポック 2）、同じ作り方（左右反転の一貫性、上位 5 個／画像）。
- 段階 1 のデータは前のクラスのラベルを消してあるので、前のクラスの物体（人・車など）が疑似未知になってしまう。
  - 前タスクの最終モデルが既知と判断した検出（前のクラスの最大 p ≥ 0.3）と IoU 0.5 以上の候補も除く。
- t2_train の 96 枚で、疑似未知 480 個がどの物体に乗るかを数えた（COCO の全 80 クラスのラベルで判定）。

| | 前のクラス（VOC） | Task 3/4 のクラス（未知） | どの物体にも乗らない |
|---|---:|---:|---:|
| 除外なし | 217（45%） | 10 | 253 |
| 前タスクのモデルで除外 | **29（6%）** | 10 | 441 |

- 段階 2（例示セット）は既知すべてにラベルがあるので、ラベルだけで除く。
- 疑似ラベルの学習は最初のエポックから効かせる（Task 1 はエポック 9 から）。

## 評価

- `OWODEvaluator(prev_known=20(t−1))` で、Prev（前のクラス）と Cur（新しいクラス）の Known AP50 も出す（PROB の表と同じ）。
- Task 4 は未知の GT がないので、U-Recall・WI・A-OSE はない。

## 結果

### Task 2（2026-11-10 19:55 終了、段階 1 4.3 時間 + 段階 2 1.4 時間）

テスト全体、w 12（`outputs/logs/eval_uniq_t2.log`）:

| Known AP50（Both） | Prev | Cur | U-Recall | WI | A-OSE | U-AP50 | 閉集合 Known |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 51.22 | 63.58 | 38.86 | 20.71 | 0.0348 | 12,847 | 9.22 | 51.80 |

- 段階 1 の終わり（エポック 9）は Prev 12.77 / Cur 39.50。前のクラスのラベルを消して学習するので Prev が大きく落ちる。段階 2 で 63.58 まで戻る（Task 1 は 64.16）。
- w の掃引（report 側）: w 4 は 52.11 / 15.53 / A-OSE 11,735、w 12 は 51.71 / 21.10 / 6,439、w 32 は 48.00 / 23.51 / 2,302。

### Task 3・Task 4（2026-11-11 08:32 全部終了）

テスト全体、w 12（`outputs/logs/eval_uniq_t{1_full,2,3,4}.log`）。PROB と同じくテスト全体で評価する。Task 1 の README の値（64.16 / 30.76 / 7,200）は w を選ぶための半分割の report 側。

| Task | Known AP50（Both） | Prev | Cur | U-Recall | WI | A-OSE | U-AP50 | 閉集合 Known |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1（G4-8） | 63.78 | — | — | 29.92 | 0.0546 | 14,507 | 9.56 | 64.27 |
| 2 | 51.22 | 63.58 | 38.86 | 20.71 | 0.0348 | 12,847 | 9.22 | 51.80 |
| 3 | 43.49 | 51.63 | 27.20 | 23.80 | 0.0180 | 5,411 | 2.36 | 44.41 |
| 4 | 38.99 | 43.59 | 25.18 | — | — | — | — | 39.77 |

- Task 3 の w の掃引（report 側）: w 4 は 44.90 / 17.97 / A-OSE 5,149、w 12 は 44.07 / 24.35 / 2,698、w 32 は 39.61 / 26.19 / 957。
- Task 4 は未知の GT がないが、w 12 の推論は一部の既知の検出を未知に回すので、Known が閉集合（未知を出さない）より 0.8 低い。Task 4 は閉集合の 39.77 を報告する選択肢もある。

## Task 1 をテスト全体で比べる（PROB と同じ評価、2026-11-11）

README の表は推論の設定を選ぶための半分割の report 側の値。論文の値と比べられるように、同じ設定（tune 側で選んだもの）のままテスト全体で評価した（`scripts/run_full_test_baselines.sh`、`tools/sweep_pp.py --full`、ログは `outputs/logs/full_test_*.log`）。report 側の値は README と全部一致した。

| Method | Post-Pro. | Known AP50 | U-Recall | WI ↓ | A-OSE ↓ |
|---|---|---:|---:|---:|---:|
| **UniQ-DETR**（w 12） | topk | 63.78 | **29.92** | 0.0546 | 14,507 |
| OW-DETR-RT（w 128） | NMS 0.5 | 61.92 | 27.23 | 0.0545 | 14,952 |
| CAT-RT（w 16） | NMS 0.7 | 62.11 | 25.60 | 0.0575 | 22,840 |
| PROB-RT（w 0.7） | NMS 0.7 | 62.54 | 24.04 | 0.0723 | 9,612 |
| RandBox-RT（w 256） | NMS 0.5 | 61.88 | 23.34 | 0.0360 | 10,823 |
| OrthogonalDet-RT（w 8） | NMS 0.5 | **65.13** | 18.79 | **0.0281** | **3,891** |
| OW-DETR-RT（w 32） | topk | 62.29 | 23.11 | 0.0552 | 6,310 |
| PROB-RT（w 0.6） | topk | 61.23 | 20.85 | 0.0598 | 4,193 |
| CAT-RT（w 2） | topk | 61.79 | 14.56 | 0.0485 | 4,300 |

- report 側と比べて、どの手法も Known は 0〜0.4、U-Recall は 0.6〜0.8 低く、A-OSE は約 2 倍。順位は変わらない。

## 比較手法の Task 2〜4（2026-11-11 開始、`scripts/run_baseline_tasks.sh`）

UniQ-DETR と同じ手順（`scripts/make_baseline_task_configs.py`）。未知の学習は各手法の元のまま（前のクラスの物体の除外はしない）。推論は README の Task 1 の設定のまま、テスト全体で評価。

| 手法 | Task | 後処理 | Known AP50 | Prev | Cur | U-Recall | WI | A-OSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| UniQ-DETR | 2 | topk（w 12） | 51.22 | 63.58 | 38.86 | 20.71 | 0.0348 | 12,847 |
| PROB-RT | 2 | NMS 0.7（w 0.7） | 50.50 | 62.63 | 38.36 | 17.33 | 0.0362 | 5,146 |
| PROB-RT | 2 | topk（w 0.6） | 49.90 | 61.35 | 38.46 | 14.78 | 0.0328 | 3,014 |
| UniQ-DETR | 3 | topk（w 12） | 43.49 | 51.63 | 27.20 | 23.80 | 0.0180 | 5,411 |
| PROB-RT | 3 | NMS 0.7（w 0.7） | 42.35 | 50.39 | 26.27 | 19.95 | 0.0175 | 2,417 |
| PROB-RT | 3 | topk（w 0.6） | 42.26 | 50.48 | 25.82 | 17.10 | 0.0150 | 1,487 |
| UniQ-DETR | 4 | topk（w 12） | 38.99 | 43.59 | 25.18 | — | — | — |
| PROB-RT | 4 | NMS 0.7（w 0.7） | 38.70 | 43.81 | 23.37 | — | — | — |
| PROB-RT | 4 | topk（w 0.6） | 38.22 | 43.23 | 23.17 | — | — | — |
| OW-DETR-RT | 2 | NMS 0.5（w 128） | 48.61 | 60.62 | 36.59 | 18.42 | 0.0347 | 7,286 |
| OW-DETR-RT | 2 | topk（w 32） | 50.03 | 62.31 | 37.74 | 15.21 | 0.0221 | 3,518 |
| OW-DETR-RT | 3 | NMS 0.5（w 128） | 41.90 | 49.79 | 26.12 | 22.19 | 0.0192 | 4,069 |
| OW-DETR-RT | 3 | topk（w 32） | 43.07 | 51.00 | 27.22 | 16.73 | 0.0108 | 2,583 |
| OW-DETR-RT | 4 | NMS 0.5（w 128） | 36.93 | 41.32 | 23.76 | — | — | — |
| OW-DETR-RT | 4 | topk（w 32） | 38.30 | 42.99 | 24.21 | — | — | — |
| RandBox-RT | 2 | NMS 0.5（w 256） | 35.30 | 43.99 | 26.61 | 13.67 | 0.0312 | 3,948 |
| RandBox-RT | 3 | NMS 0.5（w 256） | 29.54 | 34.94 | 18.73 | 16.73 | 0.0202 | 2,449 |

- RandBox-RT の Task 2 は、モデル自体は学習できている（段階 2 の学習中の評価で閉集合 Known 51.0、Prev 63.2）。Task 1 の w 256 が Task 2 のモデルには強すぎて、既知の検出が大量に未知へ回り Known が 35.3 に落ちている。Task 3 も同じ（Known 29.5、学習中の評価の閉集合 Known は 44.1）。
- RandBox-RT の Task 4 段階 2 は、2026-11-13 10:32（エポック 2 の後、評価の最初のバッチの前）から 4 時間 20 分止まった（CPU 100%・GPU 39% のまま、ログ出力なし）。14:52 に止めて OrthogonalDet-RT に進め、全部の後に段階 2 からやり直す（ログ `outputs/logs/run_baseline_tasks_randbox_t4.out`、止まったログは `randbox_rt_t4_ft.log_hung`）。
- RandBox-RT は Task 2〜4 の w を掃引し直す（`scripts/sweep_randbox_tasks.sh`、NMS 0.5、w 0 / 8 / 16 / 32 / 64 / 128 / 256、テスト全体、全部の学習の後に自動で実行）。選び方は Task 1 と同じ形で、tune 側で Known ≥（w 0 の Known − 3）の中で U-Recall 最大。
| OrthogonalDet-RT | 2 | NMS 0.5（w 8） | 35.76 | 44.84 | 26.68 | 12.91 | 0.0109 | 551 |
| OrthogonalDet-RT | 3 | NMS 0.5（w 8） | 31.97 | 36.48 | 22.95 | 15.98 | 0.0094 | 528 |

- OrthogonalDet-RT も RandBox-RT と同じく、Task 1 の w 8 で Known が落ちる（学習中の評価の閉集合 Known は 50.9 / 43.7）。w の掃引に OrthogonalDet-RT も加えた（w 0〜256）。
- OrthogonalDet-RT の Task 4 段階 2 も、2026-11-14 10:11（エポック 25 の後）から止まった。RandBox-RT と同じ止まり方で、どちらも SimOTA のマッチングを使う。`owod/zoo/simota_matcher.py` の dynamic-k の while ループが、費用に NaN があるとき（argmin が NaN を選ぶ）や GT がクエリより多いときに抜けられないので、NaN を大きな値に置き換え、Q 回で打ち切るように直した。14:42 に止めて CAT-RT に進め、RandBox-RT の Task 4 の後に段階 2 からやり直す（`outputs/logs/run_baseline_tasks_orthogonaldet_t4.out`）。

| 手法 | Task | 後処理 | Known AP50 | Prev | Cur | U-Recall | WI | A-OSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CAT-RT | 2 | NMS 0.7（w 16） | 50.06 | 61.97 | 38.16 | 18.43 | 0.0363 | 9,873 |
| CAT-RT | 2 | topk（w 2） | 50.36 | 62.20 | 38.51 | 8.81 | 0.0176 | 2,577 |
| CAT-RT | 3 | NMS 0.7（w 16） | 41.85 | 49.96 | 25.62 | 20.10 | 0.0197 | 4,058 |
| CAT-RT | 3 | topk（w 2） | 43.24 | 51.44 | 26.84 | 10.49 | 0.0099 | 1,577 |
| CAT-RT | 4 | NMS 0.7（w 16） | 38.50 | 43.08 | 24.77 | — | — | — |
| CAT-RT | 4 | topk（w 2） | 38.81 | 43.54 | 24.61 | — | — | — |
| RandBox-RT | 4 | NMS 0.5（w 256） | 26.47 | 28.70 | 19.79 | — | — | — |

- RandBox-RT の Task 4 は、SimOTA を直した後のやり直しで止まらずに終わった（2026-11-15 11:28）。

### RandBox-RT・OrthogonalDet-RT の w の掃引し直し（2026-11-15 16:45 終了、`scripts/sweep_randbox_tasks.sh`）

NMS 0.5、w 0〜256。選び方は tune 側で Known ≥（w 0 の Known − 3）の中で U-Recall 最大。Task 4 は未知の GT がないので Task 3 で選んだ w を使う。値はテスト全体（`outputs/{m}_t{t}/sweep_w_full.json`）。

| 手法 | Task | w | Known AP50 | Prev | Cur | U-Recall | WI | A-OSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RandBox-RT | 2 | 32 | 48.44 | 59.86 | 37.02 | 11.37 | 0.0190 | 9,069 |
| RandBox-RT | 3 | 32 | 41.32 | 48.66 | 26.65 | 13.69 | 0.0126 | 6,034 |
| RandBox-RT | 4 | 32 | 35.86 | 40.09 | 23.17 | — | — | — |
| OrthogonalDet-RT | 2 | 4 | 48.43 | 60.44 | 36.42 | 12.36 | 0.0116 | 1,687 |
| OrthogonalDet-RT | 3 | 4 | 42.59 | 50.42 | 26.95 | 14.94 | 0.0094 | 1,481 |
| OrthogonalDet-RT | 4 | 4 | 37.35 | 41.90 | 23.72 | — | — | — |

- w 0（未知を出さない）の Known は、RandBox-RT が 51.18 / 44.30 / 39.25、OrthogonalDet-RT が 50.94 / 43.82 / 39.11 で、UniQ-DETR（51.22 / 43.49 / 38.99）とほぼ同じ。
- どちらも、w を上げると U-Recall より先に Known が落ちる。w を最大にしても U-Recall は RandBox-RT 13.7 / 16.7、OrthogonalDet-RT 13.1 / 16.4 で、UniQ-DETR（20.71 / 23.80）に届かない。
- OrthogonalDet-RT の Task 4（Task 1 の w 8、学習のやり直し後）は 26.48 / 28.93 / 19.12。

## Hosoya et al. (IJCV 2025) metrics, full test (2026-11-16)

Re-evaluated every method at its reported setting with `scripts/run_eval_ap_range.sh` (results in `<ckpt dir>/sweep_ap.json`).
K_AP / U_AP = AP@[.5:.95] with COCO 101-point AP (11-point gave 1/11 = 9.09 to any unknown class whose top detection is
correct, which dominated U-AP); K2U = known GT not detected as its own class but covered by an unknown detection (IoU>0.5).
T2-T4 columns: Known AP (Prev / Cur). T4 U-AP is 0 by construction; K2U there counts all leftover unknown outputs.

| Method (setting) | T1 K-AP | T1 U-AP | T1 K2U | T2 K-AP (P/C) | T2 U-AP | T2 K2U | T3 K-AP (P/C) | T3 U-AP | T3 K2U | T4 K-AP (P/C) | T4 K2U |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UniQ-DETR w12 | 49.98 | 1.28 | 1524 | 37.79 (47.81/27.78) | 1.00 | 2623 | 30.87 (38.28/16.04) | 0.16 | 3684 | 27.14 (30.95/15.72) | 5075 |
| PROB-RT NMS w0.7 | 49.27 | 0.14 | 4116 | 36.92 (46.89/26.95) | 0.28 | 5035 | 29.92 (37.24/15.28) | 0.04 | 5664 | 26.67 (30.65/14.73) | 6735 |
| OW-DETR-RT topk w32 | 48.72 | 0.10 | 1956 | 37.18 (47.13/27.23) | 0.03 | 2740 | 30.26 (37.63/15.50) | 0.02 | 2421 | 26.42 (30.22/15.01) | 3953 |
| CAT-RT NMS w16 | 47.95 | 0.28 | 2525 | 36.41 (46.15/26.67) | 0.07 | 3434 | 29.50 (36.62/15.26) | 0.12 | 5213 | 26.44 (30.11/15.43) | 5766 |
| CAT-RT topk w2 | 47.79 | 0.07 | 1024 | 36.66 (46.39/26.94) | 0.02 | 1480 | 30.31 (37.57/15.79) | 0.09 | 1802 | 26.61 (30.37/15.33) | 2108 |
| RandBox-RT (T1 w256, T2-4 w32) | 46.48 | 0.53 | 2942 | 34.49 (42.99/26.00) | 0.89 | 4304 | 27.91 (34.55/14.62) | 0.11 | 4631 | 24.09 (27.38/14.20) | 6721 |
| OrthogonalDet-RT (T1 w8, T2-4 w4) | 49.37 | 1.77 | 888 | 34.28 (43.54/25.01) | 0.30 | 2123 | 28.98 (36.04/14.86) | 0.20 | 2247 | 24.84 (28.34/14.37) | 3191 |
