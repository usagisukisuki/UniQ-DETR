"""Write the M-OWODB Task 2-4 configs of the baselines (README, Tasks 2-4), same protocol as UniQ-DETR
(configs/uniq_t{t}_{train,ft}.yml, analysis/TASKS_PLAN.md):
  stage 1  configs/baselines/tasks/{m}_t{t}_train.yml  t{t}_train (Task t's 20 classes only), 10 epochs, lr 1e-5 (1e-6)
  stage 2  configs/baselines/tasks/{m}_t{t}_ft.yml     t{t}_ft exemplars (all known), 60 / 60 / 70 epochs, lr 1e-4 (1e-5)
                                                       with a 0.1x drop after 40 / 35 / 50 epochs
No pseudo-unknown files and no exclusion of the previous classes: each baseline keeps its own unknown signal, as in
the original Task 2-4 training (PROB, OW-DETR, CAT, RandBox, OrthogonalDet). Online pseudo-labelling starts at epoch 0.
Inference keeps the Task-1 settings of README (tools/sweep_pp.py on the whole test set).

  uv run python -m scripts.make_baseline_task_configs
"""
import os

EPOCHS = {2: (60, 40), 3: (60, 35), 4: (70, 50)}
# name: (Task-1 config, number of classifier rows as a function of the known classes, extra stage-1 lines)
METHODS = {
    'prob_rt': ('../../b_prob_rt.yml', lambda k: k, ''),
    'owdetr_rt': ('../owdetr_rt.yml', lambda k: k + 1, 'OWCriterion:\n  attn_pseudo_start: 0\n'),
    'cat_rt': ('../cat_rt.yml', lambda k: k + 1, 'OWCriterion:\n  attn_pseudo_start: 0\n'),
    'randbox_rt': ('../randbox_rt.yml', lambda k: k + 1, 'OWCriterion:\n  randbox_unk_start: 0\n'),
    'orthogonaldet_rt': ('../orthogonaldet_rt.yml', lambda k: 81, 'OWPostProcessor:\n  known_slots: {K}\n'),
}

OPT = '''optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm).*$'
      lr: {blr}
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
      weight_decay: 0.
  lr: {lr}
  betas: [0.9, 0.999]
  weight_decay: 0.0001
'''


def ss(m, t, kind):
    return f'    ss_file: data/mowodb/t{t}_{kind}_ss.json\n' if m == 'cat_rt' else ''


def main():
    out = 'configs/baselines/tasks'
    os.makedirs(out, exist_ok=True)
    for m, (base, rows, extra) in METHODS.items():
        for t in (2, 3, 4):
            K, P = 20 * t, 20 * (t - 1)
            ep, drop = EPOCHS[t]
            train = (
                f'# {m}, M-OWODB Task {t} stage 1 (scripts/make_baseline_task_configs.py): from the Task-{t - 1} model with the\n'
                f'# classifier grown to {K} known classes (tools/expand_classes.py), 10 epochs on owod_t{t}_train labelled with\n'
                f'# Task {t}\'s 20 classes only, lr 0.1x.\n'
                f"__include__: ['{base}']\n"
                f'output_dir: ./outputs/{m}_t{t}/train\n'
                f'num_classes: {rows(K)}\n'
                f'epoches: 10\n'
                f'owod_eval:\n  num_known: {K}\n  prev_known: {P}\n'
                + extra.format(K=K)
                + OPT.format(blr='0.000001', lr='0.00001')
                + f'\ntrain_dataloader:\n  dataset:\n    ann_file: data/mowodb/t{t}_train.json\n    num_known: {K}\n'
                + ss(m, t, 'train')
                + '    transforms:\n      policy:\n        epoch: 8\n  collate_fn:\n    stop_epoch: 8\n'
                f'val_dataloader:\n  dataset:\n    num_known: {K}\n')
            ft = (
                f'# {m}, M-OWODB Task {t} stage 2 (scripts/make_baseline_task_configs.py): fine-tuning of stage 1 on ORE\'s\n'
                f'# exemplar set owod_t{t}_ft, all {K} known classes labelled, {ep} epochs, 0.1x lr after {drop} (PROB t{t}_ft).\n'
                f"__include__: ['./{m}_t{t}_train.yml']\n"
                f'output_dir: ./outputs/{m}_t{t}/ft\n'
                f'epoches: {ep}\n'
                + OPT.format(blr='0.00001', lr='0.0001')
                + f'\nlr_scheduler:\n  type: MultiStepLR\n  milestones: [{drop}]\n  gamma: 0.1\n'
                'lr_warmup_scheduler:\n  type: LinearWarmup\n  warmup_duration: 500\n'
                f'train_dataloader:\n  dataset:\n    ann_file: data/mowodb/t{t}_ft.json\n'
                + ss(m, t, 'ft')
                + f'    transforms:\n      policy:\n        epoch: {ep - 2}\n  collate_fn:\n    stop_epoch: {ep - 2}\n')
            for kind, text in (('train', train), ('ft', ft)):
                with open(os.path.join(out, f'{m}_t{t}_{kind}.yml'), 'w') as f:
                    f.write(text)
    print('wrote', out)


if __name__ == '__main__':
    main()
