"""Detection solver with the open-world (M-OWODB) evaluation protocol."""

import copy
import datetime
import json
import time

import torch

from src.misc import MetricLogger, dist_utils
from src.solver.det_engine import train_one_epoch
from src.solver.det_solver import DetSolver

from ..eval.owod_eval import OWODEvaluator


@torch.no_grad()
def evaluate_owod(model, postprocessor, data_loader, evaluator: OWODEvaluator, device):
    model.eval()
    evaluator.cleanup()
    # detector health: the same outputs scored closed-set (p(c) only, no unknown output). Separates
    # "detector is bad" from "unknown scores push known detections out of the top-100".
    closed_pp, closed_ev = None, None
    if getattr(postprocessor, 'unk_score', 'none') != 'none':
        closed_pp = copy.copy(postprocessor)
        closed_pp.unk_score, closed_pp.known_score = 'none', 'cls'
        if postprocessor.unk_score == 'cls_last':
            closed_pp.unk_score = 'cls_last_closed'
        closed_ev = copy.copy(evaluator)
        closed_ev.cleanup()
    metric_logger = MetricLogger(delimiter="  ")
    obj_sum, obj_hi, obj_n = 0., 0., 0
    for samples, targets in metric_logger.log_every(data_loader, 50, 'Test:'):
        samples = samples.to(device)
        outputs = model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        results = postprocessor(outputs, orig_sizes)
        evaluator.update({t['image_id'].item(): r for t, r in zip(targets, results)})
        if closed_pp is not None:
            closed_ev.update({t['image_id'].item(): r for t, r in zip(targets, closed_pp(outputs, orig_sizes))})
        if 'pred_obj' in outputs:
            # saturation monitor (plan §9.2). Sigma is re-estimated from matched queries, so matched
            # queries sit at d2/D ~ 1 and o(q) <= ~exp(-tau); an absolute o>0.9 test never fires.
            # Saturation = (almost) every query falls inside the foreground shell d2/D <= 1.
            d2n = outputs['pred_obj'].float() / outputs.get('obj_dim', 256)
            obj_sum += outputs['pred_objectness'].float().sum().item()
            obj_hi += (d2n <= 1.0).float().sum().item()
            obj_n += d2n.numel()
    evaluator.synchronize_between_processes()
    if closed_ev is not None:
        closed_ev.synchronize_between_processes()
    stats = {}
    if dist_utils.is_main_process():
        evaluator.accumulate()
        stats = evaluator.summarize()
        if closed_ev is not None:
            closed_ev.accumulate()
            stats['closed_K_AP50'] = closed_ev.summarize(verbose=False)['K_AP50']
            print(f"[OWOD] closed-set Known AP50 (p(c) only, no unknown) {stats['closed_K_AP50']:.2f}")
        if obj_n:
            stats['obj_mean'] = obj_sum / obj_n
            stats['obj_frac_in_fg_shell'] = obj_hi / obj_n
            print(f"[OWOD] objectness mean {stats['obj_mean']:.4f} | "
                  f"frac(d2/D<=1) {stats['obj_frac_in_fg_shell']:.4f}  (->1.0 means saturation)")
    return stats


class OWDetSolver(DetSolver):

    def _build_ow_evaluator(self):
        ecfg = self.cfg.yaml_cfg['owod_eval']
        return OWODEvaluator(ecfg['gt_json'], num_known=ecfg.get('num_known', 20))

    def _evaluate(self):
        module = self.ema.module if self.ema else self.model
        return evaluate_owod(module, self.postprocessor, self.val_dataloader, self.ow_evaluator, self.device)

    def fit(self):
        print("Start training")
        self.train()
        self.ow_evaluator = self._build_ow_evaluator()
        args = self.cfg
        eval_freq = self.cfg.yaml_cfg.get('eval_freq', 1)
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'number of trainable parameters: {n_parameters}')

        best = {'epoch': -1, 'K_AP50': -1.}
        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                max_norm=args.clip_max_norm, print_freq=args.print_freq, ema=self.ema, scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler, writer=self.writer)

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            self.last_epoch += 1

            if self.output_dir:
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')
                if (epoch + 1) % args.checkpoint_freq == 0:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / f'checkpoint{epoch:04}.pth')

            test_stats = {}
            if (epoch + 1) % eval_freq == 0 or epoch == args.epoches - 1:
                test_stats = self._evaluate()
                if dist_utils.is_main_process() and test_stats:
                    if self.writer:
                        for k, v in test_stats.items():
                            if isinstance(v, (int, float)):
                                self.writer.add_scalar(f'Test/{k}', v, epoch)
                    if test_stats['K_AP50'] > best['K_AP50']:
                        best = {'epoch': epoch, 'K_AP50': test_stats['K_AP50'], 'U_R50': test_stats['U_R50']}
                        if self.output_dir:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
                print(f'best_stat: {best}')

            if self.output_dir and dist_utils.is_main_process():
                log = {**{f'train_{k}': v for k, v in train_stats.items()},
                       **{f'test_{k}': v for k, v in test_stats.items()},
                       'epoch': epoch, 'n_parameters': n_parameters}
                with (self.output_dir / 'log.txt').open('a') as f:
                    f.write(json.dumps(log) + '\n')

        print('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))

    def val(self):
        self.eval()
        self.ow_evaluator = self._build_ow_evaluator()
        stats = self._evaluate()
        if self.output_dir and dist_utils.is_main_process():
            pp = self.postprocessor
            name = f"eval_{getattr(pp, 'unk_score', 'na')}.json"
            with (self.output_dir / name).open('w') as f:
                json.dump(stats, f, indent=1)
        return stats
