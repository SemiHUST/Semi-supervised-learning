
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.flexmatch.utils import FlexMatchThresholdingHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import torch.nn.functional as F
from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


from .utils import FreeMatchThresholingHook

# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val
def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

@ALGORITHMS.register('freesequencematch')
class FreeSequenceMatch(AlgorithmBase):
    """
        SequenceMatch algorithm (https://arxiv.org/abs/2110.08263).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup,
            ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
    
    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        # sequencematch
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

        #freematch
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        # self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        # super().set_hooks()
        self.register_hook(FreeMatchThresholingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_m, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_m, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                feats_x_lb = outputs['feat'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_m, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(3)
                feats_x_ulb_w, feats_x_ulb_m, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(3)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']

                outs_x_lb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_lb_w['logits']
                feats_x_lb_w = outs_x_lb_w['feat']

                outs_x_ulb_m = self.model(x_ulb_m)
                logits_x_ulb_m = outs_x_ulb_m['logits']
                feats_x_ulb_m = outs_x_ulb_m['feat']

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w,'x_ulb_m': feats_x_ulb_m, 'x_ulb_s':feats_x_ulb_s}
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_ulb_m = self.compute_prob(logits_x_ulb_m.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())
                probs_x_ulb_m = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_m.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
        
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                                pseudo_label,
                                                'ce',
                                                mask=mask)

            unsup_loss_mw = F.kl_div(F.softmax(logits_x_ulb_m, dim=-1).log(),
                                    F.softmax(probs_x_ulb_w / self.T, dim=-1).detach(),
                                    reduction='none').sum(dim=1, keepdim=False)
            unsup_loss_mw = (unsup_loss_mw * mask).mean()
            
            unsup_loss_sm = F.kl_div(F.softmax(logits_x_ulb_s, dim=-1).log(),
                                    F.softmax(probs_x_ulb_m / self.T, dim=-1).detach(),
                                    reduction='none').sum(dim=1, keepdim=False)
            unsup_loss_sm = (unsup_loss_sm * mask).mean()
            
            unsup_loss_sw = F.kl_div(F.softmax(logits_x_ulb_s, dim=-1).log(),
                                    F.softmax(probs_x_ulb_w / self.T, dim=-1).detach(),
                                    reduction='none').sum(dim=1, keepdim=False)
            unsup_loss_sw = (unsup_loss_sw * mask).mean()
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            # ent_loss = 0.0
            total_loss = sup_loss + self.lambda_u * (unsup_loss + unsup_loss_mw + unsup_loss_sm + unsup_loss_sw) + self.lambda_e * ent_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                        unsup_loss=unsup_loss.item(), 
                                        total_loss=total_loss.item(), 
                                        util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        # save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        # save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()

        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        # self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        # self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        # self.print_fn("additional parameter loaded")

        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
        ]