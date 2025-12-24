import torch
import torch.nn.functional as F
from typing import List, Optional

def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 4.0):
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)

def mse_embed_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor):
    return F.mse_loss(student_emb, teacher_emb)

def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor):
    return F.cross_entropy(student_logits, labels)

def combine_losses(student_logits_list: List[torch.Tensor],
                   student_emb_list: List[torch.Tensor],
                   teacher_logits_ens: Optional[torch.Tensor],
                   teacher_emb_ens: torch.Tensor,
                   labels: torch.Tensor,
                   T: float, w_ce: float, w_kd: float, w_mse: float):
    
    logits_agg = torch.stack(student_logits_list, dim=0).mean(0)
    loss_ce_val = ce_loss(logits_agg, labels)

    loss_kd_val = 0.0
    if w_kd > 0 and teacher_logits_ens is not None:
        for sl in student_logits_list:
            loss_kd_val += kd_kl_loss(sl, teacher_logits_ens, T=T)
        loss_kd_val /= len(student_logits_list)

    loss_mse_val = 0.0
    if w_mse > 0:
        for se in student_emb_list:
            loss_mse_val += mse_embed_loss(se, teacher_emb_ens)
        loss_mse_val /= len(student_emb_list)

    total = (w_ce * loss_ce_val) + (w_kd * loss_kd_val) + (w_mse * loss_mse_val)
    parts = {'ce': loss_ce_val.item(), 'kd': loss_kd_val.item() if w_kd > 0 else 0, 'mse': loss_mse_val.item() if w_mse > 0 else 0}
    return total, parts