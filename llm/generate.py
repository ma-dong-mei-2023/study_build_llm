import torch


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_length,
             temperature=1.0, top_k=None, top_p=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)[:, -1, :]
        logits = logits / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True)
            next_id_local = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, next_id_local)
        else:
            next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
