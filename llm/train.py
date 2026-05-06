import time
import torch
import torch.nn.functional as F


def calc_loss_batch(x, y, model, device):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


def calc_loss_loader(loader, model, device, num_batches=None):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            total += calc_loss_batch(x, y, model, device).item()
            n += 1
    model.train()
    return total / max(n, 1)


def train(model, train_loader, val_loader, optimizer, device,
          num_epochs=1, eval_every=100, eval_batches=20,
          clip_grad=1.0, use_amp=True, on_eval=None):
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))
    history = {"step": [], "train_loss": [], "val_loss": []}
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % eval_every == 0:
                tr = calc_loss_loader(train_loader, model, device, eval_batches)
                vl = calc_loss_loader(val_loader, model, device, eval_batches)
                history["step"].append(step)
                history["train_loss"].append(tr)
                history["val_loss"].append(vl)
                print(f"[epoch {epoch+1} step {step:6d} {time.time()-t0:.0f}s] train={tr:.3f} val={vl:.3f}")
                if on_eval is not None:
                    on_eval(step, model)
    return history
