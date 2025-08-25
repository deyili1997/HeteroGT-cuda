import torch
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from contextlib import nullcontext


def train_with_early_stopping(model, train_dataloader, val_dataloader, test_dataloader,
                              optimizer, loss_fn, device, early_stop_patience, task_type, epochs, dec_loss_lambda = 0, 
                              val_long_seq_idx=None, test_long_seq_idx=None, eval_metric="prauc", return_model=False):

    # ---- 设备与AMP开关 ----
    device_type = device.type  # 'cuda' | 'cpu' | 'mps'
    use_amp = (device_type == "cuda")   # 仅在 CUDA 上启用 AMP/GradScaler，避免 CPU/MPS 警告
    scaler = GradScaler(enabled=use_amp)

    best_score = 0.0
    best_val_metric = None
    best_test_metric = None
    best_model_state = deepcopy(model.state_dict())
    epochs_no_improve = 0

    # 选择合适的 autocast 上下文（CPU/MPS 用 nullcontext，或手动设 enabled=False）
    amp_ctx = (autocast() if use_amp else nullcontext())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch:03d}"
        )

        for step, batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            # 移到目标设备
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            labels = batch[-1].float()

            try:
                with amp_ctx:
                    preds, dec_loss = model(*batch[:-1])
                    task_loss = loss_fn(preds.view(-1), labels.view(-1))
                    loss = task_loss + dec_loss_lambda * dec_loss

                if use_amp:
                    # AMP 路径
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # 反缩放后再裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # FP32 路径（CPU/MPS）
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                running_loss += loss.item()
                num_steps = step + 1
                progress_bar.set_postfix({"loss": f"{running_loss / num_steps:.4f}"})

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    print(f"[OOM Warning] Skipping batch {step} due to OOM.")
                    if device_type == "cuda":
                        torch.cuda.empty_cache()
                    elif device_type == "mps":
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    continue
                else:
                    raise

        epoch_loss = running_loss / max(1, (step + 1))
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        # 验证 + 早停
        (best_score, best_val_metric, best_test_metric, best_model_state,
         epochs_no_improve, early_stop_triggered) = evaluate_and_early_stop(
            model, val_dataloader, test_dataloader, device, task_type,
            val_long_seq_idx, test_long_seq_idx, eval_metric,
            best_score, best_val_metric, best_test_metric, best_model_state,
            epochs_no_improve, early_stop_patience
        )
        if early_stop_triggered:
            break

    print("\nBest validation performance:")
    print(best_val_metric)
    print("Corresponding test performance:")
    print(best_test_metric)

    model.load_state_dict(best_model_state)
    return (best_test_metric, model) if return_model else best_test_metric


def evaluate_and_early_stop(model, val_dataloader, test_dataloader, device, task_type,
                                  val_long_seq_idx, test_long_seq_idx, eval_metric,
                                  best_score, best_val_metric, best_test_metric, best_model_state,
                                  epochs_no_improve, early_stop_patience):
    """
    执行模型在验证集和测试集的评估，并进行早停检查。
    返回：
        - best_score
        - best_val_metric
        - best_test_metric
        - best_model_state
        - epochs_no_improve
        - early_stop_triggered (bool)
    """
    # --- Evaluation ---
    if val_long_seq_idx is not None:
        val_metric, val_long_seq_metric = evaluate(model, val_dataloader, device, task_type, val_long_seq_idx)
    else:
        val_metric = evaluate(model, val_dataloader, device, task_type)
        val_long_seq_metric = None

    if test_long_seq_idx is not None:
        test_metric, test_long_seq_metric = evaluate(model, test_dataloader, device, task_type, test_long_seq_idx)
    else:
        test_metric = evaluate(model, test_dataloader, device, task_type)
        test_long_seq_metric = None

    print(f"Validation: {val_metric}")
    if val_long_seq_metric is not None:
        print(f"Validation-long: {val_long_seq_metric}")

    print(f"Test:      {test_metric}")
    if test_long_seq_metric is not None:
        print(f"Test-long: {test_long_seq_metric}")

    # --- Early Stopping ---
    current_score = val_metric[eval_metric]
    early_stop_triggered = False

    if current_score > best_score:
        best_score = current_score
        best_val_metric = val_metric
        best_test_metric = test_metric
        best_model_state = deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"\nEarly stopping triggered (no improvement for {early_stop_patience} epochs).")
            early_stop_triggered = True

    return best_score, best_val_metric, best_test_metric, best_model_state, epochs_no_improve, early_stop_triggered


def run_multilabel_metrics(predictions, labels):
    # Multi-label classification: predictions [B, C], labels [B, C]
    f1s, aucs, praucs, precisions, recalls = [], [], [], [], []

    for i in range(predictions.size(0)):
        pred_i = predictions[i].clone()
        label_i = labels[i].clone()

        pred_i = (pred_i > 0).float().numpy()
        label_i = label_i.float().numpy()

        tp = (pred_i * label_i).sum()
        precision = tp / (pred_i.sum() + 1e-8)
        recall = tp / (label_i.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        try:
            auc_score = roc_auc_score(label_i, pred_i)
        except ValueError:
            auc_score = np.nan  # skip if only one class present

        prec_curve, rec_curve, _ = precision_recall_curve(label_i, pred_i)
        pr_auc_score = auc(rec_curve, prec_curve)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc_score)
        praucs.append(pr_auc_score)

    return {
        "precision": np.nanmean(precisions),
        "recall": np.nanmean(recalls),
        "f1": np.nanmean(f1s),
        "auc": np.nanmean(aucs),
        "prauc": np.nanmean(praucs),
    }


def run_binary_metrics(predictions, labels):
    predictions = predictions.view(-1)
    labels = labels.view(-1).float()
    scores = predictions.numpy()
    binary_preds = (predictions > 0).float().numpy()  # logit > 0 ≈ prob > 0.5

    tp = (binary_preds * labels.numpy()).sum()
    precision = tp / (binary_preds.sum() + 1e-8)
    recall = tp / (labels.sum().item() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    rocauc = roc_auc_score(labels.numpy(), scores)
    prec_curve, rec_curve, _ = precision_recall_curve(labels.numpy(), scores)
    prauc = auc(rec_curve, prec_curve)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": rocauc,
        "prauc": prauc,
    }

@torch.no_grad()
def evaluate(model, dataloader, device, task_type, long_seq_idx=None):
    model.eval()

    # 仅在 CUDA 上启用 autocast，避免 CPU/MPS 警告
    device_type = device.type  # 'cuda' | 'cpu' | 'mps'
    use_amp = (device_type == "cuda")
    amp_ctx = autocast() if use_amp else nullcontext()

    all_preds, all_labels = [], []

    for _, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        # move to device
        batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        labels = batch[-1]

        with amp_ctx:
            output = model(*batch[:-1])

        # 兼容 tensor / tuple / list
        preds = output[0] if isinstance(output, (tuple, list)) else output

        all_preds.append(preds)
        all_labels.append(labels)

    predictions = torch.cat(all_preds, dim=0).detach().cpu()
    labels = torch.cat(all_labels, dim=0).detach().cpu()

    # 若提供 long_seq_idx，确保可用于张量索引
    def _select_long_seq(t):
        if long_seq_idx is None:
            return None
        if isinstance(long_seq_idx, torch.Tensor):
            idx = long_seq_idx
        else:
            idx = torch.as_tensor(long_seq_idx, dtype=torch.long)
        return t[idx]

    if task_type == "binary":
        results = run_binary_metrics(predictions, labels)
        if long_seq_idx is not None:
            long_seq_results = run_binary_metrics(
                _select_long_seq(predictions), _select_long_seq(labels)
            )
            return results, long_seq_results
        else:
            return results
    else:
        results = run_multilabel_metrics(predictions, labels)
        if long_seq_idx is not None:
            long_seq_results = run_multilabel_metrics(
                _select_long_seq(predictions), _select_long_seq(labels)
            )
            return results, long_seq_results
        else:
            return results