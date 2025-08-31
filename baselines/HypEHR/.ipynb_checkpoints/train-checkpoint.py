import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, precision_recall_fscore_support
import numpy as np
import pandas as pd

PHENO_ORDER = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and related",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Pneumonia",
    "Septicemia (except in labor)",
]

def train_with_early_stopping(model, 
                              train_dataloader, 
                              val_dataloader, 
                              test_dataloader,
                              optimizer, 
                              loss_fn, 
                              device, 
                              args,
                              val_long_seq_idx = None,
                              test_long_seq_idx = None,
                              task_type="binary", 
                              eval_metric="f1"):
    best_score = 0.
    best_val_metric = None
    best_test_metric = None
    best_test_long_seq_metric = None
    epochs_no_improve = 0

    for epoch in range(1, 1 + args["epochs"]):
        model.train()
        ave_loss = 0.

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training Batches")):
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

            labels = batch[-1].float()
            output_logits = model(*batch[:-1])
            loss = loss_fn(output_logits.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ave_loss += loss.item()
            

        ave_loss /= (step + 1)

        # Evaluation
        val_metric, val_long_seq_metric = evaluate(model, val_dataloader, device, task_type=task_type, long_seq_idx=val_long_seq_idx)
        test_metric, test_long_seq_metric = evaluate(model, test_dataloader, device, task_type=task_type, long_seq_idx=test_long_seq_idx)

        if task_type != "binary":
            val_per_class_df = val_metric["per_class"]
            val_metric = val_metric["global"]
            test_per_class_df = test_metric["per_class"]
            test_metric = test_metric["global"]
            
            if val_long_seq_idx != None:
                val_long_seq_per_class_df = val_long_seq_metric["per_class"]
                val_long_seq_metric = val_long_seq_metric["global"]
            if test_long_seq_idx != None:
                test_long_seq_per_class_df = test_long_seq_metric["per_class"]
                test_long_seq_metric = test_long_seq_metric["global"]

        # Logging
        print(f"Epoch: {epoch:03d}, Average Loss: {ave_loss:.4f}")
        print(f"Validation: {val_metric}")
        print(f"Test:       {test_metric}")

        # Check for improvement
        current_score = val_metric[eval_metric]
        if current_score > best_score:
            best_score = current_score
            best_val_metric = val_metric if task_type == "binary" else {"global": val_metric, "per_class": val_per_class_df}
            best_test_metric = test_metric if task_type == "binary" else {"global": test_metric, "per_class": test_per_class_df}
            best_test_long_seq_metric = test_long_seq_metric if task_type == "binary" else {"global": test_long_seq_metric, "per_class": test_long_seq_per_class_df}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve >= args["early_stop_patience"]:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args['early_stop_patience']} epochs).")
            break

    print("\nBest validation performance:")
    print(best_val_metric)
    print("Corresponding test performance:")
    print(best_test_metric)
    if best_test_long_seq_metric is not None:
        print("Corresponding test-long performance:")
        print(best_test_long_seq_metric)
    return best_test_metric, best_test_long_seq_metric


@torch.no_grad()
def evaluate(model, dataloader, device, long_seq_idx=None, task_type="binary"):
    model.eval()
    predicted_scores, gt_labels = [], []

    # 推理：收集 logits 与 labels
    for _, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        labels = batch[-1]
        output_logits = model(*batch[:-1])
        predicted_scores.append(output_logits)
        gt_labels.append(labels)

    if task_type == "binary":
        # —— 标准二分类评估 —— #
        logits_all = torch.cat(predicted_scores, dim=0).view(-1)          # logits [N]
        labels_all = torch.cat(gt_labels, dim=0).view(-1).cpu().numpy()    # y_true [N]
        scores_all = logits_all.cpu().numpy()                              # 连续分数（logits）
        ypred_all  = (logits_all > 0).float().cpu().numpy()                # logits > 0

        tp = (ypred_all * labels_all).sum()
        precision = tp / (ypred_all.sum() + 1e-8)
        recall    = tp / (labels_all.sum() + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        roc_auc   = roc_auc_score(labels_all, scores_all)
        prec_curve, rec_curve, _ = precision_recall_curve(labels_all, scores_all)
        pr_auc    = auc(rec_curve, prec_curve)

        all_performance = {"precision": float(precision),
                           "recall": float(recall),
                           "f1": float(f1),
                           "auc": float(roc_auc),
                           "prauc": float(pr_auc)}

        subset_performance = None
        if long_seq_idx is not None:
            idx = torch.as_tensor(long_seq_idx, device=logits_all.device, dtype=torch.long)
            logits_sub = logits_all.index_select(0, idx).view(-1)
            labels_sub = torch.as_tensor(labels_all, device=logits_all.device)[idx].cpu().numpy()
            scores_sub = logits_sub.cpu().numpy()
            ypred_sub  = (logits_sub > 0).float().cpu().numpy()

            tp = (ypred_sub * labels_sub).sum()
            precision = tp / (ypred_sub.sum() + 1e-8)
            recall    = tp / (labels_sub.sum() + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            roc_auc   = roc_auc_score(labels_sub, scores_sub)
            prec_curve, rec_curve, _ = precision_recall_curve(labels_sub, scores_sub)
            pr_auc    = auc(rec_curve, prec_curve)

            subset_performance = {"precision": float(precision),
                                  "recall": float(recall),
                                  "f1": float(f1),
                                  "auc": float(roc_auc),
                                  "prauc": float(pr_auc)}

        return all_performance, subset_performance

    else:
        # —— Multi-label evaluation（按类聚合） —— #
        logits_all = torch.cat(predicted_scores, dim=0)    # [B, C]
        labels_all_t = torch.cat(gt_labels, dim=0)         # [B, C]

        def _compute_metrics(logits_sub, labels_sub):
            # 连续分数（概率）：sigmoid(logits)，CPU + fp16 先升为 fp32
            if logits_sub.device.type == "cpu" and logits_sub.dtype == torch.float16:
                prob_t = torch.sigmoid(logits_sub.float())
            else:
                prob_t = torch.sigmoid(logits_sub)
            # 二值化：logits > 0
            ypred_t = (logits_sub > 0).to(torch.int32)

            y_true = labels_sub.cpu().numpy().astype(np.int32)       # [N, C]
            y_pred = ypred_t.cpu().numpy().astype(np.int32)          # [N, C]
            scores = prob_t.cpu().numpy()                             # [N, C]

            # per-class P/R/F1
            p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            # per-class AUC / PR-AUC
            C = y_true.shape[1]
            aucs, praucs = [], []
            for c in range(C):
                yt, ys = y_true[:, c], scores[:, c]
                if yt.max() == yt.min():
                    aucs.append(np.nan)
                    praucs.append(np.nan)
                else:
                    aucs.append(roc_auc_score(yt, ys))
                    prec_curve, rec_curve, _ = precision_recall_curve(yt, ys)
                    praucs.append(auc(rec_curve, prec_curve))

            # 宏平均（忽略 NaN）
            summary = {
                "precision": float(np.mean(p_cls)),
                "recall":    float(np.mean(r_cls)),
                "f1":        float(np.mean(f1_cls)),
                "auc":       float(np.nanmean(aucs)) if np.any(~np.isnan(aucs)) else float("nan"),
                "prauc":     float(np.nanmean(praucs)) if np.any(~np.isnan(praucs)) else float("nan"),
            }

            per_class_df = pd.DataFrame({
                "precision": p_cls,
                "recall":    r_cls,
                "f1":        f1_cls,
                "auc":       aucs,
                "prauc":     praucs,
            }, index=PHENO_ORDER)   # 确保 PHENO_ORDER 已定义且长度=C

            return {"global": summary, "per_class": per_class_df}

        # 全量
        all_performance = _compute_metrics(logits_all, labels_all_t)

        # 子集
        subset_performance = None
        if long_seq_idx is not None:
            idx = torch.as_tensor(long_seq_idx, device=logits_all.device, dtype=torch.long)
            subset_performance = _compute_metrics(
                logits_all.index_select(0, idx),
                labels_all_t.index_select(0, idx)
            )

        return all_performance, subset_performance