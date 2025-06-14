# trainer.py

import os
import csv
from tqdm import tqdm
import torch

from utils import (
    initialize_metrics,
    reset_metrics,
    update_metrics,
    compute_metrics,
    save_model,
    load_model,
    plot_training_log,
)


def evaluate_model(model, data_loader, criterion, device, metrics_dict, desc="[평가]"):
    model.eval()
    running_loss = 0.0
    reset_metrics(metrics_dict)

    with torch.no_grad():
        for products_x, meta_x, targets in tqdm(data_loader, desc=desc, leave=False):
            products_x, meta_x, targets = products_x.to(device), meta_x.to(device), targets.to(device)
            logits = model(products_x, meta_x)
            loss = criterion(logits, targets)
            running_loss += loss.item() * products_x.size(0)

            preds_proba = torch.sigmoid(logits)
            update_metrics(metrics_dict, preds_proba, targets.to(torch.int))

    epoch_loss = running_loss / len(data_loader.dataset)
    computed_metrics = compute_metrics(metrics_dict)
    return epoch_loss, computed_metrics


def train_model(
    model, train_loader, val_loader, test_loader,
    optimizer, criterion, config, device,
    num_total_products, order_meta_dim,
    model_class  # CNNPredictorModel or AttentionPredictorModel
):
    best_val_f1 = 0.0
    epochs_no_improve = 0

    metrics_val = initialize_metrics(num_total_products, device, threshold=config.metrics_threshold)
    metrics_test = initialize_metrics(num_total_products, device, threshold=config.metrics_threshold)

    os.makedirs(config.model_save_path, exist_ok=True)
    full_model_path = os.path.join(config.model_save_path, config.model_save_name)
    log_path = full_model_path.replace(".pth", "_train_log.csv")

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(["val_f1_micro", "val_precision@10", "val_recall@10"])

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0

        for products_x, meta_x, targets in tqdm(train_loader, desc=f"[학습] Epoch {epoch+1}", leave=False):
            products_x, meta_x, targets = products_x.to(device), meta_x.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(products_x, meta_x), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * products_x.size(0)

        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device, metrics_val, desc=f"[검증] Epoch {epoch+1}")
        val_f1 = val_metrics.get("F1_micro_overall", 0.0)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                round(val_f1, 4),
                round(val_metrics.get("Precision@10", 0.0), 4),
                round(val_metrics.get("Recall@10", 0.0), 4),
            ])

        print(f"Epoch {epoch+1}: Val F1-micro={val_f1:.4f}, Loss={val_loss:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(model, full_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("학습 완료. 최고 F1-micro:", best_val_f1)

    if os.path.exists(full_model_path):
        print("\n최적 모델 로드 및 테스트 평가...")
        best_model = model_class(
            num_total_products=num_total_products,
            product_emb_dim=config.product_emb_dim,
            cnn_out_channels=config.cnn_out_channels,
            cnn_kernel_size=config.cnn_kernel_size,
            gru_hidden_dim=config.gru_hidden_dim,
            gru_layers=config.gru_layers,
            dropout_rate=config.dropout_rate,
            prediction_head_inter_dim=config.prediction_head_inter_dim,
            order_meta_dim=order_meta_dim,
            pool_outsize=config.cnn_pool_size
        ).to(device)
        best_model = load_model(best_model, full_model_path, device)
        test_loss, test_metrics = evaluate_model(best_model, test_loader, criterion, device, metrics_test, desc="[최종 테스트]")
        print("테스트 결과:", test_metrics)
    else:
        print("[오류] 최적 모델을 찾을 수 없음")

    plot_training_log(log_path)
