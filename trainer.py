import os
import csv
from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss, MarginRankingLoss
from utils import initialize_metrics, reset_metrics, update_metrics, compute_metrics, save_model, load_model, plot_training_log

# BPR 손실 함수 정의
def bpr_loss(pos_scores, neg_scores):
    diff = torch.sigmoid(pos_scores - neg_scores)
    diff = torch.clamp(diff, min=1e-8, max=1.0)  # 수치 안정성 확보
    return -torch.mean(torch.log(diff))

# 손실 계산 함수
def compute_loss(logits, targets, config, device, loss_type):
    bce_criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight], device=device))

    # 먼저 BCE 단독인 경우 바로 처리
    if loss_type == "bce":
        return bce_criterion(logits, targets)

    # positive/negative 마스크 추출
    pos_mask = targets == 1
    neg_mask = targets == 0
    pos_scores = logits[pos_mask]
    neg_scores = logits[neg_mask]

    # pairwise 연산 가능한 경우에만 진행
    if pos_scores.size(0) > 0 and neg_scores.size(0) > 0:
        num_pairs = min(pos_scores.size(0), neg_scores.size(0))
        pos_sampled = pos_scores[:num_pairs]

        # 랜덤 인덱스 방식으로 negative 샘플링 (속도 개선)
        neg_indices = torch.randint(0, neg_scores.size(0), (num_pairs,), device=device)
        neg_sampled = neg_scores[neg_indices]

        if loss_type == "bpr":
            return bpr_loss(pos_sampled, neg_sampled)

        elif loss_type == "margin":
            margin_criterion = MarginRankingLoss(margin=config.ranking_margin)
            return margin_criterion(pos_sampled, neg_sampled, torch.ones(num_pairs, device=device))

        elif loss_type == "bce+bpr":
            loss_bce = bce_criterion(logits, targets)
            loss_rank = bpr_loss(pos_sampled, neg_sampled)
            return loss_bce + config.ranking_weight * loss_rank

        elif loss_type == "bce+margin":
            loss_bce = bce_criterion(logits, targets)
            margin_criterion = MarginRankingLoss(margin=config.ranking_margin)
            loss_rank = margin_criterion(pos_sampled, neg_sampled, torch.ones(num_pairs, device=device))
            return loss_bce + config.ranking_weight * loss_rank

        else:
            raise ValueError(f"알 수 없는 loss_type: {loss_type}")

    else:
        # positive나 negative가 하나도 없으면 fallback으로 BCE 사용
        return bce_criterion(logits, targets)

# 평가 함수
def evaluate_model(model, data_loader, device, metrics_dict, config, loss_type):
    model.eval()
    running_loss = 0.0
    reset_metrics(metrics_dict)

    with torch.no_grad():
        for products_x, targets in tqdm(data_loader, desc="[평가]", leave=False):
            products_x, targets = products_x.to(device), targets.to(device)
            logits = model(products_x)
            loss = compute_loss(logits, targets, config, device, loss_type)
            running_loss += loss.item() * products_x.size(0)
            preds_proba = torch.sigmoid(logits)
            update_metrics(metrics_dict, preds_proba, targets.to(torch.int))

    return running_loss / len(data_loader.dataset), compute_metrics(metrics_dict)

# 전체 학습 루프
def train_model(model, train_loader, val_loader, test_loader,
                optimizer, config, device,
                num_total_products, model_class):

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

        for products_x, targets in tqdm(train_loader, desc=f"[학습] Epoch {epoch+1}", leave=False):
            products_x, targets = products_x.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(products_x)
            loss = compute_loss(logits, targets, config, device, config.loss_type)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * products_x.size(0)

        val_loss, val_metrics = evaluate_model(model, val_loader, device, metrics_val, config, config.loss_type)
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
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("학습 완료. 최고 F1-micro:", best_val_f1)

    if os.path.exists(full_model_path):
        print("최적 모델 로드 및 테스트 평가...")
        best_model = model_class(
            num_total_products=num_total_products,
            product_emb_dim=config.product_emb_dim,
            cnn_out_channels=config.cnn_out_channels,
            cnn_kernel_size=config.cnn_kernel_size,
            gru_hidden_dim=config.gru_hidden_dim,
            gru_layers=config.gru_layers,
            dropout_rate=config.dropout_rate,
            prediction_head_inter_dim=config.prediction_head_inter_dim,
            pool_outsize=config.cnn_pool_size
        ).to(device)
        best_model = load_model(best_model, full_model_path, device)
        test_loss, test_metrics = evaluate_model(best_model, test_loader, device, metrics_test, config, config.loss_type)
        print("테스트 결과:", test_metrics)
    else:
        print("[오류] 최적 모델 파일 없음")

    plot_training_log(log_path)
