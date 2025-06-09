import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from custom_dataset import OrderSequenceDataset, custom_collate_fn
# from model_attention import AttentionPredictorModel
from model_cnn import CNNPredictorModel

import os
import argparse
import numpy as np
import pandas as pd
import torchmetrics

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA 장치를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("CPU 장치를 사용합니다.")
    return device

def save_model(model, path):
    print(f"모델을 {path} 에 저장합니다.")
    torch.save(model.state_dict(), path)

def load_model(model_architecture_instance, path, device):
    print(f"{path} 에서 모델 가중치를 불러옵니다.")
    if not os.path.exists(path):
        print(f"경고: 모델 파일({path})을 찾을 수 없습니다. 새로운 모델로 학습을 시작합니다.")
        return model_architecture_instance 
    try:
        model_architecture_instance.load_state_dict(torch.load(path, map_location=device))
        model_architecture_instance.to(device)
        model_architecture_instance.eval()
        print(f"모델 가중치 로드 완료: {path}")
    except Exception as e:
        print(f"경고: 모델 가중치 로드 중 오류 발생 ({e}). 새로운 모델로 학습을 시작합니다.")
    return model_architecture_instance

def initialize_metrics(num_classes, device, top_k_list=[10, 20], threshold=0.1):
    metrics = {
        'F1_micro_overall': torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='micro', threshold=threshold).to(device),
        'F1_macro': torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro', threshold=threshold).to(device),
    }
    for k in top_k_list:
        metrics[f'Precision@{k}'] = torchmetrics.Precision(task="multilabel", num_labels=num_classes, top_k=k, threshold=threshold).to(device)
        metrics[f'Recall@{k}'] = torchmetrics.Recall(task="multilabel", num_labels=num_classes, top_k=k, threshold=threshold).to(device)
    return metrics

def update_metrics(metrics, preds_proba, targets_int):
    for metric_name, metric_val in metrics.items():
        metric_val.update(preds_proba, targets_int)

def compute_metrics(metrics):
    results = {}
    for metric_name, metric_val in metrics.items():
        results[metric_name] = metric_val.compute().item()
    return results

def reset_metrics(metrics):
    for metric_val in metrics.values():
        metric_val.reset()

def evaluate_model(model, data_loader, criterion, device, metrics_dict, desc="[평가]"):
    model.eval()
    running_loss = 0.0
    reset_metrics(metrics_dict) 
    
    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    with torch.no_grad():
        for products_x, meta_x, targets in progress_bar:
            products_x = products_x.to(device)
            meta_x = meta_x.to(device)
            targets = targets.to(device)
            
            logits = model(products_x, meta_x)
            loss = criterion(logits, targets)
            running_loss += loss.item() * products_x.size(0)
            
            preds_proba = torch.sigmoid(logits)
            targets_int = targets.to(torch.int)
            update_metrics(metrics_dict, preds_proba, targets_int)
            
            progress_bar.set_postfix(loss=loss.item())
            
    epoch_loss = running_loss / len(data_loader.dataset)
    computed_metrics_results = compute_metrics(metrics_dict)

    return epoch_loss, computed_metrics_results

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, device, 
                model_save_path, model_save_name, args_for_reload, num_total_products_for_reload, 
                order_meta_dim, early_stopping_patience=10, metrics_threshold_to_use=0.1):
    best_val_loss = float('inf')
    epochs_no_improve = 0 
    metrics_val = initialize_metrics(num_total_products_for_reload, device, threshold=metrics_threshold_to_use)
    metrics_test = initialize_metrics(num_total_products_for_reload, device, threshold=metrics_threshold_to_use)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"디렉토리 생성: {model_save_path}")

    full_model_path = os.path.join(model_save_path, model_save_name)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [학습]", leave=False)
        for products_x, meta_x, targets in train_progress_bar:
            products_x = products_x.to(device)
            meta_x = meta_x.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(products_x, meta_x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * products_x.size(0)
            train_progress_bar.set_postfix(loss=loss.item())
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_val_loss, val_metrics_results = evaluate_model(model, val_loader, criterion, device, metrics_val, desc=f"Epoch {epoch+1}/{num_epochs} [검증]")
        metrics_log_str = ", ".join([f"{name}: {value:.4f}" for name, value in val_metrics_results.items()])
        print(f"Epoch {epoch+1}/{num_epochs}, 학습 손실: {epoch_train_loss:.4f}, 검증 손실: {epoch_val_loss:.4f}, 검증 지표: [{metrics_log_str}]")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_model(model, full_model_path)
            print(f"검증 손실 개선: {best_val_loss:.4f}. 모델 저장됨: {full_model_path}")
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1
            print(f"검증 손실 개선 없음. ({epochs_no_improve}/{early_stopping_patience})")

        if epochs_no_improve >= early_stopping_patience:
            print(f"{early_stopping_patience} 에포크 동안 검증 손실 개선 없어 학습 조기 종료.")
            break 
    
    print("학습 완료.")
    print(f"최적 검증 손실: {best_val_loss:.4f}")
    print(f"최적 모델은 {full_model_path} 에 저장되었습니다.")

    if os.path.exists(full_model_path):
        print("\n최적 모델 로드 및 테스트 세트 평가 중...")
        # Attention 기반
        # best_model_instance = AttentionPredictorModel(
        #     num_total_products=num_total_products_for_reload,
        #     product_emb_dim=args_for_reload.product_emb_dim,
        #     attn_out_dim=args_for_reload.attn_out_dim,
        #     attn_heads=args_for_reload.attn_heads,
        #     attn_layers=args_for_reload.attn_layers,
        #     gru_hidden_dim=args_for_reload.gru_hidden_dim,
        #     gru_layers=args_for_reload.gru_layers,
        #     dropout_rate=args_for_reload.dropout_rate,
        #     prediction_head_inter_dim=args_for_reload.prediction_head_inter_dim,
        #     order_meta_dim=order_meta_dim
        # ).to(device)
        # CNN 기반
        best_model_instance = CNNPredictorModel(
            num_total_products=num_total_products_for_reload,
            product_emb_dim=args_for_reload.product_emb_dim,
            cnn_out_channels=args_for_reload.cnn_out_channels,
            cnn_kernel_size=args_for_reload.cnn_kernel_size,
            gru_hidden_dim=args_for_reload.gru_hidden_dim,
            gru_layers=args_for_reload.gru_layers,
            dropout_rate=args_for_reload.dropout_rate,
            prediction_head_inter_dim=args_for_reload.prediction_head_inter_dim,
            order_meta_dim=order_meta_dim
        ).to(device)
        best_model_instance = load_model(best_model_instance, full_model_path, device)
        test_loss, test_metrics_results = evaluate_model(best_model_instance, test_loader, criterion, device, metrics_test, desc="[최종 테스트 평가]")
        test_metrics_log_str = ", ".join([f"{name}: {value:.4f}" for name, value in test_metrics_results.items()])
        print(f"최종 테스트 세트 손실: {test_loss:.4f}, 테스트 지표: [{test_metrics_log_str}]")
    else:
        print(f"오류: 최적 모델 파일({full_model_path})을 찾을 수 없어 테스트 평가를 스킵합니다.")

def main():
    parser = argparse.ArgumentParser(description="Attention/CNN-ProductSequence 모델 학습 스크립트")
    # 경로
    parser.add_argument("--data_x_product_path", type=str, default="./DLResource/PreprocessData/data_X_product_5.csv", help="전체 X_product_N.csv 파일 경로")
    parser.add_argument("--data_x_meta_path", type=str, default="./DLResource/PreprocessData/data_X_5.csv", help="주문 헤더 feature 데이터 경로")
    parser.add_argument("--data_y_path", type=str, default="./DLResource/PreprocessData/data_Y_5.csv", help="전체 Y_N.csv 파일 경로")
    parser.add_argument("--embeddings_path", type=str, default="./DLResource/ev_final32.npy", help="제품 임베딩 파일 경로")
    parser.add_argument("--products_path", type=str, default="./DLResource/RawData/products.csv", help="제품 메타데이터 파일 경로")
    # 데이터 인자 설정
    parser.add_argument("--N_orders", type=int, default=5, help="주문 시퀀스 길이")
    parser.add_argument("--max_products_per_order", type=int, default=145, help="주문당 최대 제품 수")
    parser.add_argument("--product_emb_dim", type=int, default=32, help="제품 임베딩 차원")
    # CNN
    parser.add_argument("--cnn_out_channels", type=int, default=128, help="CNN 출력 차원")
    parser.add_argument("--cnn_kernel_size", type=int, default=4, help="CNN 커널 크기")
    # Attention
    parser.add_argument("--attn_out_dim", type=int, default=128, help="Attention 출력 차원")
    parser.add_argument("--attn_heads", type=int, default=4, help="Attention 헤드 수")
    parser.add_argument("--attn_layers", type=int, default=1, help="Attention block 층 수")
    # GRU
    parser.add_argument("--gru_hidden_dim", type=int, default=256, help="GRU 은닉 상태 차원")
    parser.add_argument("--gru_layers", type=int, default=2, help="GRU 계층 수")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="드롭아웃 비율")
    # FC
    parser.add_argument("--prediction_head_inter_dim", type=int, default=256, help="예측 헤드 중간 차원")
    # 학습 관련 인자
    parser.add_argument("--learning_rate", type=float, default=0.001, help="학습률")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--num_epochs", type=int, default=20, help="에포크 수")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--metrics_threshold", type=float, default=0.4, help="평가지표 계산 시 사용할 확률 임계값")
    parser.add_argument("--pos_weight", type=float, default=10, help="클래스 불균형 보정을 위한 pos_weight")
    # 모델
    parser.add_argument("--model_save_path", type=str, default="./models", help="모델 저장 경로")
    parser.add_argument("--model_save_name", type=str, default="predictor.pth", help="저장할 모델 파일 이름")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device()

    print("데이터셋 불러오는중...")
    # 데이터셋 준비
    train_dataset = OrderSequenceDataset(
        data_x_product_path=args.data_x_product_path,
        data_x_meta_path=args.data_x_meta_path,
        data_y_path=args.data_y_path,
        embeddings_npy_path=args.embeddings_path,
        products_path=args.products_path,
        N_orders=args.N_orders,
        max_products_per_order=args.max_products_per_order,
        embedding_dim=args.product_emb_dim
    )
    order_meta_dim = train_dataset.meta_feature_dim
    num_total_products_for_model = train_dataset.num_total_products

    total_samples = len(train_dataset)
    train_samples = int(0.6 * total_samples)
    val_samples = int(0.2 * total_samples)
    test_samples = total_samples - train_samples - val_samples
    train_set, val_set, test_set = random_split(
        train_dataset, 
        [train_samples, val_samples, test_samples],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print("데이터 로딩 및 분할 완료.")

    # 모델 선택(Attention or CNN, 한 줄만 주석 바꿔서 전환)
    # model = AttentionPredictorModel(
    #     num_total_products=num_total_products_for_model,
    #     product_emb_dim=args.product_emb_dim,
    #     attn_out_dim=args.attn_out_dim,
    #     attn_heads=args.attn_heads,
    #     attn_layers=args.attn_layers,
    #     gru_hidden_dim=args.gru_hidden_dim,
    #     gru_layers=args.gru_layers,
    #     dropout_rate=args.dropout_rate,
    #     prediction_head_inter_dim=args.prediction_head_inter_dim,
    #     order_meta_dim=order_meta_dim
    # ).to(device)
    model = CNNPredictorModel(
        num_total_products=num_total_products_for_model,
        product_emb_dim=args.product_emb_dim,
        cnn_out_channels=args.cnn_out_channels,
        cnn_kernel_size=args.cnn_kernel_size,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        dropout_rate=args.dropout_rate,
        prediction_head_inter_dim=args.prediction_head_inter_dim,
        order_meta_dim=order_meta_dim
    ).to(device)

    print("모델 초기화 완료.")

    # 손실 함수 정의
    if args.pos_weight is not None:
        pos_weight_tensor = torch.tensor([args.pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(f"[INFO] pos_weight 사용: {args.pos_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"[INFO] pos_weight 없이 BCEWithLogitsLoss 사용")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("모델 학습 시작...")
    train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        args.num_epochs,
        device,
        args.model_save_path,
        args.model_save_name,
        args,
        num_total_products_for_model,
        order_meta_dim,
        args.early_stopping_patience,
        args.metrics_threshold
    )

if __name__ == '__main__':
    main()
