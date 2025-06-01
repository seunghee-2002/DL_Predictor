import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from custom_dataset import OrderSequenceDataset, custom_collate_fn
from model import OrderPredictorModel
import os
import argparse
import numpy as np
import pandas as pd
import torchmetrics # torchmetrics 임포트

def get_device():
    """CUDA 사용 가능 시 GPU, 아닐 경우 CPU 장치를 반환합니다."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA 장치를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("CPU 장치를 사용합니다.")
    return device

def save_model(model, path):
    """모델의 state_dict를 지정된 경로에 저장합니다."""
    print(f"모델을 {path} 에 저장합니다.")
    torch.save(model.state_dict(), path)

def load_model(model_architecture_instance, path, device):
    """지정된 경로에서 모델의 state_dict를 불러오고, 모델을 평가 모드로 설정합니다."""
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


def initialize_metrics(num_classes, device, top_k_list=[10, 20], threshold=0.1): # threshold 인자 추가 및 기본값 설정
    print(f"알림: 평가지표가 threshold={threshold} 로 초기화됩니다.") # 확인용 로그 추가
    metrics = {
        'F1_micro_overall': torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='micro', threshold=threshold).to(device),
        'F1_macro': torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro', threshold=threshold).to(device),
    }
    for k in top_k_list:
        metrics[f'Precision@{k}'] = torchmetrics.Precision(task="multilabel", num_labels=num_classes, top_k=k, threshold=threshold).to(device)
        metrics[f'Recall@{k}'] = torchmetrics.Recall(task="multilabel", num_labels=num_classes, top_k=k, threshold=threshold).to(device)
    return metrics

def update_metrics(metrics, preds_proba, targets_int):
    """계산된 예측과 타겟으로 모든 지표를 업데이트합니다."""
    for metric_name, metric_val in metrics.items():
        metric_val.update(preds_proba, targets_int)

def compute_metrics(metrics):
    """업데이트된 모든 지표의 최종 값을 계산합니다."""
    results = {}
    for metric_name, metric_val in metrics.items():
        results[metric_name] = metric_val.compute().item()
    return results

def reset_metrics(metrics):
    """다음 에포크/평가를 위해 모든 지표를 리셋합니다."""
    for metric_val in metrics.values():
        metric_val.reset()

def evaluate_model(model, data_loader, criterion, device, metrics_dict, desc="[평가]"):
    """주어진 데이터로더로 모델을 평가하고 손실 및 추가 지표를 반환합니다."""
    model.eval()
    running_loss = 0.0
    reset_metrics(metrics_dict) 
    
    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device) 
            
            logits = model(inputs) 
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)
            
            preds_proba = torch.sigmoid(logits) 
            targets_int = targets.to(torch.int) 
            update_metrics(metrics_dict, preds_proba, targets_int)
            
            progress_bar.set_postfix(loss=loss.item())
            
    epoch_loss = running_loss / len(data_loader.dataset)
    computed_metrics_results = compute_metrics(metrics_dict)
    
    return epoch_loss, computed_metrics_results

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, device, 
                model_save_path, model_save_name, args_for_reload, num_total_products_for_reload, 
                early_stopping_patience=10, metrics_threshold_to_use=0.1):
    """모델 학습, 검증, 최적 모델 저장, Early Stopping 및 테스트 평가를 수행합니다."""
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
        
        for inputs, targets in train_progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
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
        best_model_instance = OrderPredictorModel(
            num_total_products=num_total_products_for_reload,
            product_emb_dim=args_for_reload.product_emb_dim,
            cnn_out_channels=args_for_reload.cnn_out_channels,
            cnn_kernel_size=args_for_reload.cnn_kernel_size,
            gru_hidden_dim=args_for_reload.gru_hidden_dim,
            gru_layers=args_for_reload.gru_layers,
            dropout_rate=args_for_reload.dropout_rate,
            prediction_head_inter_dim=args_for_reload.prediction_head_inter_dim
        ).to(device)

        best_model_instance = load_model(best_model_instance, full_model_path, device)
        test_loss, test_metrics_results = evaluate_model(best_model_instance, test_loader, criterion, device, metrics_test, desc="[최종 테스트 평가]")
        
        test_metrics_log_str = ", ".join([f"{name}: {value:.4f}" for name, value in test_metrics_results.items()])
        print(f"최종 테스트 세트 손실: {test_loss:.4f}, 테스트 지표: [{test_metrics_log_str}]")
    else:
        print(f"오류: 최적 모델 파일({full_model_path})을 찾을 수 없어 테스트 평가를 스킵합니다.")


def main():
    parser = argparse.ArgumentParser(description="CNN-GRU-CNN 모델 학습 스크립트")
    # 데이터 경로
    parser.add_argument("--data_x_product_path", type=str, default="./DLResource/PreprocessData/data_X_product_5.csv", help="전체 X_product_N.csv 파일 경로")
    parser.add_argument("--data_y_path", type=str, default="./DLResource/PreprocessData/data_Y_5.csv", help="전체 Y_N.csv 파일 경로")
    parser.add_argument("--embeddings_path", type=str, default="./DLResource/ev_final64.npy", help="제품 임베딩 파일 경로")
    parser.add_argument("--products_path", type=str, default="./DLResource/RawData/products.csv", help="제품 메타데이터 파일 경로")
    # 고정 인자
    parser.add_argument("--N_orders", type=int, default=5, help="GRU 입력 시퀀스 길이")
    parser.add_argument("--max_products_per_order", type=int, default=145, help="주문 당 최대 제품 수")
    parser.add_argument("--product_emb_dim", type=int, default=64, help="제품 임베딩 차원")
    # 하이퍼파라미터
    parser.add_argument("--cnn_out_channels", type=int, default=256, help="CNN 출력 채널 수")
    parser.add_argument("--cnn_kernel_size", type=int, default=5, help="CNN 커널 크기")
    parser.add_argument("--gru_hidden_dim", type=int, default=512, help="GRU 은닉 상태 차원")
    parser.add_argument("--gru_layers", type=int, default=2, help="GRU 계층 수")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="드롭아웃 비율")
    parser.add_argument("--prediction_head_inter_dim", type=int, default=256, help="예측 헤드 중간 차원")
    # 학습 관련 인자
    parser.add_argument("--learning_rate", type=float, default=0.001, help="학습률")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--num_epochs", type=int, default=0, help="에포크 수")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping을 위한 patience 값")
    parser.add_argument("--metrics_threshold", type=float, default=0.1, help="평가지표 계산 시 사용할 확률 임계값")
    # pos_weight() 관련
    parser.add_argument("--calculate_pos_weight_dynamically",default=False, help="학습 데이터셋에서 pos_weight 동적 계산 여부(manual_pos_weight가 우선)")
    parser.add_argument("--manual_pos_weight", type=float, default=100, help="BCEWithLogitsLoss에 수동으로 설정할 pos_weight 값 (예: 100.0)")
    # 데이터 분할 관련
    parser.add_argument("--train_ratio", type=float, default=0.6, help="학습 세트 비율")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="검증 세트 비율")
    # 모델 외부 관련
    parser.add_argument("--model_save_path", type=str, default="./models", help="모델 저장 경로")
    parser.add_argument("--model_save_name", type=str, default="cnn_gru_cnn.pth", help="저장할 모델 파일 이름")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    args = parser.parse_args()

    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and (args.train_ratio + args.val_ratio) < 1):
        raise ValueError("train_ratio와 val_ratio는 0과 1 사이여야 하며, 이들의 합도 1보다 작아야 합니다.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()

    if not os.path.exists(args.products_path):
        raise FileNotFoundError(f"제품 파일({args.products_path})을 찾을 수 없습니다.")
    products_df = pd.read_csv(args.products_path)
    num_total_products_from_file = len(products_df)
    print(f"'{args.products_path}' 파일에서 확인된 총 제품 수: {num_total_products_from_file}")

    print("전체 데이터셋 로딩 중 (분할 예정)...")
    full_dataset = OrderSequenceDataset(
        data_x_product_path=args.data_x_product_path,
        data_y_path=args.data_y_path,
        embeddings_npy_path=args.embeddings_path,
        products_path=args.products_path,
        N_orders=args.N_orders,
        max_products_per_order=args.max_products_per_order,
        embedding_dim=args.product_emb_dim 
    )
    
    num_total_products_for_model = full_dataset.num_total_products 
    if num_total_products_for_model != num_total_products_from_file:
        print(f"경고: products.csv ({num_total_products_from_file}개)와 "
              f"dataset.num_total_products ({num_total_products_for_model}개)의 제품 수가 일치하지 않습니다! "
              f"데이터셋의 값을 사용합니다.")

    total_samples = len(full_dataset)
    train_samples = int(args.train_ratio * total_samples)
    val_samples = int(args.val_ratio * total_samples)
    test_samples = total_samples - train_samples - val_samples

    if train_samples <= 0 or val_samples <= 0 or test_samples <= 0:
        raise ValueError(f"데이터셋 크기가 너무 작거나 분할 비율이 잘못되어 하나 이상의 세트(학습:{train_samples}, 검증:{val_samples}, 테스트:{test_samples})가 0개 또는 음수의 샘플을 갖게 됩니다.")

    print(f"데이터셋 분할: 총 {total_samples}개 -> 학습 {train_samples}개, 검증 {val_samples}개, 테스트 {test_samples}개")
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_samples, val_samples, test_samples],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    print("데이터 로딩 및 분할 완료.")

    model = OrderPredictorModel(
        num_total_products=num_total_products_for_model, 
        product_emb_dim=args.product_emb_dim,
        cnn_out_channels=args.cnn_out_channels,
        cnn_kernel_size=args.cnn_kernel_size,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        dropout_rate=args.dropout_rate,
        prediction_head_inter_dim=args.prediction_head_inter_dim
    ).to(device)
    
    print("모델 초기화 완료.")

    # --- pos_weight 계산 및 손실 함수 초기화 ---
    pos_weight_tensor = None
    if args.manual_pos_weight is not None:
        pos_weight_tensor = torch.tensor([args.manual_pos_weight], device=device)
        print(f"수동으로 설정된 pos_weight: {args.manual_pos_weight:.4f}")
    elif args.calculate_pos_weight_dynamically: # 수동 설정 없고, 동적 계산 플래그 True일 때
        print("학습 데이터 로더를 사용하여 pos_weight 계산 중...")
        num_positives = 0
        num_negatives = 0

        for _, targets_batch in tqdm(train_loader, desc="pos_weight 계산 중인 배치"):
            num_positives += torch.sum(targets_batch == 1).item()
            num_negatives += torch.sum(targets_batch == 0).item()

        if num_positives > 0:
            pos_weight_value = num_negatives / num_positives
            pos_weight_value = min(pos_weight_value, 5000.0)
            pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
            print(f"계산된 pos_weight: {pos_weight_value:.4f}")
        else:
            print("경고: 학습 데이터에서 Positive 샘플을 찾을 수 없습니다. pos_weight를 사용하지 않습니다.")
    
    if pos_weight_tensor is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print("BCEWithLogitsLoss에 pos_weight 적용 완료.")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("BCEWithLogitsLoss 기본 설정 사용 (pos_weight 없음).")
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("옵티마이저 초기화 완료.")
    
    print("손실 함수 (BCEWithLogitsLoss) 및 옵티마이저 초기화 완료.")

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
        args.early_stopping_patience,
        args.metrics_threshold
    )

if __name__ == '__main__':
    main()