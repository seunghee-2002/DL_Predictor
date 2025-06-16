import os
import torch
import torchmetrics
import pandas as pd
import matplotlib.pyplot as plt

# CUDA 또는 CPU 디바이스 설정
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'CUDA' if device.type == 'cuda' else 'CPU'} 장치를 사용합니다.")
    return device

# 모델 저장
def save_model(model, path):
    print(f"모델을 {path} 에 저장합니다.")
    torch.save(model.state_dict(), path)

# 모델 불러오기
def load_model(model_architecture_instance, path, device):
    print(f"{path} 에서 모델 가중치를 불러옵니다.")
    if not os.path.exists(path):
        print(f"[경고] 모델 파일 없음. 새로 시작합니다.")
        return model_architecture_instance
    try:
        model_architecture_instance.load_state_dict(torch.load(path, map_location=device))
        model_architecture_instance.to(device)
        model_architecture_instance.eval()
        print(f"[완료] 모델 로드 완료")
    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
    return model_architecture_instance

# 다중 클래스용 metric 초기화
def initialize_metrics(num_classes, device, top_k_list=[10, 20], threshold=0.1):
    metrics = {
        'F1_micro_overall': torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='micro', threshold=threshold).to(device),
        'F1_macro': torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro', threshold=threshold).to(device),
    }
    for k in top_k_list:
        metrics[f'Precision@{k}'] = torchmetrics.Precision(task="multilabel", num_labels=num_classes, top_k=k, threshold=threshold).to(device)
        metrics[f'Recall@{k}'] = torchmetrics.Recall(task="multilabel", num_labels=num_classes, top_k=k, threshold=threshold).to(device)
    return metrics

# 메트릭 업데이트
def update_metrics(metrics, preds_proba, targets_int):
    for metric in metrics.values():
        metric.update(preds_proba, targets_int)

# 메트릭 계산
def compute_metrics(metrics):
    return {name: metric.compute().item() for name, metric in metrics.items()}

# 메트릭 초기화
def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset()

# 학습 로그 시각화
def plot_training_log(log_path):
    if not os.path.exists(log_path):
        print(f"[Log 파일 없음] {log_path}")
        return

    df = pd.read_csv(log_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df.index + 1, df["val_f1_micro"], label="Val F1-micro", marker='o')
    plt.plot(df.index + 1, df["val_precision@10"], label="Val Precision@10", linestyle='--')
    plt.plot(df.index + 1, df["val_recall@10"], label="Val Recall@10", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_path.replace(".csv", ".png"))
    # plt.show()
