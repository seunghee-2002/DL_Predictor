from dataclasses import dataclass

# 실험에 필요한 모든 설정값을 담는 구성 클래스
@dataclass
class Config:
    model_type: str = "cnn"    # 사용할 모델 선택: "cnn", "attention"

    # 파일 경로 설정
    data_x_product_path: str = "./DLResource/data_X_product_5.csv"  # 입력 제품 시퀀스
    data_y_path: str = "./DLResource/data_Y_5.csv"                  # 타겟 제품 정보
    embeddings_path: str = "./DLResource/ev_final64.npy"                           # 제품 임베딩 벡터

    model_save_path: str = "./models"                                              # 모델 저장 디렉토리
    model_save_name: str = "predictor.pth"                                         # 저장할 모델 파일명

    # 시퀀스 데이터 관련 설정
    N_orders: int = 5                        # 한 유저당 입력할 과거 주문 수
    max_products_per_order: int = 145       # 한 주문당 최대 제품 수
    product_emb_dim: int = 64               # 제품 임베딩 벡터 차원

    # CNN 모델 구조 설정
    cnn_out_channels: int = 128             # CNN 출력 채널 수
    cnn_kernel_size: int = 4                # CNN 커널 사이즈
    cnn_pool_size: int = 8                 # Adaptive MaxPool1d 출력 크기

    # Attention 모델 구조 설정
    attn_out_dim: int = 128                 # Attention Pooling 출력 차원
    attn_heads: int = 4                     # Multi-head attention 헤드 수
    attn_layers: int = 2                    # Transformer Encoder 층 수

    # GRU + FC 구조 설정 (공통)
    gru_hidden_dim: int = 128               # GRU hidden size
    gru_layers: int = 2                     # GRU 레이어 수
    dropout_rate: float = 0.2               # Dropout 비율
    prediction_head_inter_dim: int = 128    # FC 중간 hidden layer 크기

    # 학습 관련 하이퍼파라미터
    learning_rate: float = 0.001           # 학습률
    batch_size: int = 16                    # 배치 사이즈
    num_epochs: int = 100                   # 에폭 수
    early_stopping_patience: int = 20       # 검증 성능이 향상되지 않을 때 기다릴 최대 epoch 수 (조기 종료 기준)

    # 손실 함수 관련
    loss_type: str = "bce"           # 손실 함수 설정: bce / bce+bpr / bce+margin
    pos_weight: float = 10.0                # BCEWithLogitsLoss용 positive class weight
    ranking_margin: float = 0.2             # ranking 손실의 margin 값
    ranking_weight: float = 0.1             # bce+ranking일 때 ranking 비중
    ranking_sampling_ratio: float = 0.1     # negative 샘플 비율
    metrics_threshold: float = 0.4          # F1/Precision 계산 시 threshold

    # 랜덤 시드
    seed: int = 42
