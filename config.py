# config.py

from dataclasses import dataclass

@dataclass
class Config:
    # 모델 설정
    model_type: str = "cnn"  # cnn/attention
    
    # 경로 설정
    data_x_product_path: str = "./DLResource/PreprocessData/data_X_product_5.csv"
    data_x_meta_path: str = "./DLResource/PreprocessData/data_X_5.csv"
    data_y_path: str = "./DLResource/PreprocessData/data_Y_5.csv"
    embeddings_path: str = "./DLResource/ev_final32.npy"
    products_path: str = "./DLResource/RawData/products.csv"
    model_save_path: str = "./models"
    model_save_name: str = "predictor.pth"

    # 데이터 관련
    N_orders: int = 5
    max_products_per_order: int = 145
    product_emb_dim: int = 32

    # CNN 관련
    cnn_out_channels: int = 256
    cnn_kernel_size: int = 5
    cnn_pool_size: int = 16

    # Attention 관련
    attn_out_dim: int = 128
    attn_heads: int = 4
    attn_layers: int = 2

    # GRU/FC
    gru_hidden_dim: int = 256
    gru_layers: int = 2
    dropout_rate: float = 0.2
    prediction_head_inter_dim: int = 256

    # 학습 관련
    learning_rate: float = 0.0005
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_patience: int = 20
    metrics_threshold: float = 0.4
    pos_weight: float = 15.0

    # 기타
    seed: int = 42
