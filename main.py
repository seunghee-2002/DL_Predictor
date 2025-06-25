import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from config import Config
from custom_dataset import OrderSequenceDataset, custom_collate_fn
from model_cnn import CNNPredictorModel
from model_attention import AttentionPredictorModel
from utils import get_device
from trainer import train_model

# 메인 실행 함수
def main():
    config = Config()

    # 랜덤 시드 고정
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 디바이스 설정
    device = get_device()

    # 데이터셋 로딩
    dataset = OrderSequenceDataset(
        data_x_product_path=config.data_x_product_path,
        data_y_path=config.data_y_path,
        embeddings_npy_path=config.embeddings_path,
        N_orders=config.N_orders,
        max_products_per_order=config.max_products_per_order,
        embedding_dim=config.product_emb_dim
    )

    num_total_products = dataset.num_total_products

    # 데이터 분할
    total_samples = len(dataset)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)
    test_size = total_samples - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 모델 선택
    if config.model_type == "cnn":
        model = CNNPredictorModel(
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
        model_class = CNNPredictorModel

    elif config.model_type == "attention":
        model = AttentionPredictorModel(
            num_total_products=num_total_products,
            product_emb_dim=config.product_emb_dim,
            attn_out_dim=config.attn_out_dim,
            attn_heads=config.attn_heads,
            attn_layers=config.attn_layers,
            gru_hidden_dim=config.gru_hidden_dim,
            gru_layers=config.gru_layers,
            dropout_rate=config.dropout_rate,
            prediction_head_inter_dim=config.prediction_head_inter_dim
        ).to(device)
        model_class = AttentionPredictorModel

    else:
        raise ValueError(f"알 수 없는 모델 타입: {config.model_type}")

    # 손실 함수 및 옵티마이저 정의
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 학습 시작
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=config,
        device=device,
        num_total_products=num_total_products,
        model_class=model_class
    )

if __name__ == "__main__":
    main()
