import torch
import torch.nn as nn

# CNN + GRU 기반 제품 재구매 예측 모델
class CNNPredictorModel(nn.Module):
    def __init__(self,
                 num_total_products: int,          # 전체 제품 수 (출력 차원)
                 product_emb_dim: int,             # 제품 임베딩 차원
                 cnn_out_channels: int,            # CNN 출력 채널
                 cnn_kernel_size: int,             # CNN 커널 크기
                 gru_hidden_dim: int,              # GRU hidden 차원
                 gru_layers: int,                  # GRU 레이어 수
                 dropout_rate: float,              # FC dropout 비율
                 prediction_head_inter_dim: int,   # FC 중간 hidden 크기
                 pool_outsize: int):               # Adaptive pooling 출력 크기
        super().__init__()

        # 주문 내 제품 리스트 임베딩을 CNN으로 인코딩
        self.order_encoder_cnn = nn.Sequential(
            nn.Conv1d(product_emb_dim, cnn_out_channels, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=pool_outsize)
        )

        # CNN 출력을 GRU로 시퀀스 인코딩
        self.order_sequence_gru = nn.GRU(
            input_size=cnn_out_channels * pool_outsize,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )

        # 최종 예측 레이어: FC -> FC -> num_products
        self.prediction_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, prediction_head_inter_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prediction_head_inter_dim, num_total_products)
        )

    def forward(self, product_x):
        # product_x: (B, N, L, D)
        B, N, L, D = product_x.shape

        # CNN 인코딩
        cnn_input = product_x.view(B * N, L, D).permute(0, 2, 1)     # (B*N, D, L)
        encoded = self.order_encoder_cnn(cnn_input).flatten(start_dim=1)  # (B*N, C*pool_out)

        # GRU 인코딩
        encoded_seq = encoded.view(B, N, -1)                         # (B, N, C*pool_out)
        _, h_n = self.order_sequence_gru(encoded_seq)               # h_n: (num_layers, B, hidden)
        return self.prediction_head(h_n[-1])                        # 마지막 hidden만 사용
