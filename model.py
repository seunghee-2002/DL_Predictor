import torch
import torch.nn as nn

class OrderPredictorModel(nn.Module):
    def __init__(self,
                 num_total_products: int,
                 product_emb_dim: int = 64,
                 cnn_out_channels: int = 128,
                 cnn_kernel_size: int = 3,
                 gru_hidden_dim: int = 256,
                 gru_layers: int = 1,
                 dropout_rate: float = 0.1,
                 prediction_head_inter_dim: int = 128):
        super(OrderPredictorModel, self).__init__()

        self.product_emb_dim = product_emb_dim
        self.cnn_out_channels = cnn_out_channels
        self.gru_hidden_dim = gru_hidden_dim
        self.num_total_products = num_total_products # 전체 제품 수

        # 1. 초기 CNN (OrderEncoderCNN)
        self.order_encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.product_emb_dim,
                      out_channels=self.cnn_out_channels,
                      kernel_size=cnn_kernel_size, # 인자로 받은 커널 크기 사용
                      padding=(cnn_kernel_size - 1) // 2), # 커널 크기에 따른 동적 패딩
            nn.ReLU(),
            nn.Dropout(dropout_rate), # 드롭아웃 레이어 추가
            nn.AdaptiveMaxPool1d(output_size=1) # 각 주문의 제품 시퀀스를 단일 벡터로 압축
        )

        # 2. GRU (OrderSequenceGRU)
        self.order_sequence_gru = nn.GRU(
            input_size=self.cnn_out_channels,
            hidden_size=self.gru_hidden_dim,
            num_layers=gru_layers, # 인자로 받은 GRU 레이어 수 사용
            batch_first=True, # 입력 및 출력 텐서의 첫 번째 차원이 배치 크기임을 명시
            dropout=dropout_rate if gru_layers > 1 else 0 # GRU 레이어가 2개 이상일 때만 드롭아웃 적용
        )

        # 3. 최종 FC (PredictionHead) - 다중 레이블 분류를 위한 수정
        self.prediction_head = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, prediction_head_inter_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prediction_head_inter_dim, self.num_total_products)
        )

    def forward(self, x):
        # x shape: (batch_size, N_orders, max_products_per_order, product_emb_dim)
        batch_size, N_orders, max_products_per_order, _ = x.shape

        # CNN 입력을 위해 (batch_size * N_orders, max_products_per_order, product_emb_dim) 형태로 변환
        # 그 후 (batch_size * N_orders, product_emb_dim, max_products_per_order) 형태로 permute
        cnn_input = x.view(batch_size * N_orders, max_products_per_order, self.product_emb_dim)
        cnn_input = cnn_input.permute(0, 2, 1) # (B * N, D_emb, L_max_prod)

        # CNN 통과
        # cnn_output shape: (batch_size * N_orders, cnn_out_channels, 1)
        order_vectors_squeezed = self.order_encoder_cnn(cnn_input)
        # order_vectors shape: (batch_size * N_orders, cnn_out_channels)
        order_vectors = order_vectors_squeezed.squeeze(-1)

        # GRU 입력을 위해 (batch_size, N_orders, cnn_out_channels) 형태로 변환
        gru_input = order_vectors.view(batch_size, N_orders, self.cnn_out_channels)

        # GRU 통과
        # gru_output shape: (batch_size, N_orders, gru_hidden_dim)
        # h_n shape: (num_layers, batch_size, gru_hidden_dim)
        _, h_n = self.order_sequence_gru(gru_input) # gru_output은 사용하지 않음

        # 마지막 GRU 레이어의 은닉 상태 사용
        # h_n[-1] shape: (batch_size, gru_hidden_dim)
        gru_final_state = h_n[-1] 

        # Prediction Head 통과
        # logits shape: (batch_size, num_total_products)
        logits = self.prediction_head(gru_final_state)
        
        return logits
