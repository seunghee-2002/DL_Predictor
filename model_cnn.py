import torch
import torch.nn as nn

class CNNPredictorModel(nn.Module):
    def __init__(self,
                 num_total_products: int = 49688, # 전체 예측 대상 제품 수 (output class 수)
                 product_emb_dim: int = 64,      # 제품 임베딩 벡터의 차원
                 cnn_out_channels: int = 128,    # CNN을 통과한 뒤의 벡터 차원(특징 추출 후)
                 cnn_kernel_size: int = 3,       # CNN 커널 크기(제품 간 관계 탐색 범위)
                 gru_hidden_dim: int = 256,      # GRU의 히든 상태 벡터 크기
                 gru_layers: int = 1,            # GRU 계층 수
                 dropout_rate: float = 0.1,      # FC, GRU 등에서의 dropout 확률
                 prediction_head_inter_dim: int = 128, # 최종 FC 레이어 중간 크기
                 order_meta_dim: int = 5,         # 주문 feature 벡터의 차원 (data_X_5.csv 기준)
                 pool_outsize: int = 8
                 ):
        super().__init__()
        self.product_emb_dim = product_emb_dim
        self.cnn_out_channels = cnn_out_channels
        self.gru_hidden_dim = gru_hidden_dim
        self.num_total_products = num_total_products
        self.order_meta_dim = order_meta_dim

        # 주문 내 제품 임베딩 시퀀스를 1개의 벡터로 압축하는 CNN 블록
        self.order_encoder_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=self.product_emb_dim,      # 입력: 제품 임베딩 차원
                out_channels=self.cnn_out_channels,    # 출력: CNN 특징 벡터 차원
                kernel_size=cnn_kernel_size,
                padding=(cnn_kernel_size - 1) // 2    # 시퀀스 길이 보존용 패딩
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=pool_outsize)       # (길이 차원) 평균/최대 Pool → 1개 벡터로
        )
        # 여러 주문(시퀀스)들을 시간축 따라 처리하는 GRU (입력: 제품+주문메타 concat)
        self.order_sequence_gru = nn.GRU(
            input_size=self.cnn_out_channels * pool_outsize + order_meta_dim,  # 제품CNN+주문feature concat 후 입력
            hidden_size=self.gru_hidden_dim,                    # GRU 은닉 상태 크기
            num_layers=gru_layers,
            batch_first=True,                                   # (B, 시퀀스, 벡터) 형태 사용
            dropout=dropout_rate if gru_layers > 1 else 0
        )
        # 마지막 hidden state → 최종 예측 (2층 FC)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, prediction_head_inter_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prediction_head_inter_dim, self.num_total_products)
        )

    def forward(self, product_x, meta_x):
        """
        Args:
            product_x: (B, N_orders, max_products_per_order, D_product) - 제품 임베딩 시퀀스
            meta_x: (B, N_orders, D_meta) - 주문별 메타 feature (ex. 요일, 시간, 간격 등)
        Returns:
            logits: (B, num_total_products) - multi-label 예측 로짓
        """
        batch_size, N_orders, max_products_per_order, D = product_x.shape

        # (1) CNN 입력 형태 맞추기: (B*N_orders, max_products_per_order, D) → (B*N_orders, D, max_products_per_order)
        cnn_input = product_x.view(batch_size * N_orders, max_products_per_order, self.product_emb_dim)
        cnn_input = cnn_input.permute(0, 2, 1)  # (B*N_orders, D_product, L)

        # (2) CNN 적용: (B*N_orders, D_product, L) → (B*N_orders, cnn_out_channels, pool_outsize)
        order_vectors_squeezed = self.order_encoder_cnn(cnn_input)
        # pool_outsize 차원까지 flatten 후 GRU 입력 차원 맞추기
        order_vectors = order_vectors_squeezed.flatten(start_dim=1)
        # (3) 주문 단위 시퀀스 복원: (B, N_orders, cnn_out_channels)
        order_vecs = order_vectors.view(batch_size, N_orders, -1)

        # (4) 주문정보 feature와 concat: (B, N_orders, cnn_out_channels + D_meta)
        if meta_x is not None:
            x_concat = torch.cat([order_vecs, meta_x], dim=-1)
        else:
            x_concat = order_vecs

        # (5) GRU 처리: (B, N_orders, ...) → 마지막 hidden state(h_n[-1], B, gru_hidden_dim)
        _, h_n = self.order_sequence_gru(x_concat)
        gru_final_state = h_n[-1]

        # (6) FC 예측
        logits = self.prediction_head(gru_final_state)
        return logits
