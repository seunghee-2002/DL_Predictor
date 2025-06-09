import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
        # 하나의 learnable query가 전체 제품에 attention하여 세트(집합)을 1개 벡터로 압축
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))  # (1, 1, D)
        # PyTorch MultiheadAttention: input=(B, L, D)
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, output_dim)  # 출력 차원 변환

    def forward(self, x, mask=None):
        # x: (B*N_orders, L, D) = 제품 시퀀스(주문별로 풀어서)
        batch_size, seq_len, d = x.size()
        q = self.query.expand(batch_size, -1, -1)  # (B*N_orders, 1, D), query 복제
        if mask is not None:
            # mask: (B*N_orders, L) True=유효, False=패딩 → attn에서는 True=패딩!
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
        # MultiheadAttention 수행 (쿼리가 한 개라서 output shape: (B*N_orders, 1, D))
        attn_output, _ = self.mha(q, x, x, key_padding_mask=attn_mask)
        out = attn_output.squeeze(1)  # (B*N_orders, D)
        return self.fc(out)           # (B*N_orders, output_dim)로 변환

class AttentionPredictorModel(nn.Module):
    def __init__(self,
                 num_total_products: int,
                 product_emb_dim: int = 64,     # 제품 임베딩 차원
                 attn_out_dim: int = 128,       # attention 블록 출력 차원
                 attn_heads: int = 4,           # multi-head 수
                 attn_layers: int = 1,          # attention block 층 수
                 gru_hidden_dim: int = 256,     # GRU 히든 차원
                 gru_layers: int = 1,           # GRU 계층 수
                 dropout_rate: float = 0.1,     # 드롭아웃 확률
                 prediction_head_inter_dim: int = 128, # FC 헤드 중간 차원
                 order_meta_dim: int = 5        # 주문 정보 feature 수
                 ):
        super().__init__()
        self.product_emb_dim = product_emb_dim
        self.attn_out_dim = attn_out_dim
        self.num_total_products = num_total_products
        self.order_meta_dim = order_meta_dim

        # (1) 주문 내 제품 임베딩들 → attention block 통과
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=product_emb_dim,            # 제품 임베딩 차원
                nhead=attn_heads,                   # 멀티헤드 수
                batch_first=True,                   # (B, L, D) 포맷 지원
                dim_feedforward=product_emb_dim * 2,# 내부 FFN 크기
                dropout=dropout_rate,
                activation='relu',
                norm_first=True
            ) for _ in range(attn_layers)
        ])
        self.attn_pool = AttentionPooling(input_dim=product_emb_dim, output_dim=attn_out_dim, num_heads=attn_heads)

        # (2) GRU (주문 feature 포함, input_size = attn_out_dim + order_meta_dim)
        self.order_sequence_gru = nn.GRU(
            input_size=attn_out_dim + order_meta_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )
        # (3) FC 예측 헤드
        self.prediction_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, prediction_head_inter_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prediction_head_inter_dim, self.num_total_products)
        )

    def forward(self, product_x, meta_x, mask=None):
        """
        Args:
            product_x: (B, N_orders, L, D_product) - 주문별 제품 임베딩 시퀀스
            meta_x: (B, N_orders, D_meta)           - 주문별 메타 feature
            mask: (B, N_orders, L)                  - 제품 패딩 마스크(True=유효, False=패딩)
        Returns:
            logits: (B, num_total_products)         - multi-label 예측 로짓
        """
        batch_size, N_orders, L, D = product_x.shape

        # (1) 제품 feature에 대해 attention block 반복 적용
        x_reshaped = product_x.view(batch_size * N_orders, L, D)
        mask_reshaped = mask.view(batch_size * N_orders, L) if mask is not None else None
        for layer in self.attn_layers:
            x_reshaped = layer(
                x_reshaped,
                src_key_padding_mask=~mask_reshaped.bool() if mask_reshaped is not None else None
            )
        # (2) attention pooling → 주문별로 1개 벡터로 압축
        order_vecs = self.attn_pool(x_reshaped, mask=mask_reshaped)  # (B*N_orders, attn_out_dim)
        order_vecs = order_vecs.view(batch_size, N_orders, self.attn_out_dim)  # (B, N_orders, attn_out_dim)

        # (3) 주문정보 feature와 concat → GRU 입력
        if meta_x is not None:
            x_concat = torch.cat([order_vecs, meta_x], dim=-1)  # (B, N_orders, attn_out_dim + D_meta)
        else:
            x_concat = order_vecs

        # (4) 주문 시퀀스(시간축) GRU 처리
        _, h_n = self.order_sequence_gru(x_concat)   # h_n: (num_layers, B, gru_hidden_dim)
        gru_final_state = h_n[-1]                    # 마지막 GRU 은닉 상태만 추출 (B, gru_hidden_dim)

        # (5) FC 예측(다중 레이블 로짓)
        logits = self.prediction_head(gru_final_state)
        return logits
