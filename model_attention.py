import torch
import torch.nn as nn

# Transformer 기반 Attention Pooling 모듈
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, num_queries=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, input_dim))  # 학습 가능한 쿼리 벡터
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(num_queries * input_dim, output_dim)           # 압축 후 FC

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1)  # (B, num_queries, D)
        attn_output, _ = self.mha(q, x, x, key_padding_mask=~mask.bool() if mask is not None else None)
        return self.fc(attn_output.reshape(B, -1))  # (B, output_dim)

# Attention 기반 예측 모델
class AttentionPredictorModel(nn.Module):
    def __init__(self,
                 num_total_products: int,
                 product_emb_dim: int,
                 attn_out_dim: int,
                 attn_heads: int,
                 attn_layers: int,
                 gru_hidden_dim: int,
                 gru_layers: int,
                 dropout_rate: float,
                 prediction_head_inter_dim: int):
        super().__init__()

        # 주문 내 제품 시퀀스를 인코딩하는 Transformer Encoder 층들
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=product_emb_dim,
                nhead=attn_heads,
                dim_feedforward=product_emb_dim * 2,
                batch_first=True,
                norm_first=True
            )
            for _ in range(attn_layers)
        ])

        # Transformer 출력 시퀀스를 압축
        self.attn_pool = AttentionPooling(product_emb_dim, attn_out_dim, num_heads=attn_heads)

        # 과거 주문 벡터 시퀀스를 GRU로 처리
        self.gru = nn.GRU(
            input_size=attn_out_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )

        # 최종 예측 헤드
        self.prediction_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, prediction_head_inter_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prediction_head_inter_dim, num_total_products)
        )

    def forward(self, product_x, mask=None):
        # product_x: (B, N, L, D)
        B, N, L, D = product_x.shape
        x = product_x.view(B * N, L, D)

        # Transformer 인코딩
        for layer in self.attn_layers:
            x = layer(x)

        # 각 주문을 요약 벡터로 변환
        pooled = self.attn_pool(x)                 # (B*N, attn_out_dim)
        order_vecs = pooled.view(B, N, -1)         # (B, N, attn_out_dim)

        # GRU + FC
        _, h_n = self.gru(order_vecs)              # (num_layers, B, hidden)
        return self.prediction_head(h_n[-1])
