import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class OrderSequenceDataset(Dataset):
    def __init__(
        self,
        data_x_product_path,  # 주문별 제품 데이터 (order+product, 예: data_X_product_5.csv)
        data_x_meta_path,     # 주문 헤더 feature 데이터 (예: data_X_5.csv)
        data_y_path,          # 정답 라벨(예: data_Y_5.csv)
        embeddings_npy_path,  # 제품 임베딩 (np.load로 불러오는 .npy)
        products_path,        # 제품 메타정보 (예: products.csv)
        N_orders,
        max_products_per_order,
        embedding_dim=64
    ):
        super().__init__()
        self.N_orders = N_orders
        self.max_products_per_order = max_products_per_order
        self.embedding_dim = embedding_dim

        # (1) 주문별 제품 데이터 로드 (한 주문에 여러 제품 포함)
        x_prod_df = pd.read_csv(data_x_product_path)
        # (2) 주문 헤더 feature 데이터 로드 (각 주문의 요일, 시간, 간격 등 메타데이터 포함)
        x_meta_df = pd.read_csv(data_x_meta_path)
        # (3) 타겟 데이터 (각 샘플별로 예측할 제품 multi-hot 벡터 생성용)
        y_df = pd.read_csv(data_y_path)
        # (4) 제품 임베딩 로드 (제품 번호 → 임베딩 벡터 매핑용)
        self.product_embeddings_all = np.load(embeddings_npy_path)
        # (5) 제품 id → 행 index 매핑 (임베딩 벡터 추출시 사용)
        products_df = pd.read_csv(products_path)
        self.product_id_to_idx_map = {pid: i for i, pid in enumerate(products_df['product_id'])}
        self.num_total_products = len(products_df)

        # (6) 내부에 DF 저장(혹시 __getitem__에서 쓰려면)
        self.x_prod_df = x_prod_df
        self.x_meta_df = x_meta_df
        self.y_df = y_df

        # (7) 주문 메타 feature 컬럼 자동 추출(키 컬럼은 제외)
        self.meta_feature_cols = [
            col for col in x_meta_df.columns
            if col not in ["index", "sequence_num"]
        ]
        print(f"활용할 주문 feature: {self.meta_feature_cols}")
        self.meta_feature_dim = len(self.meta_feature_cols)

        # (8) 샘플 리스트 만들기: index(=시퀀스 id)별로 그룹핑 후, 시퀀스별 orders, 타겟 추출
        x_meta_grouped = x_meta_df.groupby('index')
        x_prod_grouped = x_prod_df.groupby('index')
        y_grouped = y_df.groupby('index')
        self.samples = []

        for idx, group_meta in x_meta_grouped:
            # 제품/타겟 없는 샘플은 스킵
            if idx not in x_prod_grouped.groups or idx not in y_grouped.groups:
                continue
            group_prod = x_prod_grouped.get_group(idx)
            group_y = y_grouped.get_group(idx)

            # 시퀀스 내 각 주문별로
            orders_meta = []  # 주문별 메타 feature
            orders_prod = []  # 주문별 제품 id 리스트
            for seq in range(self.N_orders):
                # (a) 해당 주문의 메타 feature 추출(없으면 0벡터)
                meta_row = group_meta[group_meta['sequence_num'] == seq]
                if not meta_row.empty:
                    feat = meta_row[self.meta_feature_cols].iloc[0].values.astype(np.float32)
                else:
                    feat = np.zeros(self.meta_feature_dim, dtype=np.float32)
                orders_meta.append(feat)
                # (b) 해당 주문의 제품 id 추출(없으면 빈 리스트)
                prod_row = group_prod[group_prod['sequence_num'] == seq]
                pids = prod_row['product_id'].tolist()
                orders_prod.append(pids)

            # (c) 타겟 제품 id 리스트
            target_pids = group_y['product_id'].tolist()
            self.samples.append({
                "orders_meta": orders_meta,   # (N_orders, meta_dim)
                "orders_prod": orders_prod,   # (N_orders, 제품 id list)
                "target_pids": target_pids    # (list)
            })

    def __len__(self):
        # 전체 시퀀스 샘플 수 반환
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # (1) 주문 메타 feature 텐서 (N_orders, meta_dim)
        order_meta_tensor = torch.tensor(np.array(sample["orders_meta"]), dtype=torch.float32)

        # (2) 제품 임베딩 텐서 (N_orders, max_products_per_order, embedding_dim)
        input_seq_tensor = torch.zeros(self.N_orders, self.max_products_per_order, self.embedding_dim, dtype=torch.float32)
        for i in range(self.N_orders):
            # 해당 주문에 들어간 실제 제품 id 리스트(최대 max_products_per_order개만)
            for j, pid in enumerate(sample["orders_prod"][i][:self.max_products_per_order]):
                if pd.isna(pid): continue
                emb_idx = self.product_id_to_idx_map.get(int(pid))
                if emb_idx is not None and emb_idx < len(self.product_embeddings_all):
                    embedding_vector = self.product_embeddings_all[emb_idx]
                    input_seq_tensor[i, j] = torch.tensor(embedding_vector, dtype=torch.float32)
        # (3) 타겟 multi-hot 벡터 (num_total_products 차원)
        target = torch.zeros(self.num_total_products, dtype=torch.float32)
        for pid in sample["target_pids"]:
            if pd.isna(pid): continue
            emb_idx = self.product_id_to_idx_map.get(int(pid))
            if emb_idx is not None and emb_idx < self.num_total_products:
                target[emb_idx] = 1.0
        return input_seq_tensor, order_meta_tensor, target  # (N_orders, max_products_per_order, D), (N_orders, D_meta), (num_total_products,)

def custom_collate_fn(batch):
    """
    DataLoader에서 여러 샘플을 batch로 합칠 때 사용
    - 제품 임베딩 텐서: (B, N_orders, max_products_per_order, D_product)
    - 주문 메타 feature 텐서: (B, N_orders, D_meta)
    - 타겟 벡터: (B, num_total_products)
    """
    inputs_product, inputs_meta, targets = zip(*batch)
    return (
        torch.stack(inputs_product),  # 제품 임베딩 batch
        torch.stack(inputs_meta),     # 주문 메타정보 batch
        torch.stack(targets),         # multi-hot 타겟 벡터 batch
    )
