import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# 제품 임베딩 기반의 시퀀스 데이터셋 정의
class OrderSequenceDataset(Dataset):
    def __init__(self,
                 data_x_product_path,
                 data_y_path,
                 embeddings_npy_path,
                 N_orders,
                 max_products_per_order,
                 embedding_dim=64):
        super().__init__()
        self.N_orders = N_orders
        self.max_products_per_order = max_products_per_order
        self.embedding_dim = embedding_dim

        # 시퀀스 데이터 로드
        x_prod_df = pd.read_csv(data_x_product_path)
        y_df = pd.read_csv(data_y_path)

        # 제품 임베딩 (numpy: [num_products, emb_dim])
        self.product_embeddings_all = np.load(embeddings_npy_path)

        # 샘플 단위로 입력 시퀀스 및 타겟 구성
        x_prod_grouped = x_prod_df.groupby('index')
        y_grouped = y_df.groupby('index')
        self.samples = []

        for idx in x_prod_grouped.groups.keys():
            if idx not in y_grouped.groups:
                continue
            group_prod = x_prod_grouped.get_group(idx)
            group_y = y_grouped.get_group(idx)

            orders_prod = []
            for seq in range(self.N_orders):
                prod_row = group_prod[group_prod['sequence_num'] == seq]
                pids = prod_row['product_id'].tolist()
                orders_prod.append(pids)

            target_pids = group_y['product_id'].tolist()
            self.samples.append({
                "orders_prod": orders_prod,       # 과거 N개의 주문 제품 리스트
                "target_pids": target_pids        # 다음 주문의 정답 제품 리스트
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 입력 시퀀스 텐서 초기화: [N_orders, max_products_per_order, emb_dim]
        input_seq_tensor = torch.zeros(self.N_orders, self.max_products_per_order, self.embedding_dim, dtype=torch.float32)

        # 각 주문별 제품 ID → 임베딩 변환
        for i in range(self.N_orders):
            for j, pid in enumerate(sample["orders_prod"][i][:self.max_products_per_order]):
                if pd.isna(pid): continue
                emb_idx = self.product_id_to_idx_map.get(int(pid))
                if emb_idx is not None and emb_idx < len(self.product_embeddings_all):
                    input_seq_tensor[i, j] = torch.tensor(self.product_embeddings_all[emb_idx], dtype=torch.float32)

        # 타겟 벡터: [num_total_products] → 구매한 제품에 대해 1로 설정
        target = torch.zeros(self.num_total_products, dtype=torch.float32)
        for pid in sample["target_pids"]:
            if pd.isna(pid): continue
            emb_idx = self.product_id_to_idx_map.get(int(pid))
            if emb_idx is not None and emb_idx < self.num_total_products:
                target[emb_idx] = 1.0

        return input_seq_tensor, target

# DataLoader에서 사용할 collate 함수
def custom_collate_fn(batch):
    inputs_product, targets = zip(*batch)
    return torch.stack(inputs_product), torch.stack(targets)
