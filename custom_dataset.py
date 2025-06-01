import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class OrderSequenceDataset(Dataset):
    def __init__(self, data_x_product_path, data_y_path, embeddings_npy_path, products_path, N_orders, max_products_per_order, embedding_dim=64):
        super(OrderSequenceDataset, self).__init__() # nn.Module이 아니므로 super() 또는 super(OrderSequenceDataset, self)
        self.N_orders = N_orders
        self.max_products_per_order = max_products_per_order
        self.embedding_dim = embedding_dim

        print(f"데이터셋 초기화 시작 (다중 레이블 분류 모드): N_orders={N_orders}, max_products_per_order={max_products_per_order}")

        if not os.path.exists(embeddings_npy_path):
            raise FileNotFoundError(f"임베딩 파일 없음: {embeddings_npy_path}")
        self.product_embeddings_all = np.load(embeddings_npy_path)
        print(f"임베딩 로드 완료: {embeddings_npy_path}, 형태: {self.product_embeddings_all.shape}")

        if not os.path.exists(products_path):
            raise FileNotFoundError(f"제품 파일 없음: {products_path}")
        products_df = pd.read_csv(products_path)
        self.product_id_to_idx_map = {pid: i for i, pid in enumerate(products_df['product_id'])}
        self.num_total_products = len(products_df)
        print(f"제품 메타데이터 로드 완료: {products_path}, 총 제품 수 (매핑 기준): {self.num_total_products}")

        if not os.path.exists(data_x_product_path):
            raise FileNotFoundError(f"입력 X 제품 파일 없음: {data_x_product_path}")
        x_prod_df = pd.read_csv(data_x_product_path, dtype={'product_id': 'Int64', 'index': 'Int64', 'sequence_num': 'Int64'})

        if not os.path.exists(data_y_path):
            raise FileNotFoundError(f"타겟 Y 파일 없음: {data_y_path}")
        y_df = pd.read_csv(data_y_path, dtype={'product_id': 'Int64', 'index': 'Int64'})
        
        print(f"입력 X 제품 파일 로드 완료: {data_x_product_path}, 행 수: {len(x_prod_df)}")
        print(f"타겟 Y 파일 로드 완료: {data_y_path}, 행 수: {len(y_df)}")

        self.samples = []
        x_prod_df_sorted = x_prod_df.sort_values(by=['index', 'sequence_num'])
        grouped_x = x_prod_df_sorted.groupby('index')
        grouped_y = y_df.groupby('index')

        num_valid_samples = 0
        num_skipped_samples_y_missing = 0

        for uid, user_x_orders_group in tqdm(grouped_x, desc="샘플 구성 중 (다중 레이블 타겟)"):
            target_prods_group = grouped_y.get_group(uid) if uid in grouped_y.groups else None
            
            if target_prods_group is None or target_prods_group.empty:
                num_skipped_samples_y_missing +=1
                continue
            
            orders_in_sequence = []
            for seq_idx in range(self.N_orders):
                current_order_df = user_x_orders_group[user_x_orders_group['sequence_num'] == seq_idx]
                orders_in_sequence.append(current_order_df['product_id'].tolist())
            
            target_product_ids = target_prods_group['product_id'].tolist()
            
            self.samples.append({
                'index': uid, 
                'past_orders_pids': orders_in_sequence,
                'target_pids': target_product_ids
            })
            num_valid_samples += 1
        
        print(f"샘플 구성 완료. 총 유효 샘플 수: {num_valid_samples}")
        if num_skipped_samples_y_missing > 0:
            print(f"  경고: {num_skipped_samples_y_missing}개의 샘플이 타겟 주문 누락으로 스킵됨.")
        if not self.samples:
            print("경고: 구성된 샘플이 없습니다. 데이터 파일 또는 N_orders 설정을 확인하세요.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        past_orders_pids_list = sample['past_orders_pids']
        target_pids = sample['target_pids']

        input_sequence_tensor = torch.zeros(self.N_orders, self.max_products_per_order, self.embedding_dim, dtype=torch.float32)
        for i in range(self.N_orders):
            product_ids_for_order_i = past_orders_pids_list[i]
            for j in range(min(len(product_ids_for_order_i), self.max_products_per_order)):
                pid = product_ids_for_order_i[j]
                if pd.isna(pid): continue
                emb_idx = self.product_id_to_idx_map.get(int(pid))
                if emb_idx is not None and emb_idx < len(self.product_embeddings_all):
                    embedding_vector = self.product_embeddings_all[emb_idx]
                    input_sequence_tensor[i, j] = torch.tensor(embedding_vector, dtype=torch.float32)

        target_multi_hot_vector = torch.zeros(self.num_total_products, dtype=torch.float32)
        if target_pids:
            for pid in target_pids:
                if pd.isna(pid): continue
                emb_idx = self.product_id_to_idx_map.get(int(pid))
                if emb_idx is not None and emb_idx < self.num_total_products:
                    target_multi_hot_vector[emb_idx] = 1.0

        return input_sequence_tensor, target_multi_hot_vector

def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_tensor = torch.stack(inputs)
    targets_tensor = torch.stack(targets)
    return inputs_tensor, targets_tensor


if __name__ == '__main__':
    N_ORDERS_FOR_TEST = 5 
    MAX_PRODUCTS_FOR_TEST = 145
    
    sample_data_x_product_path = "./DLResource/PreprocessData/data_X_product_5.csv"
    sample_data_y_path = "./DLResource/PreprocessData/data_Y_5.csv"
    sample_embeddings_path = "./DLResource/ev_final64.npy"
    sample_products_path = "./DLResource/RawData/products.csv"

    print(f"테스트용 파일 경로:")
    print(f"  X_product: {sample_data_x_product_path}")
    print(f"  Y: {sample_data_y_path}")
    print(f"  Embeddings: {sample_embeddings_path}")
    print(f"  Products: {sample_products_path}")
    print(f"  테스트용 N_orders: {N_ORDERS_FOR_TEST}")
    print(f"  테스트용 max_products_per_order: {MAX_PRODUCTS_FOR_TEST}")

    required_files = [sample_data_x_product_path, sample_data_y_path, sample_embeddings_path, sample_products_path]
    files_exist = all(os.path.exists(f) for f in required_files)

    if not files_exist:
        missing_files = [f for f in required_files if not os.path.exists(f)]
        print(f"\n경고: 다음 예시 데이터 파일을 찾을 수 없습니다: {missing_files}. 테스트를 건너<0xEB><0><0x88>니다.")
        print("위 경로에 실제 데이터 파일이 있는지 확인하거나, 경로를 수정해주세요.")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
    else:
        print("\n테스트 데이터셋 생성 중...")
        try:
            test_dataset = OrderSequenceDataset(
                data_x_product_path=sample_data_x_product_path,
                data_y_path=sample_data_y_path,
                embeddings_npy_path=sample_embeddings_path,
                products_path=sample_products_path,
                N_orders=N_ORDERS_FOR_TEST,
                max_products_per_order=MAX_PRODUCTS_FOR_TEST 
            )

            if len(test_dataset) > 0:
                print(f"\n데이터셋 생성 성공. 총 샘플 수: {len(test_dataset)}")
                print(f"  데이터셋의 num_total_products: {test_dataset.num_total_products}")

                test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=4, 
                    shuffle=True, 
                    collate_fn=custom_collate_fn
                )

                for i, (input_batch, target_batch) in enumerate(test_dataloader):
                    print(f"\n배치 {i+1} 형태:")
                    print(f"  입력 배치 형태: {input_batch.shape}") 
                    print(f"  타겟 배치 형태: {target_batch.shape}")
                    
                    if input_batch.shape[0] > 0 and input_batch.shape[1] > 0 and input_batch.shape[2] > 0 and input_batch.shape[3] > 0 :
                         print(f"    첫 샘플, 첫 주문, 첫 제품 임베딩 (일부): {input_batch[0, 0, 0, :5].tolist()}...")
                    else:
                        print(f"    첫 샘플, 첫 주문에 제품이 없거나 패딩됨.")
                    
                    target_sample_sparse = target_batch[0].nonzero().squeeze().tolist()
                    if isinstance(target_sample_sparse, int):
                        target_sample_sparse = [target_sample_sparse]
                    print(f"    첫 샘플 타겟 (1로 표시된 인덱스 일부, 최대 10개): {target_sample_sparse[:10]}...")
                    print(f"    첫 샘플 타겟에서 1의 개수: {int(torch.sum(target_batch[0]).item())}")
                    break 
            else:
                print("\n데이터셋은 생성되었으나, 처리할 샘플이 없습니다.")
        except FileNotFoundError as e:
            print(f"\n테스트 중 파일 에러 발생: {e}")
        except Exception as e:
            print(f"\n데이터셋 테스트 중 예외 발생: {e}")
            import traceback
            traceback.print_exc()