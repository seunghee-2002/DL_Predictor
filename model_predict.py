import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import os
from model import OrderPredictorModel  # 모델 정의가 포함된 파일에서 OrderPredictorModel 클래스를 가져옵니다.

def get_device():
    """CUDA 사용 가능 시 GPU, 아닐 경우 CPU 장치를 반환합니다."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA 장치를 사용합니다.")
    else:
        device = torch.device("cpu")
        print("CPU 장치를 사용합니다.")
    return device

def load_model_for_prediction(model_architecture_instance, path, device):
    """지정된 경로에서 모델의 state_dict를 불러오고, 모델을 평가 모드로 설정합니다."""
    print(f"{path} 에서 모델 가중치를 불러옵니다.")
    model_architecture_instance.load_state_dict(torch.load(path, map_location=device))
    model_architecture_instance.to(device)
    model_architecture_instance.eval() # 평가 모드로 설정
    return model_architecture_instance

def preprocess_input_for_prediction(past_orders_product_names_list,
                                     products_df, # 현재 함수 내에서는 직접 사용되지 않으나, 일관성 또는 향후 확장 위해 유지
                                     product_id_to_idx_map,
                                     product_name_to_id_map,
                                     product_embeddings_all_np,
                                     N_orders,
                                     max_products_per_order,
                                     embedding_dim,
                                     device):
    """
    사용자 입력을 받아 모델 입력 텐서로 전처리합니다.
    """
    input_sequence_tensor = torch.zeros(N_orders, max_products_per_order, embedding_dim, dtype=torch.float32)

    for i in range(N_orders):
        if i < len(past_orders_product_names_list): # 입력된 과거 주문 수만큼만 처리
            product_names_for_order_i = past_orders_product_names_list[i]

            for j in range(min(len(product_names_for_order_i), max_products_per_order)):
                p_name = product_names_for_order_i[j].strip()
                pid = product_name_to_id_map.get(p_name)

                if pid is not None:
                    emb_idx = product_id_to_idx_map.get(int(pid))
                    if emb_idx is not None and emb_idx < product_embeddings_all_np.shape[0]:
                        embedding_vector = product_embeddings_all_np[emb_idx]
                        input_sequence_tensor[i, j] = torch.tensor(embedding_vector, dtype=torch.float32)

    return input_sequence_tensor.unsqueeze(0).to(device)


def get_predicted_product_details_from_logits(model_instance,
                                             input_tensor,
                                             idx_to_product_id_map,
                                             product_id_to_name_map,
                                             product_embeddings_all_np,
                                             candidate_pool_size,
                                             probability_threshold):
    """
    모델 예측(로짓)을 수행하고, 조건에 맞는 제품들의 상세 정보 리스트를 반환합니다.
    """
    with torch.no_grad():
        logits = model_instance(input_tensor)

    probabilities = torch.sigmoid(logits).squeeze(0)

    actual_pool_size = min(candidate_pool_size, len(probabilities))
    top_candidate_probs, top_candidate_indices = torch.topk(probabilities, k=actual_pool_size)

    predicted_product_details_list = []

    for prob_tensor, original_emb_idx_tensor in zip(top_candidate_probs, top_candidate_indices):
        prob = prob_tensor.item()
        original_emb_idx = original_emb_idx_tensor.item()

        if prob >= probability_threshold:
            pid = idx_to_product_id_map.get(original_emb_idx)
            if pid is not None:
                p_name = product_id_to_name_map.get(pid, f"알 수 없는 제품 (ID: {pid})")
                if original_emb_idx < product_embeddings_all_np.shape[0]:
                    product_embedding_vector = product_embeddings_all_np[original_emb_idx]
                    predicted_product_details_list.append({
                        "name": p_name,
                        "id": pid,
                        "probability": prob,
                        "embedding": product_embedding_vector
                    })

    predicted_product_details_list.sort(key=lambda x: x["probability"], reverse=True)
    return predicted_product_details_list


def main():
    parser = argparse.ArgumentParser(description="학습된 CNN-GRU-CNN 모델로 다음 주문 예측")

    parser.add_argument("--model_path", type=str, default="./models/cnn_gru_cnn.pth", help="학습된 모델 파일 (.pth) 경로") # 학습 시 사용된 모델 이름으로 변경 필요
    parser.add_argument("--embeddings_path", type=str, default="./DLResource/ev_final64.npy", help="제품 임베딩 파일 경로")
    parser.add_argument("--products_path", type=str, default="./DLResource/RawData/products.csv", help="제품 메타데이터 파일 경로")

    # --- 모델 아키텍처 파라미터 (학습 시 사용된 값과 정확히 일치해야 함) ---
    parser.add_argument("--N_orders", type=int, default=5, help="모델 입력으로 사용된 과거 주문 시퀀스 길이")
    parser.add_argument("--max_products_per_order", type=int, default=145, help="주문 당 최대 제품 수")
    parser.add_argument("--product_emb_dim", type=int, default=64, help="제품 임베딩 차원")
    
    # 학습 시 설정과 일치시켜야 하는 주요 하이퍼파라미터들
    parser.add_argument("--cnn_out_channels", type=int, default=128, help="CNN 출력 채널 수") # 학습 시 값으로 변경 (예: 256)
    parser.add_argument("--cnn_kernel_size", type=int, default=3, help="CNN 커널 크기") # 학습 시 값으로 변경 (예: 5)
    parser.add_argument("--gru_hidden_dim", type=int, default=256, help="GRU 은닉 상태 차원") # 학습 시 값으로 변경 (예: 512)
    parser.add_argument("--gru_layers", type=int, default=1, help="GRU 계층 수") # 학습 시 값으로 변경 (예: 2)
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="모델 내 드롭아웃 비율") # 학습 시 값으로 변경
    parser.add_argument("--prediction_head_inter_dim", type=int, default=128, help="예측 헤드 중간 차원") # 학습 시 값으로 변경 (예: 256)

    parser.add_argument("--candidate_pool_size", type=int, default=145, help="확률 상위 N개 후보군 크기")
    parser.add_argument("--probability_threshold", type=float, default=0.1, help="제품 선택 확률 임계값")

    args = parser.parse_args()
    device = get_device()

    # ... (데이터 로드 부분은 동일) ...
    print("데이터 로드 중...")
    if not os.path.exists(args.products_path):
        print(f"오류: 제품 파일({args.products_path})을 찾을 수 없습니다.")
        return
    products_df = pd.read_csv(args.products_path)
    num_total_products = len(products_df)
    # ... (나머지 데이터 로드) ...
    if not os.path.exists(args.embeddings_path):
        print(f"오류: 임베딩 파일({args.embeddings_path})을 찾을 수 없습니다.")
        return
    product_embeddings_all_np = np.load(args.embeddings_path)

    # 임베딩 파일 행 수와 products.csv 제품 수 비교 (조정)
    if product_embeddings_all_np.shape[0] == num_total_products + 1:
        print(f"알림: 임베딩 파일에 제품 수보다 1개 많은 행이 있습니다. 첫 번째 행을 패딩/UNK용으로 간주하고 실제 제품 임베딩을 사용합니다.")
        # 이 경우 product_id_to_idx_map에서 생성된 인덱스를 그대로 사용하면 되지만, 
        # 만약 product_embeddings_all_np[0]이 패딩이고 실제 product_id 1이 emb_idx 1에 매핑된다면
        # custom_dataset.py 와 동일한 로직으로 product_id_to_idx_map을 사용해야 합니다.
        # 현재 custom_dataset.py는 product_id를 0부터 시작하는 인덱스로 매핑합니다.
        # ev_final64.npy가 (49689, 64)이고 products.csv가 49688개라면,
        # product_id_to_idx_map의 결과 emb_idx가 0~49687 범위일 것이고, 
        # product_embeddings_all_np[emb_idx] 접근은 문제가 없습니다.
        # 만약 product_embeddings_all_np[0]이 특별한 의미라면, product_id_to_idx_map이 이를 반영해야 합니다.
        # 여기서는 product_id_to_idx_map이 0-based 인덱스를 products.csv의 product_id에 직접 매핑한다고 가정합니다.
        pass # 특별한 처리 없이 진행, 단, product_id_to_idx_map이 0부터 시작하는 올바른 인덱스를 제공해야 함
    elif product_embeddings_all_np.shape[0] != num_total_products:
         print(f"경고: 임베딩 파일 행 수({product_embeddings_all_np.shape[0]})와 products.csv 제품 수({num_total_products})가 일치하지 않아 문제 발생 가능성이 있습니다.")
         # 필요시 여기서 중단하거나, 사용자가 문제를 인지하도록 강력히 경고

    product_id_to_idx_map = {pid: i for i, pid in enumerate(products_df['product_id'])}
    idx_to_product_id_map = {i: pid for i, pid in enumerate(products_df['product_id'])}
    product_name_to_id_map = pd.Series(products_df.product_id.values, index=products_df.product_name).to_dict()
    product_id_to_name_map = pd.Series(products_df.product_name.values, index=products_df.product_id).to_dict()
    print(f"제품 메타데이터 및 전체 임베딩 로드 완료. 총 제품 수: {num_total_products}")


    # --- 2. 모델 로드 ---
    print("모델 초기화 및 가중치 로드 중...")
    model = OrderPredictorModel(
        num_total_products=num_total_products,
        product_emb_dim=args.product_emb_dim,
        cnn_out_channels=args.cnn_out_channels,
        cnn_kernel_size=args.cnn_kernel_size,       # 추가된 인자 전달
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        dropout_rate=args.dropout_rate,             # 추가된 인자 전달
        prediction_head_inter_dim=args.prediction_head_inter_dim
    )
    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일({args.model_path})을 찾을 수 없습니다.")
        print("model_train.py에서 저장된 모델의 경로와 파일 이름을 정확히 입력했는지 확인하세요.")
        print(f"예상 경로: {os.path.join(os.getcwd(), args.model_path)}")
        return
        
    model = load_model_for_prediction(model, args.model_path, device)
    print("모델 로드 완료.")

    # ... (나머지 main 함수 로직은 동일) ...
    # --- 3. 사용자 입력 정의 (스크립트 내에서 직접 정의하는 방식) ---
    past_orders_to_predict_input = [
        ["Organic Strawberries"],
        ["Organic Baby Spinach", "Organic Hass Avocado", "Organic Strawberries"],
        ["Large Lemon", "Chicken Breast", "Organic Whole Milk", "Organic Strawberries"],
        ["Organic Whole Milk"],
        ["Organic Hass Avocado", "Organic Strawberries", "Organic Whole Milk"],
    ]

    if len(past_orders_to_predict_input) > args.N_orders:
        print(f"경고: 입력된 과거 주문의 수({len(past_orders_to_predict_input)})가 설정된 N_orders({args.N_orders})보다 많습니다. 앞의 {args.N_orders}개만 사용합니다.")
        past_orders_to_predict_input = past_orders_to_predict_input[:args.N_orders]
    elif len(past_orders_to_predict_input) < args.N_orders:
        # print(f"알림: 입력된 과거 주문의 수({len(past_orders_to_predict_input)})가 설정된 N_orders({args.N_orders})보다 적습니다. 나머지는 빈 주문으로 처리합니다.")
        past_orders_to_predict_input.extend([[] for _ in range(args.N_orders - len(past_orders_to_predict_input))])
    
    print(f"\n사용될 과거 주문 입력 (총 {args.N_orders}개):")
    for i, order in enumerate(past_orders_to_predict_input):
        print(f"  과거 주문 {i+1}: {order if order else '(비어 있음)'}")

    # --- 4. 입력 전처리 ---
    print("\n입력 데이터 전처리 중...")
    input_tensor = preprocess_input_for_prediction(
        past_orders_to_predict_input,
        products_df,
        product_id_to_idx_map,
        product_name_to_id_map,
        product_embeddings_all_np,
        args.N_orders,
        args.max_products_per_order,
        args.product_emb_dim,
        device
    )
    print(f"전처리된 입력 텐서 형태: {input_tensor.shape}")

    # --- 5. 예측 수행 및 결과 해석 ---
    print("예측 수행 및 결과 해석 중...")
    predicted_products_info = get_predicted_product_details_from_logits(
        model,
        input_tensor,
        idx_to_product_id_map, 
        product_id_to_name_map, 
        product_embeddings_all_np, 
        args.candidate_pool_size,
        args.probability_threshold
    )

    # --- 6. 결과 출력 ---
    print(f"\n--- 다음 주문 추천 (상위 {args.candidate_pool_size} 후보 중, 예측 확률 > {args.probability_threshold:.2f} 인 제품) ---")
    if predicted_products_info:
        print(f"총 {len(predicted_products_info)}개의 제품이 추천되었습니다.")
        for i, info in enumerate(predicted_products_info):
            print(f"{i+1}. 제품명: {info['name']} (ID: {info['id']})")
            print(f"   예측 확률: {info['probability']:.4f}")
    else:
        print("추천할 제품을 찾지 못했습니다 (모든 후보 제품이 설정된 확률 임계값 미만).")

if __name__ == '__main__':
    main()