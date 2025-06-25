from json import load,dump
import numpy as np
import pandas as pd
import os
from pathlib import Path

PATH=Path(os.path.abspath(__file__)).parent

# =======================================
# 정수는 int32, 소수는 float32로 저장할 것!!
# =======================================

N=5 # 입력의 시퀀스 길이

S_len=N+1 # 시퀀스 전체 길이

data_Y={
    "index":[], # int
    "product_id":[] #int
}
data_X={
    "index":[], # int
    "sequence_num":[], # int
    "order_dow_sin":[], # float
    "order_dow_cos":[], # float
    "order_hour_of_day_sin":[], # float
    "order_hour_of_day_cos":[], # float
    "days_since_prior_order":[] # float, 최대일로 나눠버리기
}
data_X_product={
    "index":[], # int
    "sequence_num":[], # int
    "product_id":[], # int
    "is_reordered":[] # int
}

solution_Y={
    "index":[], # int
    "product_id":[] #int
}
solution_X={
    "index":[], # int
    "sequence_num":[], # int
    "order_dow_sin":[], # float
    "order_dow_cos":[], # float
    "order_hour_of_day_sin":[], # float
    "order_hour_of_day_cos":[], # float
    "days_since_prior_order":[] # float, 최대일(30)로 나눠버리기 -> 날의 갯수는 계층적 의미를 지님
}
solution_X_product={
    "index":[], # int
    "sequence_num":[], # int
    "product_id":[], # int
    "is_reordered":[] # int(bool)
}

def process_cyclic_time(val,cycle_size):
    # 크지않은 범주의 주기적 반복되는 feature 처리
    # 예: 0~23 -> cycle_size==24
    radians = 2 * np.pi * val / cycle_size
    return np.sin(radians), np.cos(radians)


with open(PATH / "meta.json") as f:
    meta=load(f)
with open(PATH / "modified.json") as f:
    modified=load(f)

# ================================================================================================================
# 예상되는 null order 삽입 없이 사용할 수 있는 user 갯수
# N: 2, user_data_live: 1047343, user_data_removed: 0, user_solution_live: 206209, user_solution_removed: 0
# N: 3, user_data_live: 763591, user_data_removed: 8686, user_solution_live: 197523, user_solution_removed: 8686
# N: 4, user_data_live: 584140, user_data_removed: 31137, user_solution_live: 190372, user_solution_removed: 15837
# N: 5, user_data_live: 465455, user_data_removed: 49404, user_solution_live: 184544, user_solution_removed: 21665
# ================================================================================================================
# user_data_live=0
# user_data_removed=0
# user_solution_live=0
# user_solution_removed=0
# for user_id in meta:
#     size=int(meta[user_id]["size"]/(N+1))
#     user_data_live+=size
#     if size==0:
#         user_data_removed+=1
#         if meta[user_id]["test"]:
#             user_solution_removed+=1
#             continue
#     user_solution_live+=1
# print(f"N: {N}, user_data_live: {user_data_live}, user_data_removed: {user_data_removed}, user_solution_live: {user_solution_live}, user_solution_removed: {user_solution_removed}")


# ====================
# 최대 주문 간격 -> 30일
# ====================
# M=0 # 30
# for user_id in modified:
#     orders=modified[user_id]
#     for order_num in orders:
#         order=orders[order_num]
#         days=order["days_since_prior_order"]
#         if not np.isnan(days):
#             M=max(M,days)
# print(M)

# ==============================================
# 생성되는 시퀀스 갯수
# N: 2, data_size: 680874, solution_size: 366469
# N: 3, data_size: 500890, solution_size: 262701
# N: 4, data_size: 382928, solution_size: 201212
# N: 5, data_size: 304531, solution_size: 160924
# ==============================================
# data_size=0
# solution_size=0
# for user_id in meta:
#     header=meta[user_id]
#     size=header["size"]//S_len
#     if header["test"]:
#         solution_size+=size
#     else:
#         data_size+=size
# print(f"N: {N}, data_size: {data_size}, solution_size: {solution_size}")

# =================================================
# 데이터 생성(마지막 단계에서 1만큼 증가함)
# N: 2, index: {'data': 680875, 'solution': 366470}
# N: 3, index: {'data': 500891, 'solution': 262702}
# N: 4, index: {'data': 382929, 'solution': 201213}
# N: 5, index: {'data': 304532, 'solution': 160925}
# =================================================
index={
    "data":1,
    "solution":1
}
for user_id in modified:
    header=meta[user_id]
    orders=modified[user_id]
    offset=header["size"]%S_len
    if header["test"]:
        index_key="solution"
        X=solution_X
        X_product=solution_X_product
        Y=solution_Y
    else:
        index_key="data"
        X=data_X
        X_product=data_X_product
        Y=data_Y
    for i in range(1,header["size"]-offset+1):
        order_num=offset+i
        index_value=index[index_key]
        order=orders[str(order_num)]
        sequence_num=(i-1)%S_len+1
        order_dow_sin,order_dow_cos=process_cyclic_time(order["order_dow"],7)
        order_hour_of_day_sin,order_hour_of_day_cos=process_cyclic_time(order["order_hour_of_day"],24)
        days_since_prior_order=order["days_since_prior_order"]/30
        if i%S_len==0:
            for product_id in order["products"]:
                Y["index"].append(index_value)
                Y["product_id"].append(product_id)
            index[index_key]+=1
        else:
            for product_id,is_reordered in zip(order["products"],order["is_reordered"]):
                X_product["index"].append(index_value)
                X_product["product_id"].append(product_id)
                X_product["sequence_num"].append(sequence_num)
                X_product["is_reordered"].append(is_reordered)
            X["index"].append(index_value)
            X["sequence_num"].append(sequence_num)
            X["order_dow_sin"].append(order_dow_sin)
            X["order_dow_cos"].append(order_dow_cos)
            X["order_hour_of_day_sin"].append(order_hour_of_day_sin)
            X["order_hour_of_day_cos"].append(order_hour_of_day_cos)
            X["days_since_prior_order"].append(days_since_prior_order)
CSV_PATH=PATH / f"N_{N}"
if not os.path.isdir(CSV_PATH):
    os.mkdir(CSV_PATH)
pd.DataFrame(data_Y).to_csv(CSV_PATH / f"data_Y_{N}.csv",index=False)
pd.DataFrame(data_X).to_csv(CSV_PATH / f"data_X_{N}.csv",index=False)
pd.DataFrame(data_X_product).to_csv(CSV_PATH / f"data_X_product_{N}.csv",index=False)
pd.DataFrame(solution_Y).to_csv(CSV_PATH / f"solution_Y_{N}.csv",index=False)
pd.DataFrame(solution_X).to_csv(CSV_PATH / f"solution_X_{N}.csv",index=False)
pd.DataFrame(solution_X_product).to_csv(CSV_PATH / f"solution_X_product_{N}.csv",index=False)
print(f"N: {N}, index: {index}")