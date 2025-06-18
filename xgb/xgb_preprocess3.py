# xgb_prepare_rank.py
import os, gc, argparse
import numpy as np, pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz

# ------------------------------- 인자 -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, default=300, help="per-user candidate 수")
args = parser.parse_args()
K = args.topk

# ---------------------- 0. 데이터 로드 ------------------------------
print("📂  데이터 로드 …")
U  = np.load("./DLResource/PreprocessData/seq_vec128.npy").astype(np.float32)   # (N_u ,128)
P  = np.load("./DLResource/PreprocessData/product128.npy").astype(np.float32)   # (N_p ,128)
yt = pd.read_csv("./DLResource/PreprocessData/data_Y_5.csv")                    # (index, product_id)

N_u, N_p = U.shape[0], P.shape[0]
print(f"   users={N_u:,}  products={N_p:,}  top-K={K}")

# ---------------------- 1. Top-K 후보 생성 --------------------------
print("🔍  Top-K 후보 계산 (dot-product)…")
cand_idx   = np.zeros((N_u, K), dtype=np.int32)
cand_score = np.zeros((N_u, K), dtype=np.float32)

P_T  = P.T                      # (128, N_p)
batch = 4096
for s in tqdm(range(0, N_u, batch)):
    e   = min(s + batch, N_u)
    sim = U[s:e] @ P_T         # (b , N_p)

    topk_idx  = np.argpartition(-sim, K - 1, axis=1)[:, :K]       # unsorted
    topk_val  = np.take_along_axis(sim, topk_idx, axis=1)

    order     = np.argsort(-topk_val, axis=1)
    cand_idx[s:e]   = np.take_along_axis(topk_idx, order, axis=1)
    cand_score[s:e] = np.take_along_axis(topk_val, order, axis=1)

    del sim; gc.collect()

np.save("cand_idx.npy",   cand_idx)
np.save("cand_score.npy", cand_score)
print("✅  후보 저장 완료")

# ---------------------- 2. 라벨 CSR 생성 ----------------------------
print("🏷️   타깃 라벨 매핑 …")

# (index → [product_id ...]) 를 딕셔너리로
y_dict = {}
for idx, pid in yt.itertuples(index=False):
    y_dict.setdefault(idx, []).append(pid)

rows, cols, data = [], [], []

for u in tqdm(range(N_u), desc="user-label map"):
    cand = cand_idx[u]                     # (K,)  user-specific Top-K product_id
    pid2col = {pid: j for j, pid in enumerate(cand)}   # ★ 0‥K-1 로 매핑

    for pid in y_dict.get(u, []):          # 정답 product 들
        j = pid2col.get(pid)
        if j is not None:                  # Top-K 안에 포함될 때만 1
            rows.append(u)
            cols.append(j)
            data.append(1)

Y_csr = csr_matrix((data, (rows, cols)),
                   shape=(N_u, K), dtype=np.int8)
save_npz("Y_topk.npz", Y_csr)
print(f"✅  Y_topk.npz 저장 완료  (shape: {Y_csr.shape})")

# ---------------------- 3. 입력 X CSR 저장 --------------------------
X_csr = csr_matrix(U)                      # (N_u ,128)  float32 CSR
save_npz("X_128.npz", X_csr)
print(f"✅  X_128.npz 저장 완료  (shape: {X_csr.shape})")
