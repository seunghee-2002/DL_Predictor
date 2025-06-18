import numpy as np, pandas as pd
from tqdm import tqdm

ev = np.load("./DLResource/ev_final16.npy").astype(np.float32)          # (49 238,16)

# ----- (A) W, b 준비 -------------------------------------------------
np.random.seed(42)                                         # 재현성
W = np.random.randn(16, 128).astype(np.float32)            # (16,128)
b = np.zeros(128, dtype=np.float32)
np.save("W.npy", W)      # ❶ 반드시 저장!
np.save("b.npy", b)      # (b는 0이면 안 써도 무방)

# ----- (B) order-level 벡터 ------------------------------------------
order_vec = {}
x = pd.read_csv("./DLResource/PreprocessData/data_X_product_5.csv")                    # index,sequence_num,product_id

for (idx, seq), grp in tqdm(x.groupby(["index", "sequence_num"]), total=x.groupby(["index","sequence_num"]).ngroups):
    P = ev[grp["product_id"].values.astype(int)]           # (m,16)
    h = np.maximum(P @ W + b, 0)                           # (m,128) ReLU
    order_vec[(idx, seq)] = h.max(axis=0).astype(np.float16)

# ----- (C) user-sequence 벡터 ----------------------------------------
λ = 0.8
n_user  = x["index"].max() + 1
seq_vec = np.zeros((n_user, 128), dtype=np.float16)

for idx in tqdm(range(n_user), desc="user-agg"):
    vs, ws = [], []
    for s in range(5):
        if (idx, s) in order_vec:
            vs.append(order_vec[(idx, s)])
            ws.append(np.exp(-λ*(4-s)))
    if vs:
        vs = np.vstack(vs); ws = np.asarray(ws)[:,None]
        seq_vec[idx] = (vs*ws).sum(0)/ws.sum()

np.save("seq_vec128.npy", seq_vec)                         # ❷ 저장
print("✅ seq_vec128.npy / W.npy 저장 완료")
