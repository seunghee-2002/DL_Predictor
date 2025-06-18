import numpy as np

# --- 파일 경로 확인 ---
ev16_path = "./DLResource/ev_final16.npy"
W_path    = "W.npy"                 # 방금 저장한 것
out_path  = "product128.npy"

# --- 로드 ---
ev16 = np.load(ev16_path).astype(np.float32)  # (49 238, 16)
W    = np.load(W_path).astype(np.float32)     # (16 ,128)

# --- 변환 ---
P128 = ev16 @ W                               # (49 238, 128)

# --- 저장 ---
np.save(out_path, P128)
print("✅ product128.npy 저장 완료 :", P128.shape)
