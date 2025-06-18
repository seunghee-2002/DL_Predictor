# xgb_run_agg.py  ── 한 CSV+그래프에 통합
import os, gc, argparse, csv, json, numpy as np, xgboost as xgb
from tqdm import tqdm
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--epoch",  type=int, default=300)
ap.add_argument("--early",  type=int, default=30)
args = ap.parse_args()

# ────── 데이터 로드
X = load_npz("./DLResource/XGBData/X_128.npz")
Y = load_npz("./DLResource/XGBData/Y_topk.npz")
N_u, K = Y.shape
tr_idx, val_idx = train_test_split(np.arange(N_u), test_size=0.2, random_state=42)
Xtr, Xval, Ytr, Yval = X[tr_idx], X[val_idx], Y[tr_idx], Y[val_idx]

# ────── 공통 파라미터
base_par = dict(
    objective      = "binary:logistic",
    eval_metric    = ["logloss"],
    max_depth      = 8,
    learning_rate  = 0.1,
    reg_lambda     = 0.0,
    tree_method    = "gpu_hist",
    predictor      = "gpu_predictor",
    random_state   = 42,
)

# ────── 로그 누적용 컨테이너
loss_tr_all, loss_val_all = [], []    # 각 클래스별 curve 저장
f1_list = []

# ────── One-Vs-Rest 학습 루프
for col in tqdm(range(K), desc="train"):
    y_tr = Ytr[:, col].toarray().ravel()
    y_va = Yval[:, col].toarray().ravel()
    pos_w = (len(y_tr) - y_tr.sum()) / y_tr.sum() if y_tr.sum() else 1.0
    params = dict(base_par, scale_pos_weight=float(pos_w))

    dtr  = xgb.DMatrix(Xtr,  label=y_tr)
    dval = xgb.DMatrix(Xval, label=y_va)

    eval_hist = {}
    bst = xgb.train(
        params,
        dtr,
        num_boost_round       = args.epoch,
        evals                 = [(dtr, "train"), (dval, "val")],
        early_stopping_rounds = args.early if args.early else None,
        evals_result          = eval_hist,
        verbose_eval=False,
    )

    # ── 곡선 저장
    loss_tr_all.append(eval_hist["train"]["logloss"])
    loss_val_all.append(eval_hist["val"  ]["logloss"])

    # ── F1
    y_pred = (bst.predict(dval, iteration_range=(0, bst.best_iteration+1)) >= 0.4)
    f1_list.append(f1_score(y_va, y_pred, zero_division=0))

    del bst, dtr, dval; gc.collect()

# ────── ① CSV 저장  ───────────────────────────────────────
os.makedirs("logs", exist_ok=True)
max_round = max(len(l) for l in loss_tr_all)   # 클래스마다 조기종료 길이가 다름

with open("logs/metrics_log.csv", "w", newline="") as fp:
    writer = csv.writer(fp)
    header = ["round"] + [f"train_loss_c{c}" for c in range(K)] \
                       + [f"val_loss_c{c}"   for c in range(K)]
    writer.writerow(header)
    for r in range(max_round):
        row = [r]
        # 패딩: 없는 라운드는 None
        row += [loss_tr_all[c][r] if r < len(loss_tr_all[c]) else "" for c in range(K)]
        row += [loss_val_all[c][r] if r < len(loss_val_all[c]) else "" for c in range(K)]
        writer.writerow(row)

# ────── ② 전체 평균 곡선 시각화  ──────────────────────────
mean_tr = np.nanmean([np.pad(l, (0, max_round-len(l)), constant_values=np.nan)
                      for l in loss_tr_all], axis=0)
mean_val= np.nanmean([np.pad(l, (0, max_round-len(l)), constant_values=np.nan)
                      for l in loss_val_all], axis=0)

plt.figure(figsize=(6,4))
plt.plot(mean_tr,  label="train_logloss(Ø)")
plt.plot(mean_val, label="val_logloss(Ø)")
plt.xlabel("round"); plt.ylabel("logloss")
plt.title("Average logloss over all classes")
plt.legend(); plt.grid(True, ls="--", alpha=.3)
plt.tight_layout()
plt.savefig("logs/avg_logloss_curve.png")

# ────── ③ 최종 F1-micro 출력  ────────────────────────────
print(f"mean F1@0.4 (val) = {np.mean(f1_list):.4f}")
