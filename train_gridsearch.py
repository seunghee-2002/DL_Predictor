import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import argparse
from custom_dataset import OrderSequenceDataset, custom_collate_fn
from model_cnn import CNNPredictorModel

def train_n_epochs(model, loader, optimizer, criterion, device, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        bar = tqdm(loader, desc=f"[Train epoch {epoch+1}/{n_epochs}]", leave=False)
        for products_x, meta_x, targets in bar:
            products_x, meta_x, targets = products_x.to(device), meta_x.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(products_x, meta_x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

def eval_microf1(model, val_loader, device, threshold=0.4):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for products_x, meta_x, targets in tqdm(val_loader, desc="[Eval]", leave=False):
            products_x, meta_x = products_x.to(device), meta_x.to(device)
            logits = model(products_x, meta_x)
            all_logits.append(logits.cpu())
            all_targets.append(targets)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def gridsearch_pool_outsize(
    model_class, model_kwargs_base, train_loader, val_loader, device,
    pool_outsize_list=[8,16,32],
    n_epochs=3,
    pos_weight=15,
    threshold=0.4,
    lr=0.001
):
    best_f1 = -1
    best_pool = None
    results = []
    for outsize in pool_outsize_list:
        print(f"\n[GridSearch] AdaptiveMaxPool1d output_size={outsize}")
        model_kwargs = model_kwargs_base.copy()
        model_kwargs['pool_outsize'] = outsize
        model = model_class(**model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        # 3 epoch 학습
        train_n_epochs(model, train_loader, optimizer, criterion, device, n_epochs=n_epochs)
        # 검증평가
        precision, recall, f1 = eval_microf1(model, val_loader, device, threshold=threshold)
        results.append((outsize, precision, recall, f1))
        print(f"output_size={outsize} → Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_pool = outsize
    print(f"\n최고 F1={best_f1:.4f} → output_size={best_pool}")
    print("전체 결과:", results)
    return results, best_pool

def main():
    parser = argparse.ArgumentParser(description="Grid search for AdaptiveMaxPool1d output_size in CNN (3 epoch quick test)")
    # 데이터 경로 인자
    parser.add_argument("--data_x_product_path", type=str, default="./DLResource/PreprocessData/data_X_product_5.csv", help="전체 X_product_N.csv 파일 경로")
    parser.add_argument("--data_x_meta_path", type=str, default="./DLResource/PreprocessData/data_X_5.csv", help="주문 헤더 feature 데이터 경로")
    parser.add_argument("--data_y_path", type=str, default="./DLResource/PreprocessData/data_Y_5.csv", help="전체 Y_N.csv 파일 경로")
    parser.add_argument("--embeddings_path", type=str, default="./DLResource/ev_final64.npy", help="제품 임베딩 파일 경로")
    parser.add_argument("--products_path", type=str, default="./DLResource/RawData/products.csv", help="제품 메타데이터 파일 경로")
    # 데이터 구조 인자
    parser.add_argument("--N_orders", type=int, default=5)
    parser.add_argument("--max_products_per_order", type=int, default=145)
    parser.add_argument("--product_emb_dim", type=int, default=64)
    # 모델 구조 인자
    parser.add_argument("--cnn_out_channels", type=int, default=128)
    parser.add_argument("--cnn_kernel_size", type=int, default=4)
    parser.add_argument("--gru_hidden_dim", type=int, default=256)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--prediction_head_inter_dim", type=int, default=256)
    # 학습 관련 인자
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    # grid search 인자
    parser.add_argument("--pool_outsize_list", type=int, nargs="+", default=[8,16,32])
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--pos_weight", type=float, default=15)
    parser.add_argument("--threshold", type=float, default=0.4)
    # train/val split
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로딩
    dataset = OrderSequenceDataset(
        data_x_product_path=args.data_x_product_path,
        data_x_meta_path=args.data_x_meta_path,
        data_y_path=args.data_y_path,
        embeddings_npy_path=args.embeddings_path,
        products_path=args.products_path,
        N_orders=args.N_orders,
        max_products_per_order=args.max_products_per_order,
        embedding_dim=args.product_emb_dim
    )
    order_meta_dim = dataset.meta_feature_dim
    num_total_products = dataset.num_total_products

    total_samples = len(dataset)
    val_samples = int(args.val_ratio * total_samples)
    train_samples = total_samples - val_samples
    train_set, val_set = random_split(
        dataset,
        [train_samples, val_samples],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 모델 인자 dict 준비 (pool_outsize는 나중에 넣음)
    model_kwargs_base = dict(
        num_total_products=num_total_products,
        product_emb_dim=args.product_emb_dim,
        cnn_out_channels=args.cnn_out_channels,
        cnn_kernel_size=args.cnn_kernel_size,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        dropout_rate=args.dropout_rate,
        prediction_head_inter_dim=args.prediction_head_inter_dim,
        order_meta_dim=order_meta_dim
    )

    # grid search 실행
    gridsearch_pool_outsize(
        CNNPredictorModel,
        model_kwargs_base,
        train_loader,
        val_loader,
        device,
        pool_outsize_list=args.pool_outsize_list,
        n_epochs=args.n_epochs,
        pos_weight=args.pos_weight,
        threshold=args.threshold,
        lr=args.learning_rate
    )

if __name__ == "__main__":
    main()
