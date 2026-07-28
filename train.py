import torch
from torch.utils.data import DataLoader

from config import Config
from utils.data_loader import load_qos_data, density_split, build_mask, QoSDataset
from utils.metrics import MAE, RMSE
from models.full_model import QoSModel
from utils.logger import log


def evaluate(model, loader):
    model.eval()
    preds, trues, masks = [], [], []

    with torch.no_grad():
        for x, m in loader:
            x, m = x.to(Config.device), m.to(Config.device)
            pred, _, _ = model(x)

            preds.append(pred)
            trues.append(x[:, -1])
            masks.append(m[:, -1])

    preds = torch.cat(preds)
    trues = torch.cat(trues)
    masks = torch.cat(masks)

    return MAE(preds, trues, masks), RMSE(preds, trues, masks)


def train():
    Q = load_qos_data("data/tpdata.txt")

    for density in [0.05, 0.1, 0.15, 0.2]:
        print(f"\n===== Density {density*100:.0f}% =====")

        train_idx, val_idx, test_idx = density_split(Q, density)

        train_mask = build_mask(Q, train_idx)
        val_mask = build_mask(Q, val_idx)
        test_mask = build_mask(Q, test_idx)

        train_loader = DataLoader(QoSDataset(Q, train_mask), batch_size=Config.batch_size, shuffle=True)
        val_loader = DataLoader(QoSDataset(Q, val_mask), batch_size=Config.batch_size)
        test_loader = DataLoader(QoSDataset(Q, test_mask), batch_size=Config.batch_size)

        model = QoSModel().to(Config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

        for epoch in range(Config.epochs):
            model.train()
            total_loss = 0

            for x, m in train_loader:
                x, m = x.to(Config.device), m.to(Config.device)

                pred, w, loss_a = model(x)

                target = x[:, -1]
                mask = m[:, -1]

                loss_p = ((pred - target) ** 2 * mask * w).mean()
                loss = loss_p + Config.lambda_a * loss_a

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            mae, rmse = evaluate(model, val_loader)
            log(epoch+1, total_loss, mae, rmse)

        test_mae, test_rmse = evaluate(model, test_loader)
        print(f"TEST RESULT | MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    train()