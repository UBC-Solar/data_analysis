from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch

def evaluate_model(model, df_test_scaled, scaler, state_cols, control_cols, start_idx=0, n_samples=None):
    model.eval()
    device = next(model.parameters()).device
    n_states   = len(state_cols)
    n_controls = len(control_cols)
    cols       = state_cols + control_cols

    if n_samples is None:
        n_samples = len(df_test_scaled) - start_idx

    h, c = None, None
    all_y_true, all_y_pred = [], []

    for i in range(start_idx, start_idx + n_samples):
        row = df_test_scaled[cols].iloc[i].values.astype(np.float32)
        x_tensor = torch.tensor(row[:n_states]).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            hidden = (h, c) if h is not None else None
            y_pred_tensor, (h, c) = model(x_tensor, hidden=hidden)
            h, c = h.detach(), c.detach()

        all_y_true.append(row[n_states:])
        all_y_pred.append(y_pred_tensor.squeeze().cpu().numpy())

    # unscale
    def unscale(arr):
        dummy = np.zeros((len(arr), n_states + n_controls))
        dummy[:, n_states:] = arr
        return scaler.inverse_transform(dummy)[:, n_states:]

    y_true = unscale(np.array(all_y_true))
    y_pred = unscale(np.array(all_y_pred).reshape(-1, n_controls))


    brake_true = y_true[:, 0]
    brake_pred = y_pred[:, 0]
    accel_true = y_true[:, 1]
    accel_pred = y_pred[:, 1]

    brake_pred_binary = (brake_pred > 0.5).astype(int)
    brake_true_binary = (brake_true > 0.5).astype(int)

    print("=== accel_position (regression) ===")
    print(f"  MAE:  {mean_absolute_error(accel_true, accel_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(accel_true, accel_pred)):.4f}")
    print(f"  R²:   {r2_score(accel_true, accel_pred):.4f}")

    print("\n=== brake_pressed (classification) ===")
    print(f"  Accuracy:  {accuracy_score(brake_true_binary, brake_pred_binary):.4f}")
    print(f"  Precision: {precision_score(brake_true_binary, brake_pred_binary, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(brake_true_binary, brake_pred_binary, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(brake_true_binary, brake_pred_binary, zero_division=0):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(brake_true_binary, brake_pred):.4f}")

#evaluate_model(model, df_test_scaled, scaler, STATE_COLS, CONTROL_COLS)