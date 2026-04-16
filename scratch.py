import numpy as np

SEQ_LEN = 24

X = np.arange(100)

i = 30
# In fit:
xg_fit_0 = X[i]
xg_fit_1 = X[max(0, i-20)]
print(f"FIT XGB: base={xg_fit_0}, past={xg_fit_1}, slope interval={xg_fit_0 - xg_fit_1}")

# In predict:
X_scaled = X[i - SEQ_LEN + 1 : i + 1]
xg_pred_0 = X_scaled[-1]
xg_pred_1 = X_scaled[max(0, len(X_scaled)-20)]
print(f"PREDICT XGB: base={xg_pred_0}, past={xg_pred_1}, slope interval={xg_pred_0 - xg_pred_1}")

# For LSTM:
# sequence in fit:
xs = X[i - SEQ_LEN : i]
print(f"LSTM FIT ends at: {xs[-1]}, should be predicting y[{i}]")

# sequence in predict:
xs_pred = X_scaled
print(f"LSTM PRED ends at: {xs_pred[-1]}, predicting y[{i}]")
