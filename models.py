import os
import pickle
import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("xgboost not installed. Running LSTM-only mode.")

from config import (
    SEQ_LEN, TRAIN_EPOCHS, MODEL_DIR,
    REPLAY_BUFFER_SIZE, REPLAY_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


# ==================== LSTM ====================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.25):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden, num_layers=layers,
                               batch_first=True, dropout=dropout)
        self.norm    = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden, 3)   # 3 classes: short / long / neutral

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(self.norm(out[:, -1, :])))


# ==================== ENSEMBLE ====================

class EnsembleModel:
    """
    Combines an LSTM (sequence model) and XGBoost (flat features) via
    weighted averaging. Weights are updated after every closed trade
    based on rolling 30-trade accuracy of each component.
    """

    def __init__(self, feat_cols, symbol_tag='BTC'):
        self.feat_cols   = feat_cols
        self.symbol_tag  = symbol_tag
        self.n_feats     = len(feat_cols)

        # LSTM
        self.lstm     = LSTMModel(self.n_feats)
        self.lstm_opt = torch.optim.Adam(self.lstm.parameters(), lr=0.001,
                                         weight_decay=1e-5)

        # XGBoost
        if XGB_AVAILABLE:
            self.xgb = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1,
            )
            self.xgb_trained = False
        else:
            self.xgb         = None
            self.xgb_trained = False

        # Scaler
        self.scaler = MinMaxScaler()

        # Ensemble weights
        self.lstm_weight = 0.5
        self.xgb_weight  = 0.5

        # Rolling accuracy trackers (last 30 trades)
        self._lstm_acc = deque(maxlen=30)
        self._xgb_acc  = deque(maxlen=30)

        # Store last raw probabilities for weight updates
        self._last_lstm_probs = None
        self._last_xgb_probs  = None

        # Replay buffer for online LSTM updates
        self.replay = deque(maxlen=REPLAY_BUFFER_SIZE)

    # ---- helpers ----

    def _make_sequences(self, X_scaled):
        seqs = np.array([X_scaled[i - SEQ_LEN:i] for i in range(SEQ_LEN, len(X_scaled))])
        return torch.FloatTensor(seqs)

    def _xgb_features(self, X_scaled):
        """
        Flat features for XGBoost: last candle + slope features
        computed as differences across the sequence window.
        """
        last   = X_scaled[-1]
        slopes = (X_scaled[-1] - X_scaled[max(0, len(X_scaled)-20)]) / 20.0
        return np.concatenate([last, slopes]).reshape(1, -1)

    # ---- training ----

    def fit(self, df, labels):
        """Initial training on historical data."""
        X = self.scaler.fit_transform(df[self.feat_cols].values)
        y = labels

        # Sequences for LSTM (drops first SEQ_LEN rows)
        Xs     = self._make_sequences(X)
        ys_seq = torch.LongTensor(y[SEQ_LEN:])

        # Flat features for XGBoost
        X_flat = np.array([
            np.concatenate([X[i], (X[i] - X[max(0, i-20)]) / 20.0])
            for i in range(SEQ_LEN, len(X))
        ])
        y_flat = y[SEQ_LEN:]

        # ---- Train LSTM ----
        logger.info("Training LSTM...")
        self.lstm.train()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.lstm_opt, T_max=TRAIN_EPOCHS
        )
        for epoch in range(TRAIN_EPOCHS):
            self.lstm_opt.zero_grad()
            loss = nn.CrossEntropyLoss()(self.lstm(Xs), ys_seq)
            loss.backward()
            nn.utils.clip_grad_norm_(self.lstm.parameters(), 1.0)
            self.lstm_opt.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"  LSTM epoch {epoch+1}/{TRAIN_EPOCHS} | loss {loss.item():.4f}")

        # ---- Train XGBoost ----
        if self.xgb is not None:
            logger.info("Training XGBoost...")
            split    = int(len(X_flat) * 0.85)
            eval_set = [(X_flat[split:], y_flat[split:])]
            self.xgb.fit(
                X_flat[:split], y_flat[:split],
                eval_set=eval_set,
                verbose=False,
            )
            self.xgb_trained = True

        return self

    # ---- inference ----

    def predict(self, df_window):
        """
        Returns (pred_class, confidence, x_tensor).
        x_tensor is saved for online learning later.
        """
        X    = self.scaler.transform(df_window[self.feat_cols].values[-SEQ_LEN:])
        xt   = torch.FloatTensor(X).unsqueeze(0)      # (1, SEQ_LEN, n_feats)
        xf   = self._xgb_features(X)                  # (1, 2*n_feats)

        # LSTM probs
        self.lstm.eval()
        with torch.no_grad():
            lstm_p = torch.softmax(self.lstm(xt), dim=1)[0].numpy()

        # XGBoost probs
        if self.xgb_trained:
            xgb_p = self.xgb.predict_proba(xf)[0]
        else:
            xgb_p = lstm_p.copy()

        self._last_lstm_probs = lstm_p
        self._last_xgb_probs  = xgb_p

        combined = self.lstm_weight * lstm_p + self.xgb_weight * xgb_p
        pred     = int(combined.argmax())
        conf     = float(combined.max())

        return pred, conf, xt

    # ---- feedback & online learning ----

    def record_outcome(self, actual_label):
        """
        Called after a trade closes. Updates ensemble weights
        based on which model was more accurate recently.
        """
        if self._last_lstm_probs is None:
            return

        lstm_right = int(np.argmax(self._last_lstm_probs) == actual_label)
        xgb_right  = int(np.argmax(self._last_xgb_probs)  == actual_label)

        self._lstm_acc.append(lstm_right)
        self._xgb_acc.append(xgb_right)

        if len(self._lstm_acc) >= 10:
            la   = np.mean(self._lstm_acc) + 1e-6
            xa   = np.mean(self._xgb_acc)  + 1e-6
            tot  = la + xa
            self.lstm_weight = la / tot
            self.xgb_weight  = xa / tot
            logger.info(f"Ensemble weights → LSTM: {self.lstm_weight:.2f} | XGB: {self.xgb_weight:.2f}")

    def online_update(self, x_tensor, label):
        """
        Adds trade to replay buffer and trains LSTM on a random mini-batch.
        Uses the persistent optimizer so momentum accumulates across updates.
        """
        self.replay.append((x_tensor, label))

        if len(self.replay) < REPLAY_BATCH_SIZE:
            logger.info(f"Replay buffer: {len(self.replay)}/{REPLAY_BATCH_SIZE}")
            return

        idx     = np.random.choice(len(self.replay), REPLAY_BATCH_SIZE, replace=False)
        batch_x = torch.cat([self.replay[i][0] for i in idx], dim=0)
        batch_y = torch.LongTensor([self.replay[i][1] for i in idx])

        self.lstm.train()
        self.lstm_opt.zero_grad()
        loss = nn.CrossEntropyLoss()(self.lstm(batch_x), batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.lstm.parameters(), 1.0)
        self.lstm_opt.step()

        logger.info(f"Online update | loss: {loss.item():.4f} | buffer: {len(self.replay)}")

    # ---- persistence ----

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        tag = self.symbol_tag
        torch.save(self.lstm.state_dict(),     os.path.join(MODEL_DIR, f'lstm_{tag}.pt'))
        torch.save(self.lstm_opt.state_dict(), os.path.join(MODEL_DIR, f'lstm_opt_{tag}.pt'))
        with open(os.path.join(MODEL_DIR, f'scaler_{tag}.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        if self.xgb_trained:
            with open(os.path.join(MODEL_DIR, f'xgb_{tag}.pkl'), 'wb') as f:
                pickle.dump(self.xgb, f)
        logger.info(f"Model saved ({tag})")

    def load(self):
        tag       = self.symbol_tag
        lstm_path  = os.path.join(MODEL_DIR, f'lstm_{tag}.pt')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{tag}.pkl')
        xgb_path   = os.path.join(MODEL_DIR, f'xgb_{tag}.pkl')

        if not (os.path.exists(lstm_path) and os.path.exists(scaler_path)):
            return False

        self.lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))
        opt_path = os.path.join(MODEL_DIR, f'lstm_opt_{tag}.pt')
        if os.path.exists(opt_path):
            self.lstm_opt.load_state_dict(torch.load(opt_path, map_location='cpu'))
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        if os.path.exists(xgb_path) and self.xgb is not None:
            with open(xgb_path, 'rb') as f:
                self.xgb = pickle.load(f)
            self.xgb_trained = True

        logger.info(f"Model loaded ({tag})")
        return True
