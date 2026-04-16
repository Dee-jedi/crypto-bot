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
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden, num_layers=layers,
                               batch_first=True, dropout=dropout)
        self.norm    = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden, 64)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(64, 2)   # 2 classes: short(0) / long(1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.dropout(self.norm(out[:, -1, :]))
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ==================== ENSEMBLE ====================

class EnsembleModel:
    """
    Combines an LSTM (sequence model) and XGBoost (flat features) via
    weighted averaging.  Weights are updated after every closed trade
    based on rolling 30-trade accuracy of each component.

    KEY FIX: Added evaluate() method so the caller can check whether the
    model is actually useful before using it as a trade gate.  A model
    with < 52 % binary accuracy is no better than random and should be
    retrained, not deployed.
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
                eval_metric='logloss',
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

        # Rolling accuracy trackers (last 30 trades) — total + per-direction
        self._lstm_acc       = deque(maxlen=30)
        self._xgb_acc        = deque(maxlen=30)
        self._lstm_acc_long  = deque(maxlen=30)
        self._xgb_acc_long   = deque(maxlen=30)
        self._lstm_acc_short = deque(maxlen=30)
        self._xgb_acc_short  = deque(maxlen=30)

        # Replay buffer for online LSTM updates and periodic XGB refits
        self.replay               = deque(maxlen=REPLAY_BUFFER_SIZE)
        self._xgb_refit_every     = 200
        self._online_update_count = 0

        # FIX: track whether fit() completed successfully
        self.is_fitted = False

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_sequences(self, X_scaled):
        """
        Build sliding-window sequences from a 2-D scaled array.

        Output shape : (len(X_scaled) - SEQ_LEN,  SEQ_LEN,  n_feats)
        Label slice  : y[SEQ_LEN:]   — same number of rows, correct alignment.
        """
        n = len(X_scaled)
        if n <= SEQ_LEN:
            raise ValueError(
                f"Not enough rows ({n}) to build sequences of length {SEQ_LEN}."
            )
        seqs = np.stack(
            [X_scaled[i : i + SEQ_LEN] for i in range(n - SEQ_LEN)],
            axis=0,
        )
        return torch.FloatTensor(seqs)

    def _xgb_features(self, X_scaled):
        """
        Flat feature vector for XGBoost.
        Safe window prevents wrong denominator when SEQ_LEN < 21.
        """
        last      = X_scaled[-1]
        look_back = min(20, len(X_scaled) - 1)
        if look_back > 0:
            slopes = (X_scaled[-1] - X_scaled[-(look_back + 1)]) / look_back
        else:
            slopes = np.zeros_like(last)
        return np.concatenate([last, slopes]).reshape(1, -1)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, df, labels):
        """
        Initial training on historical data.
        Drops NEUTRAL(2) rows to prevent model collapse on imbalanced data.

        Returns self for chaining. Sets self.is_fitted = True on success.
        """
        # FIX: guard against being called with empty training data
        if df is None or len(df) <= SEQ_LEN + 10:
            raise ValueError(
                f"Training DataFrame has only {len(df) if df is not None else 0} rows. "
                f"Need at least {SEQ_LEN + 10}. "
                f"Check that TRAIN_RATIO is > 0 (e.g. 0.75)."
            )

        X = self.scaler.fit_transform(df[self.feat_cols].values)
        y = labels

        Xs_all  = self._make_sequences(X)
        ys_all  = y[SEQ_LEN:]

        mask    = ys_all != 2
        Xs      = Xs_all[mask]
        ys_seq  = torch.LongTensor(ys_all[mask])

        n_long  = (ys_all[mask] == 1).sum()
        n_short = (ys_all[mask] == 0).sum()
        logger.info(
            f"  Binary filter: {mask.sum():,} directional / {len(mask):,} total "
            f"({mask.sum() / len(mask) * 100:.1f}%) | long={n_long} short={n_short}"
        )

        X_flat_all = np.array([
            self._xgb_features(X[: i + SEQ_LEN]).flatten()
            for i in range(len(Xs_all))
        ])
        X_flat = X_flat_all[mask]
        y_flat = ys_all[mask]

        # ---- Validation split (last 15% of directional samples) ----
        val_size = max(50, int(len(Xs) * 0.15))
        Xs_tr, Xs_val  = Xs[:-val_size], Xs[-val_size:]
        ys_tr, ys_val  = ys_seq[:-val_size], ys_seq[-val_size:]
        Xf_tr, Xf_val  = X_flat[:-val_size], X_flat[-val_size:]
        yf_tr, yf_val  = y_flat[:-val_size], y_flat[-val_size:]

        logger.info(f"  Train samples: {len(Xs_tr)} | Val samples: {len(Xs_val)}")

        # ---- Train LSTM ----
        logger.info("Training LSTM...")
        self.lstm.train()
        batch_size = 256
        n_samples  = len(Xs_tr)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.lstm_opt, T_max=TRAIN_EPOCHS
        )
        best_val_acc     = 0.0
        best_state       = None
        patience_counter = 0

        for epoch in range(TRAIN_EPOCHS):
            self.lstm.train()
            indices    = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, n_samples, batch_size):
                end     = min(start + batch_size, n_samples)
                idx     = indices[start:end]
                batch_x = Xs_tr[idx]
                batch_y = ys_tr[idx]

                self.lstm_opt.zero_grad()
                loss = nn.CrossEntropyLoss()(self.lstm(batch_x), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.lstm.parameters(), 1.0)
                self.lstm_opt.step()
                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step()

            # Validate every epoch for early stopping logic
            self.lstm.eval()
            with torch.no_grad():
                val_logits = self.lstm(Xs_val)
                val_preds  = val_logits.argmax(dim=1)
                val_acc    = (val_preds == ys_val).float().mean().item()

            # Keep best weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in self.lstm.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    f"  LSTM epoch {epoch + 1}/{TRAIN_EPOCHS} | "
                    f"loss {avg_loss:.4f} | val_acc {val_acc:.3f}"
                )

            if patience_counter >= 15: # Early Stopping Patience (Strict Epoch Count)
                logger.info(f"  Early stopping triggered at epoch {epoch + 1} (Restoring Best Weights from Epoch {epoch + 1 - patience_counter})")
                break

        # Restore best weights
        if best_state is not None:
            self.lstm.load_state_dict(best_state)
            logger.info(f"  LSTM best val accuracy: {best_val_acc:.3f}")

        # ---- Train XGBoost ----
        if self.xgb is not None:
            # Guard: Ensure we have at least 2 classes in training/val for XGB
            unique_tr  = np.unique(yf_tr)
            unique_val = np.unique(yf_val)
            
            if len(unique_tr) < 2 or len(unique_val) < 2:
                logger.warning(f"  XGBoost skipped: insufficient class diversity (Tr:{len(unique_tr)} Val:{len(unique_val)})")
                self.xgb_trained = False
            else:
                logger.info("Training XGBoost...")
                self.xgb.fit(
                    Xf_tr, yf_tr,
                    eval_set=[(Xf_val, yf_val)],
                    verbose=False,
                )
                self.xgb_trained = True

                xgb_val_acc = (self.xgb.predict(Xf_val) == yf_val).mean()
                logger.info(f"  XGBoost val accuracy: {xgb_val_acc:.3f}")

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  FIX: Model validation / pre-deployment check                       #
    # ------------------------------------------------------------------ #

    def evaluate(self, df, labels):
        """
        Run the model on a held-out slice and return per-class accuracy.

        Call this BEFORE using the model as a trade gate.  If accuracy is
        below ~0.52 the model is performing at chance and should be retrained.

        Returns
        -------
        dict with keys: lstm_acc, xgb_acc, ensemble_acc, n_samples
        """
        X      = self.scaler.transform(df[self.feat_cols].values)
        y      = labels
        Xs_all = self._make_sequences(X)
        ys_all = y[SEQ_LEN:]
        mask   = ys_all != 2

        if mask.sum() < 20:
            logger.warning("evaluate(): fewer than 20 directional samples — skipping.")
            return {'lstm_acc': None, 'xgb_acc': None, 'ensemble_acc': None, 'n_samples': 0}

        Xs     = Xs_all[mask]
        ys     = ys_all[mask]

        # LSTM
        self.lstm.eval()
        with torch.no_grad():
            lstm_preds = self.lstm(Xs).argmax(dim=1).numpy()
        lstm_acc = (lstm_preds == ys).mean()

        # XGBoost
        if self.xgb_trained:
            X_flat_all = np.array([
                self._xgb_features(X[: i + SEQ_LEN]).flatten()
                for i in range(len(Xs_all))
            ])
            Xf    = X_flat_all[mask]
            xgb_preds = self.xgb.predict(Xf)
            xgb_acc   = (xgb_preds == ys).mean()
        else:
            xgb_acc   = lstm_acc  # fallback
            xgb_preds = lstm_preds

        # Ensemble
        self.lstm.eval()
        with torch.no_grad():
            lstm_probs = torch.softmax(self.lstm(Xs), dim=1).numpy()
        if self.xgb_trained:
            xgb_probs = self.xgb.predict_proba(Xf)
        else:
            xgb_probs = lstm_probs.copy()
        combined      = self.lstm_weight * lstm_probs + self.xgb_weight * xgb_probs
        ens_preds     = combined.argmax(axis=1)
        ensemble_acc  = (ens_preds == ys).mean()

        logger.info(
            f"[evaluate] n={mask.sum()} | "
            f"LSTM acc={lstm_acc:.3f} | XGB acc={xgb_acc:.3f} | "
            f"Ensemble acc={ensemble_acc:.3f}"
        )
        return {
            'lstm_acc':     float(lstm_acc),
            'xgb_acc':      float(xgb_acc),
            'ensemble_acc': float(ensemble_acc),
            'n_samples':    int(mask.sum()),
        }

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, df_window):
        """
        Returns (pred_class, confidence, x_tensor, lstm_probs, xgb_probs).

        x_tensor, lstm_probs, xgb_probs should be stored on the trade object
        and passed back into record_outcome / online_update after the trade
        closes.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call model.fit() or model.load() first."
            )

        X  = self.scaler.transform(df_window[self.feat_cols].values[-SEQ_LEN:])
        xt = torch.FloatTensor(X).unsqueeze(0)
        xf = self._xgb_features(X)

        self.lstm.eval()
        with torch.no_grad():
            lstm_p = torch.softmax(self.lstm(xt), dim=1)[0].numpy()

        if self.xgb_trained:
            xgb_p = self.xgb.predict_proba(xf)[0]
        else:
            xgb_p = lstm_p.copy()

        combined = self.lstm_weight * lstm_p + self.xgb_weight * xgb_p
        pred     = int(combined.argmax())
        conf     = float(combined.max())

        return pred, conf, xt, lstm_p, xgb_p

    # ------------------------------------------------------------------ #
    #  Feedback & online learning                                          #
    # ------------------------------------------------------------------ #

    def record_outcome(self, actual_label, lstm_probs, xgb_probs):
        lstm_right = int(np.argmax(lstm_probs) == actual_label)
        xgb_right  = int(np.argmax(xgb_probs)  == actual_label)

        self._lstm_acc.append(lstm_right)
        self._xgb_acc.append(xgb_right)

        if actual_label == 1:
            self._lstm_acc_long.append(lstm_right)
            self._xgb_acc_long.append(xgb_right)
        else:
            self._lstm_acc_short.append(lstm_right)
            self._xgb_acc_short.append(xgb_right)

        if len(self._lstm_acc) >= 10:
            la  = np.mean(self._lstm_acc) + 1e-6
            xa  = np.mean(self._xgb_acc)  + 1e-6
            tot = la + xa
            self.lstm_weight = la / tot
            self.xgb_weight  = xa / tot

            la_l = np.mean(self._lstm_acc_long)  if self._lstm_acc_long  else 0.0
            la_s = np.mean(self._lstm_acc_short) if self._lstm_acc_short else 0.0
            xa_l = np.mean(self._xgb_acc_long)   if self._xgb_acc_long   else 0.0
            xa_s = np.mean(self._xgb_acc_short)  if self._xgb_acc_short  else 0.0

            logger.info(
                f"Ensemble weights → LSTM: {self.lstm_weight:.2f} | XGB: {self.xgb_weight:.2f} | "
                f"LSTM acc long/short: {la_l:.2f}/{la_s:.2f} | "
                f"XGB  acc long/short: {xa_l:.2f}/{xa_s:.2f}"
            )

    def online_update(self, x_tensor, label):
        """
        Adds closed trade to replay buffer and trains LSTM on a mini-batch.
        NOTE: Only call this in LIVE trading, not during backtesting —
        calling it during a backtest contaminates the out-of-sample test.
        """
        self.replay.append((x_tensor, label))
        self._online_update_count += 1

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

        logger.info(
            f"Online LSTM update | loss: {loss.item():.4f} | "
            f"buffer: {len(self.replay)} | update #{self._online_update_count}"
        )

        if (
            self.xgb is not None
            and self._online_update_count % self._xgb_refit_every == 0
            and len(self.replay) >= REPLAY_BATCH_SIZE
        ):
            self._refit_xgb_on_replay()

    def _refit_xgb_on_replay(self):
        if self.xgb is None:
            return
        logger.info("Periodic XGBoost refit on replay buffer...")
        try:
            all_x  = np.array([e[0].squeeze(0).numpy()[-1] for e in self.replay])
            slopes = np.zeros_like(all_x)
            X_flat = np.concatenate([all_x, slopes], axis=1)
            y_flat = np.array([e[1] for e in self.replay])

            if len(np.unique(y_flat)) < 2:
                logger.warning("XGB refit skipped: only one class in replay buffer.")
                return

            self.xgb.fit(X_flat, y_flat, verbose=False)
            self.xgb_trained = True
            logger.info(f"  XGBoost refit complete on {len(y_flat)} samples.")
        except Exception as e:
            logger.error(f"XGBoost refit failed: {e}")

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        tag = self.symbol_tag

        torch.save(self.lstm.state_dict(),
                   os.path.join(MODEL_DIR, f'lstm_{tag}.pt'))
        torch.save(self.lstm_opt.state_dict(),
                   os.path.join(MODEL_DIR, f'lstm_opt_{tag}.pt'))

        with open(os.path.join(MODEL_DIR, f'scaler_{tag}.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        if self.xgb_trained:
            with open(os.path.join(MODEL_DIR, f'xgb_{tag}.pkl'), 'wb') as f:
                pickle.dump(self.xgb, f)

        meta = {
            'lstm_weight':          self.lstm_weight,
            'xgb_weight':           self.xgb_weight,
            'online_update_count':  self._online_update_count,
            'is_fitted':            self.is_fitted,
        }
        with open(os.path.join(MODEL_DIR, f'meta_{tag}.pkl'), 'wb') as f:
            pickle.dump(meta, f)

        logger.info(f"Model saved ({tag})")

    def load(self):
        tag         = self.symbol_tag
        lstm_path   = os.path.join(MODEL_DIR, f'lstm_{tag}.pt')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{tag}.pkl')
        xgb_path    = os.path.join(MODEL_DIR, f'xgb_{tag}.pkl')
        opt_path    = os.path.join(MODEL_DIR, f'lstm_opt_{tag}.pt')
        meta_path   = os.path.join(MODEL_DIR, f'meta_{tag}.pkl')

        if not (os.path.exists(lstm_path) and os.path.exists(scaler_path)):
            return False

        self.lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))
        if os.path.exists(opt_path):
            self.lstm_opt.load_state_dict(torch.load(opt_path, map_location='cpu'))

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        if os.path.exists(xgb_path) and self.xgb is not None:
            with open(xgb_path, 'rb') as f:
                self.xgb = pickle.load(f)
            self.xgb_trained = True

        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.lstm_weight          = meta.get('lstm_weight', 0.5)
            self.xgb_weight           = meta.get('xgb_weight', 0.5)
            self._online_update_count = meta.get('online_update_count', 0)
            self.is_fitted            = meta.get('is_fitted', True)  # assume fitted if file exists

        logger.info(
            f"Model loaded ({tag}) | "
            f"weights LSTM/XGB: {self.lstm_weight:.2f}/{self.xgb_weight:.2f}"
        )
        return True