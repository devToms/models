import functools
import collections
from river import compose, linear_model, preprocessing, metrics, drift, optim, stats
from pathlib import Path
import sys
import os
import pickle
import gzip
from datetime import datetime
import time
import signal
import json
import asyncio
import websockets
import numpy as np
import math
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from helpers.csv_data_reader import CsvDataReader
from loggers.light_logger import LightLogger

training_log_file = "river_training_v10_log.txt"
training_logger = LightLogger(training_log_file, logger_name="training_logger")


class Std(stats.Var):
    def get(self):
        return math.sqrt(super().get())

stats.Std = Std
# Zmiana w klasie EnhancedEMA:
class EnhancedEMA:
    """Rozszerzona implementacja EMA z adaptacyjnym współczynnikiem"""
    def __init__(self, period: int = 5):
        self.period = period
        self.value = None
        self.alpha = 2 / (period + 1)
        self.volatility = stats.Var()  # Zmienione z Std() na Var()
        
    def update(self, price: float) -> 'EnhancedEMA':
        if self.value is None:
            self.value = price
        else:
            # Dynamiczne alpha w zależności od zmienności
            self.volatility.update(price)
            # Oblicz odchylenie standardowe jako pierwiastek z wariancji
            std_dev = math.sqrt(self.volatility.get())
            vol_adjusted_alpha = self.alpha * (1 + 0.5 * math.tanh(std_dev * 10000))
            self.value = price * vol_adjusted_alpha + self.value * (1 - vol_adjusted_alpha)
        return self
    
    def get(self) -> float:
        return self.value if self.value is not None else 0

    # W metodzie _setup_indicators():
    def _setup_indicators(self):
        self.indicators = {
            'ema_short': EnhancedEMA(5),
            'ema_long': EnhancedEMA(20),
            'sma_short': stats.Mean(),
            'sma_long': stats.Mean(),
            'volatility': stats.Var(),  # Zmienione z Std() na Var()
            'momentum': stats.Mean()
        }
        self.price_buffer = collections.deque(maxlen=100)
        self.min_pips_threshold = 0.00015
        self.warning_threshold = 0.15

    # W metodzie _generate_features():
    def _generate_features(self, bid: float, ask: float) -> Dict[str, float]:
        spread = ask - bid
        momentum = bid - self.prev_bid if self.prev_bid else 0
        volatility = math.sqrt(self.indicators['volatility'].get())  # Oblicz odchylenie standardowe
        
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'momentum': momentum,
            'momentum_acc': momentum - (self.prev_bid - self.prev_prev_bid) if self.prev_prev_bid else 0,
            'ema_short': self.indicators['ema_short'].get(),
            'ema_long': self.indicators['ema_long'].get(),
            'ema_ratio': self.indicators['ema_short'].get() / (self.indicators['ema_long'].get() + 1e-8),
            'sma_short': self.indicators['sma_short'].get(),
            'sma_long': self.indicators['sma_long'].get(),
            'sma_diff': self.indicators['sma_short'].get() - self.indicators['sma_long'].get(),
            'volatility': volatility,  # Użyj obliczonego odchylenia standardowego
            'volatility_ratio': volatility / (bid + 1e-8),
            'rsi': self._calculate_rsi(),
            'bollinger': (bid - self.indicators['sma_long'].get()) / (2 * volatility + 1e-8),
            'mean_price': np.mean(self.price_buffer) if self.price_buffer else bid
        }

class RiverModelTrainer:
    def __init__(self, model_save_path="river_model_v10.pkl.gz"):
        self.logger = training_logger
        self.model_save_path = model_save_path
        
        
        self.prediction_horizon = 300  # 5 minut w sekundach
        self.reassessment_interval = 150  # 2.5 minuty
        self.last_prediction_time = None
        self.current_position = None
        self.prediction_buffer = collections.deque(maxlen=10)
       
        
        # Poprawione: nie nadpisuj modelu!
        self.model = self._load_or_init_model()
        
        self.metrics = {
            'accuracy': metrics.Accuracy(),
            'precision': metrics.Precision(),
            'recall': metrics.Recall()
        }
        self.drift_detector = drift.ADWIN(delta=0.002)
        self._setup_indicators()
        self._init_tracking()
        
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        
    def _init_model(self):
        return compose.Pipeline(
            ('features', preprocessing.StandardScaler()),
            ('model', linear_model.LogisticRegression(
                optimizer=optim.SGD(0.015),
                l2=0.03,
                intercept_lr=0.2
            ))
        )

    def _setup_indicators(self):
        self.indicators = {
            'ema_short': EnhancedEMA(5),
            'ema_long': EnhancedEMA(20),
            'sma_short': stats.Mean(),
            'sma_long': stats.Mean(),
            'volatility': stats.Std(),
            'momentum': stats.Mean()
        }
        self.price_buffer = collections.deque(maxlen=100)
        self.min_pips_threshold = 0.00015
        self.warning_threshold = 0.15

    def _init_tracking(self):
        self.tick_count = 0
        self.save_interval = 100
        self.benchmark = {'correct': 0, 'total': 0}
        self.prev_bid = None
        self.prev_prev_bid = None

    @functools.lru_cache(maxsize=1000)
    def _calculate_rsi(self):
        """Uproszczone RSI na podstawie bufora cen"""
        if len(self.price_buffer) < 14:
            return 50
            
        gains = [max(0, self.price_buffer[i] - self.price_buffer[i-1]) 
                for i in range(1, len(self.price_buffer))]
        losses = [max(0, self.price_buffer[i-1] - self.price_buffer[i]) 
                for i in range(1, len(self.price_buffer))]
                
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        return 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-8))))

    def _generate_features(self, bid: float, ask: float) -> Dict[str, float]:
        spread = ask - bid
        momentum = bid - self.prev_bid if self.prev_bid else 0
        
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'momentum': momentum,
            'momentum_acc': momentum - (self.prev_bid - self.prev_prev_bid) if self.prev_prev_bid else 0,
            'ema_short': self.indicators['ema_short'].get(),
            'ema_long': self.indicators['ema_long'].get(),
            'ema_ratio': self.indicators['ema_short'].get() / (self.indicators['ema_long'].get() + 1e-8),
            'sma_short': self.indicators['sma_short'].get(),
            'sma_long': self.indicators['sma_long'].get(),
            'sma_diff': self.indicators['sma_short'].get() - self.indicators['sma_long'].get(),
            'volatility': self.indicators['volatility'].get(),
            'volatility_ratio': self.indicators['volatility'].get() / (bid + 1e-8),
            'rsi': self._calculate_rsi(),
            'bollinger': (bid - self.indicators['sma_long'].get()) / (2 * self.indicators['volatility'].get() + 1e-8),
            'mean_price': np.mean(self.price_buffer) if self.price_buffer else bid
        }

    def _validate_signal(self, X: Dict[str, float], y_pred: int) -> bool:
        """Filtrowanie słabych sygnałów"""
        conditions = [
            abs(X['momentum']) > self.min_pips_threshold,
            abs(X['bollinger']) < 2.5,
            (X['ema_ratio'] - 1) * (y_pred - 0.5) > 0,
            X['rsi'] < 70 if y_pred == 1 else X['rsi'] > 30
        ]
        return all(conditions)

    def is_prediction_about_to_change(self, X: Dict[str, float], current_pred: int) -> bool:
        """Detekcja potencjalnej zmiany sygnału"""
        proba = self.model.predict_proba_one(X)
        current_conf = proba.get(current_pred, 0.5)
        opposite_conf = proba.get(1 - current_pred, 0.5)
        return opposite_conf > (current_conf - self.warning_threshold)

    def process_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            current_time = datetime.strptime(tick_data["time"], "%Y-%m-%d %H:%M:%S")
            tick = {
                "time": tick_data["time"],
                "bid": float(tick_data["bid"]),
                "ask": float(tick_data["ask"])
            }

            # Aktualizacja wskaźników
            for indicator in self.indicators.values():
                if isinstance(indicator, EnhancedEMA):
                    indicator.update(tick["bid"])
                else:
                    indicator.update(tick["bid"])
            
            self.price_buffer.append(tick["bid"])

            # Inicjalizacja czasu dla pierwszego ticka
            if self.last_prediction_time is None:
                self.last_prediction_time = current_time
                self.prev_bid = tick["bid"]
                return None

            # Główna logika predykcji 5-minutowych
            elapsed = (current_time - self.last_prediction_time).total_seconds()
            X = self._generate_features(tick["bid"], tick["ask"])
            y_true = 1 if tick["bid"] > self.prev_bid else 0

            # Predykcja główna co 5 minut
            if elapsed >= self.prediction_horizon:
                y_pred = self.model.predict_one(X)
                confidence = max(self.model.predict_proba_one(X).values(), default=0.5)
                
                self.current_position = {
                    "prediction": y_pred,
                    "entry_time": current_time,
                    "entry_price": tick["bid"],
                    "confidence": confidence,
                    "buffer": collections.deque(maxlen=5)  # Bufor dla korekt
                }
                self.last_prediction_time = current_time
                
                # Logika uczenia tylko na pełnych okresach
                self.model.learn_one(X, y_true)
                for metric in self.metrics.values():
                    metric.update(y_true, y_pred)

            # Korekta co 2.5 minuty
            elif elapsed >= self.reassessment_interval and self.current_position:
                new_pred = self.model.predict_one(X)
                self.current_position["buffer"].append(new_pred)
                
                # Warunek korekty (3/5 predykcji przeciwne)
                if sum(1 for p in self.current_position["buffer"] 
                    if p != self.current_position["prediction"]) >= 3:
                    self.current_position.update({
                        "prediction": 1 - self.current_position["prediction"],
                        "confidence": max(self.model.predict_proba_one(X).values(), default=0.5),
                        "adjustment_time": current_time
                    })

            # Generowanie sygnału wyjściowego
            if self.current_position:
                pips = round((tick["bid"] - self.current_position["entry_price"]) * 10000, 1)
                warning = self.is_prediction_about_to_change(X, self.current_position["prediction"])
                
                log_entry = {
                    "time": tick["time"],
                    "bid": tick["bid"],
                    "prediction": self.current_position["prediction"],
                    "confidence": self.current_position["confidence"],
                    "pips": pips,
                    "warning": warning,
                    "metrics": {k: round(v.get(), 4) for k, v in self.metrics.items()},
                    "phase": "initial" if elapsed >= self.prediction_horizon else "reassessment"
                }
                
                self.logger.info(f"River-V8-DECISION: {json.dumps(log_entry)}")
                
                self.prev_prev_bid = self.prev_bid
                self.prev_bid = tick["bid"]
                
                return {
                    "action": "BUY" if self.current_position["prediction"] == 1 else "SELL",
                    **log_entry
                }

            self.prev_bid = tick["bid"]
            return None

        except Exception as e:
            self.logger.error(f"Error processing tick: {str(e)}")
            return None
        
    def save_model(self):
        # --- POPRAWKA: backup z timestampem ---
        with gzip.open(self.model_save_path, "wb") as f:
            pickle.dump(self.model, f)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"river_model_v8_{timestamp}.pkl.gz"
        with gzip.open(backup_path, "wb") as f:
            pickle.dump(self.model, f)
            
        # self._cleanup_old_backups(max_backups=5)
    
        
    def _load_or_init_model(self):
        if Path(self.model_save_path).exists():
            with gzip.open(self.model_save_path, "rb") as f:
                return pickle.load(f)
        return self._init_model()
        
        
    def _cleanup_old_backups(self, max_backups=5):
        model_dir = os.path.dirname(self.model_save_path)
        model_files = [
            f for f in os.listdir(model_dir) 
            if f.startswith("river_model_v10") and f.endswith(".pkl.gz")
        ]
        model_files.sort(reverse=True)
        
        for old_file in model_files[max_backups:]:
            os.remove(os.path.join(model_dir, old_file))


       
    def _handle_exit(self, signum, frame):
        self.logger.info("Shutting down gracefully...")
        self.save_model()
        sys.exit(0)

async def river_websocket_client():
    uri = "ws://127.0.0.1:8769"
    model = RiverModelTrainer()

    async with websockets.connect(uri) as websocket:
        print("River V10 connected to websocket")
        while True:
            message = await websocket.recv()
            tick_data = json.loads(message)
            decision = model.process_tick(tick_data)

            if decision:
                action = decision['action']
                confidence = decision['confidence']
                pips = decision['pips']
                warning = decision['warning']

                # System kolorów i emoji
                if action == "BUY":
                    if confidence > 0.7:
                        color = "🟢"
                    elif confidence > 0.6:
                        color = "🟡"
                    else:
                        color = "🟠"
                else:  # SELL
                    if confidence > 0.7:
                        color = "🔴"
                    elif confidence > 0.6:
                        color = "🟣"
                    else:
                        color = "🟤"

                warning_msg = " ⚠️" if warning else ""
                print(f"{color} {action} @ {decision['bid']} (Pips: {pips}, Conf: {confidence:.2f}{warning_msg})")

if __name__ == "__main__":
    asyncio.run(river_websocket_client())