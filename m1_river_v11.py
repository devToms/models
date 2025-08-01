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

from pathlib import Path
import sys
import os

# Dodanie ścieżki do systemu
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from helpers.csv_data_reader import CsvDataReader
from loggers.light_logger import LightLogger

training_log_file = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/training_logs/river_training_v11_log.txt"
training_logger = LightLogger(training_log_file, logger_name="training_logger")

# =============================================
# SUGESTIA OPTYMALIZACYJNA 1 (oparta na wynikach):
# Wprowadzić dynamiczne progi pewności zamiast stałych
DYNAMIC_CONFIDENCE_THRESHOLDS = {
    'high': 0.8,  # Dla przedziału (0.801, 0.901] gdzie skuteczność ~72%
    'medium': 0.7, # Dla przedziału (0.702, 0.801] gdzie skuteczność ~65%
    'low': 0.6     # Dla niższych wartości (mniej skuteczne)
}
# =============================================

class Std(stats.Var):
    def get(self):
        return math.sqrt(super().get())

stats.Std = Std

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
            std_dev = math.sqrt(self.volatility.get())
            vol_adjusted_alpha = self.alpha * (1 + 0.5 * math.tanh(std_dev * 10000))
            self.value = price * vol_adjusted_alpha + self.value * (1 - vol_adjusted_alpha)
        return self
    
    def get(self) -> float:
        return self.value if self.value is not None else 0

    # =============================================
    # SUGESTIA OPTYMALIZACYJNA 2:
    # Dodaj śledzenie skuteczności dla różnych okresów EMA
    TRACK_EMA_PERFORMANCE = True
    ema_performance = {'short': {'correct': 0, 'total': 0}, 
                      'long': {'correct': 0, 'total': 0}}
    # =============================================

class RiverModelTrainer:
    def __init__(self, model_save_path="/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/river_models/river_model_v11.pkl.gz"):
        self.logger = training_logger
        self.model_save_path = model_save_path
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
        
        # =============================================
        # SUGESTIA OPTYMALIZACYJNA 3:
        # Dodaj śledzenie skuteczności w zależności od pory dnia
        self.hourly_performance = {h: {'correct': 0, 'total': 0} for h in range(24)}
        # =============================================

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

        # =============================================
        # SUGESTIA OPTYMALIZACYJNA 4:
        # Dynamiczny próg pipsów w oparciu o zmienność
        self.dynamic_pips_threshold = True
        self.volatility_multiplier = 1.5
        # =============================================

    def _init_tracking(self):
        self.tick_count = 0
        self.save_interval = 100
        self.benchmark = {'correct': 0, 'total': 0}
        self.prev_bid = None
        self.prev_prev_bid = None

    @functools.lru_cache(maxsize=1000)
    def _calculate_rsi(self):
        if len(self.price_buffer) < 14:
            return 50
                
        gains = [max(0, self.price_buffer[i] - self.price_buffer[i-1]) 
                for i in range(1, len(self.price_buffer))]
        losses = [max(0, self.price_buffer[i-1] - self.price_buffer[i]) 
                for i in range(1, len(self.price_buffer))]
                    
        avg_gain = np.mean(gains[-14:]) if gains else 1e-8
        avg_loss = np.mean(losses[-14:]) if losses else 1e-8
        return 100 - (100 / (1 + (avg_gain / avg_loss)))

    def _generate_features(self, bid: float, ask: float) -> Dict[str, float]:
        spread = ask - bid
        momentum = bid - self.prev_bid if self.prev_bid else 0
        
        # Zabezpieczenie przed dzieleniem przez zero
        ema_long = self.indicators['ema_long'].get() or 1e-8
        sma_long = self.indicators['sma_long'].get() or 1e-8
        volatility = self.indicators['volatility'].get() or 1e-8
        
        features = {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'momentum': momentum,
            'momentum_acc': momentum - (self.prev_bid - self.prev_prev_bid) if self.prev_prev_bid else 0,
            'ema_short': self.indicators['ema_short'].get(),
            'ema_long': ema_long,
            'ema_ratio': self.indicators['ema_short'].get() / ema_long,
            'sma_short': self.indicators['sma_short'].get(),
            'sma_long': sma_long,
            'sma_diff': self.indicators['sma_short'].get() - sma_long,
            'volatility': volatility,
            'volatility_ratio': volatility / (bid + 1e-8),
            'rsi': self._calculate_rsi(),
            'bollinger': (bid - sma_long) / (2 * volatility + 1e-8),
            'mean_price': np.mean(self.price_buffer) if self.price_buffer else bid,
            'confidence_weighted_ema': self.indicators['ema_short'].get() * (self.metrics['accuracy'].get() or 0.5),
            'trend_strength': abs(self.indicators['ema_short'].get() - ema_long) / volatility
        }
        return features

    def _validate_signal(self, X: Dict[str, float], y_pred: int) -> bool:
        """Filtrowanie słabych sygnałów"""
        # Zabezpieczenie przed dzieleniem przez zero
        volatility = X['volatility'] or 1e-8
        
        # Dynamiczny próg pipsów
        min_pips = self.min_pips_threshold
        if self.dynamic_pips_threshold:
            min_pips = max(min_pips, volatility * self.volatility_multiplier * 0.0001)
        
        conditions = [
            abs(X['momentum']) > min_pips,
            abs(X['bollinger']) < 2.5,
            (X['ema_ratio'] - 1) * (y_pred - 0.5) > 0,
            X['rsi'] < 70 if y_pred == 1 else X['rsi'] > 30,
            volatility < (np.mean(list(self.price_buffer)) * 0.01) if len(self.price_buffer) > 10 else True
        ]
        return all(conditions)

    def is_prediction_about_to_change(self, X: Dict[str, float], current_pred: int) -> bool:
        proba = self.model.predict_proba_one(X)
        current_conf = proba.get(current_pred, 0.5)
        opposite_conf = proba.get(1 - current_pred, 0.5)
        
        # =============================================
        # SUGESTIA OPTYMALIZACYJNA 7:
        # Dynamiczny próg ostrzegawczy w oparciu o zmienność
        volatility_factor = 1 + (X['volatility'] * 10000)
        return opposite_conf > (current_conf - (self.warning_threshold / volatility_factor))
        # =============================================

    def process_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            tick = {
                "time": tick_data["time"],
                "bid": float(tick_data["bid"]),
                "ask": float(tick_data["ask"])
            }

            # Aktualizacja wskaźników
            for name, indicator in self.indicators.items():
                if isinstance(indicator, EnhancedEMA):
                    indicator.update(tick["bid"])
                    # =============================================
                    # SUGESTIA OPTYMALIZACYJNA 8:
                    # Śledź skuteczność EMA jeśli włączone
                    if EnhancedEMA.TRACK_EMA_PERFORMANCE and self.prev_bid is not None:
                        direction_correct = (indicator.get() > self.prev_bid) == (tick["bid"] > self.prev_bid)
                        key = 'short' if 'short' in name else 'long'
                        EnhancedEMA.ema_performance[key]['total'] += 1
                        if direction_correct:
                            EnhancedEMA.ema_performance[key]['correct'] += 1
                    # =============================================
                else:
                    indicator.update(tick["bid"])
            
            self.price_buffer.append(tick["bid"])

            if self.prev_bid is None:
                self.prev_bid = tick["bid"]
                return None

            X = self._generate_features(tick["bid"], tick["ask"])
            y_true = 1 if tick["bid"] > self.prev_bid else 0

            if abs(X['momentum']) < (self.min_pips_threshold if not self.dynamic_pips_threshold 
                                   else max(self.min_pips_threshold, X['volatility'] * self.volatility_multiplier * 0.0001)):
                return None

            y_pred = self.model.predict_one(X)
            
            if not self._validate_signal(X, y_pred):
                return None

            self.model.learn_one(X, y_true)
            
            for metric in self.metrics.values():
                metric.update(y_true, y_pred)

            error = 0 if y_true == y_pred else 1
            self.drift_detector.update(error)
            if self.drift_detector.drift_detected:
                self.logger.warning("Drift detected! Resetting model...")
                self.model = self._init_model()
                self.drift_detector = drift.ADWIN(delta=0.002)

            pips = round((tick["bid"] - self.prev_bid) * 10000, 1)
            confidence = max(self.model.predict_proba_one(X).values(), default=0.5)
            warning = self.is_prediction_about_to_change(X, y_pred)
            
            # =============================================
            # SUGESTIA OPTYMALIZACYJNA 9:
            # Śledź skuteczność godzinową
            current_hour = datetime.now().hour
            self.hourly_performance[current_hour]['total'] += 1
            if y_true == y_pred:
                self.hourly_performance[current_hour]['correct'] += 1
                
            # Loguj dodatkowe metryki
            additional_metrics = {
                'hourly_accuracy': round(self.hourly_performance[current_hour]['correct'] / 
                                       (self.hourly_performance[current_hour]['total'] or 1), 4),
                'ema_short_accuracy': round(EnhancedEMA.ema_performance['short']['correct'] / 
                                          (EnhancedEMA.ema_performance['short']['total'] or 1), 4),
                'ema_long_accuracy': round(EnhancedEMA.ema_performance['long']['correct'] / 
                                         (EnhancedEMA.ema_performance['long']['total'] or 1), 4),
                'confidence_level': self._get_confidence_level(confidence)
            }
            # =============================================

            log_entry = {
                "time": tick["time"],
                "bid": tick["bid"],
                "prediction": y_pred,
                "confidence": confidence,
                "pips": pips,
                "warning": warning,
                "metrics": {k: round(v.get(), 4) for k, v in self.metrics.items()},
                "additional_metrics": additional_metrics
            }
            self.logger.info(f"River-V11-DECISION: {json.dumps(log_entry)}")

            self.tick_count += 1
            if self.tick_count % self.save_interval == 0:
                self.save_model()

            self.prev_prev_bid = self.prev_bid
            self.prev_bid = tick["bid"]

            return {
                "action": "BUY" if y_pred == 1 else "SELL",
                **log_entry,
                # =============================================
                # SUGESTIA OPTYMALIZACYJNA 10:
                # Dodaj sugerowaną wielkość pozycji na podstawie pewności
                "position_size": self._calculate_position_size(confidence, X['volatility'])
                # =============================================
            }

        except Exception as e:
            self.logger.error(f"Error processing tick: {str(e)}")
            return None
            
    # =============================================
    # NOWE METODY DODANE JAKO SUGESTIE OPTYMALIZACYJNE
    def _get_confidence_level(self, confidence: float) -> str:
        """Klasyfikuj poziom pewności na podstawie statystyk skuteczności"""
        if confidence >= DYNAMIC_CONFIDENCE_THRESHOLDS['high']:
            return 'high'
        elif confidence >= DYNAMIC_CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        return 'low'
        
    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """Oblicz sugerowaną wielkość pozycji na podstawie pewności i zmienności"""
        base_size = 1.0  # Podstawowa wielkość pozycji
        confidence_factor = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }.get(self._get_confidence_level(confidence), 0.5)
        
        volatility_adjustment = 1 / (1 + volatility * 10000)  # Mniejsze pozycje przy wysokiej zmienności
        return round(base_size * confidence_factor * volatility_adjustment, 2)
    # =============================================

    def save_model(self):
        with gzip.open(self.model_save_path, "wb") as f:
            pickle.dump(self.model, f)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"river_model_v6_{timestamp}.pkl.gz"
        with gzip.open(backup_path, "wb") as f:
            pickle.dump(self.model, f)
            
    def _load_or_init_model(self):
        if Path(self.model_save_path).exists():
            with gzip.open(self.model_save_path, "rb") as f:
                return pickle.load(f)
        return self._init_model()
        
    def _handle_exit(self, signum, frame):
        self.logger.info("Shutting down gracefully...")
        self.save_model()
        sys.exit(0)


from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK



async def river_websocket_client():
    uri = "ws://127.0.0.1:8767"
    model = RiverModelTrainer()

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
                print("✅ River V11 connected to websocket")

                while True:
                    try:
                        message = await websocket.recv()
                        tick_data = json.loads(message)
                        decision = model.process_tick(tick_data)

                        if decision:
                            action = decision['action']
                            confidence = decision['confidence']
                            pips = decision['pips']
                            warning = decision['warning']
                            position_size = decision.get('position_size', 1.0)

                            confidence_level = model._get_confidence_level(confidence)
                            confidence_symbol = {
                                'high': '★',
                                'medium': '✦',
                                'low': '✧'
                            }.get(confidence_level, '?')

                            color_code = {
                                ('BUY', 'high'): '🟢',
                                ('BUY', 'medium'): '🟡',
                                ('BUY', 'low'): '🟠',
                                ('SELL', 'high'): '🔴',
                                ('SELL', 'medium'): '🟣',
                                ('SELL', 'low'): '🟤'
                            }.get((action, confidence_level), '⚪')

                            warning_msg = " ⚠️" if warning else ""
                            size_indicator = f"×{position_size}" if position_size != 1.0 else ""

                            print(f"{color_code}{confidence_symbol} {action}{size_indicator} @ {decision['bid']} "
                                  f"(Pips: {pips}, Conf: {confidence:.2f}{warning_msg})")

                    except (ConnectionClosedError, ConnectionClosedOK) as e:
                        print(f"❌ Połączenie zerwane przez serwer: {e}")
                        break
                    except Exception as e:
                        print(f"⚠️ Błąd przy odbiorze wiadomości: {e}")
        except Exception as e:
            print(f"❗ Nie można połączyć się z websocketem: {e}")

        print("🔄 Ponawianie połączenia za 5 sekund...")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(river_websocket_client())
