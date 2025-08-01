import functools
import collections
from river import compose, linear_model, preprocessing, metrics, drift, optim, stats
from pathlib import Path
import sys
import os
import pickle
import gzip
from datetime import datetime
from datetime import datetime as dt
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
import requests
# Dodanie ścieżki do systemu
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from helpers.csv_data_reader import CsvDataReader
from loggers.light_logger import LightLogger
import logging

import json

logger = logging.getLogger(__name__)

training_log_file = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/training_logs/river_training_v13_log.txt"
training_logger = LightLogger(training_log_file, logger_name="training_logger")

training_log_file = "ragent_v1_log.txt"
agent_logger = LightLogger(training_log_file, logger_name="agent_logger")


class TradingSignal:
    """Reprezentacja sygnału tradingowego z pełnymi danymi"""
    def __init__(self, action: str, confidence: float, bid: float, 
                 pips: float, warning: bool, volatility: float = 0.0):
        self.action = action
        self.confidence = confidence
        self.bid = bid
        self.pips = pips
        self.warning = warning
        self.volatility = volatility
        self.position_size = self._calculate_size(confidence, volatility)
        self.timestamp = datetime.now()
        
    def _calculate_size(self, confidence, volatility):
        base_size = 1.0
        vol_adjustment = 1 / (1 + (volatility * 10000))
        conf_level = self._get_confidence_level(confidence)
        conf_adjustment = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }.get(conf_level, 0.5)
        return round(base_size * conf_adjustment * vol_adjustment, 2)
        
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence > 0.8:
            return 'high'
        elif confidence > 0.7:
            return 'medium'
        return 'low'
        
    def __str__(self):
        warning_str = " ⚠️" if self.warning else ""
        return (f"{self.action}@{self.bid} (Conf: {self.confidence:.2f}, "
                f"Size: {self.position_size:.2f}, Pips: {self.pips}{warning_str})")




class ApiOrderClient:
    def __init__(self, websocket_url: str = "ws://127.0.0.1:5001"):
        self.websocket_url = websocket_url
        self.websocket = None
        self.connection_lock = asyncio.Lock()  # Dodajemy lock dla bezpieczeństwa wielowątkowego

    async def _send_request(self, action: str):
        """Proste wysłanie żądania bez zbędnych sprawdzeń"""
        try:
            if self.websocket is None:
                self.websocket = await websockets.connect(self.websocket_url)
            
            await self.websocket.send(json.dumps({'action': action}))
            response = await self.websocket.recv()
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Błąd: {str(e)}")
            self.websocket = None
            return {'status': 'error', 'message': str(e)}

    async def close(self):
        """Proste zamknięcie połączenia"""
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

    # Metody handlowe pozostają bez zmian
    async def buy(self): return await self._send_request('buy')
    async def sell(self): return await self._send_request('sell')
    async def close_order(self): return await self._send_request('close_order')
    
    async def get_account_info(self): 
        try:
            resp = await self._send_request('get_account_info')
            if not isinstance(resp, dict):
                logger.error(f"Invalid response format: {resp}")
                return {'status': 'error', 'message': 'Invalid response format', 'result': {}}
            return resp
        except Exception as e:
            logger.error(f"Error in get_account_info: {str(e)}")
            return {'status': 'error', 'message': str(e), 'result': {}}
    
    async def get_open_positions(self): 
        try:
            resp = await self._send_request('get_open_positions')
            if not isinstance(resp, dict):
                logger.error(f"Invalid response format: {resp}")
                return {'status': 'error', 'message': 'Invalid response format', 'result': []}
            return resp
        except Exception as e:
            logger.error(f"Error in get_open_positions: {str(e)}")
            return {'status': 'error', 'message': str(e), 'result': []}




class OrderExecutor:
    """Wykonawca zleceń z rozszerzonym zarządzaniem ryzykiem"""
    
    def __init__(self, api_client, trade_direction='sell'):
        self.api = api_client
        self.account = AccountManager(api_client, agent_logger)
        self.logger = agent_logger
        self.last_action = None
        self.min_pips_diff = 1.5
        self.profit_target = 5.0  # Zwiększony take profit dla V7
        self.max_drawdown = 1.0   # Nieco większy stop loss
        self.min_confidence = 0.7  # Zwiększony próg dla V7
        self.trade_direction = trade_direction.lower()
        self.force_close_threshold = -1.0
        
    async def initialize(self):
        """Inicjalizacja, która musi być wywołana po utworzeniu obiektu"""
        await self.account.initialize()

    async def execute(self, signal: TradingSignal) -> bool:
        """Wykonuje zlecenie z rozszerzonymi zasadami ryzyka"""
        try:
            # Filtrowanie kierunku
            if self.trade_direction == 'buy' and signal.action != 'BUY':
                self.logger.debug(f"IGNORED {signal.action} (trading only BUY)")
                return False
            elif self.trade_direction == 'sell' and signal.action != 'SELL':
                self.logger.debug(f"IGNORED {signal.action} (trading only SELL)")
                return False
                
            if signal.confidence < self.min_confidence:
                self.logger.debug(f"Pomijam sygnał {signal} - zbyt niska pewność")
                return False
                
            if not await self.account.refresh_data():
                self.logger.error("Nie udało się odświeżyć danych konta")
                return False

            current_position = await self.account.get_open_position()
            
            if current_position:
                return await self._handle_existing_position(signal, current_position)
            return await self._open_new_position(signal)

        except Exception as e:
            self.logger.error(f"Błąd wykonania zlecenia: {str(e)}")
            return False
        
    async def _handle_existing_position(self, signal: TradingSignal, position: dict) -> bool:
        if signal.action != position['type']:
            self.logger.info(f"🔀 Sygnał odwrotny ({signal.action}), zamykam pozycję {position['type']}")
            return await self._close_position()

        close_reason = await self._should_close_position(signal, position)
        if close_reason:
            self.logger.info(f"🛑 Warunek zamknięcia: {close_reason}")
            return await self._close_position()

        self.logger.debug(f"🔄 Pomijam sygnał {signal.action} - już mam otwartą pozycję")
        return False

    async def _should_close_position(self, signal: TradingSignal, position: dict) -> Optional[str]:
        try:
            position_size = float(position['size'])
            current_profit = float(position['profit'])
            symbol = position.get('symbol', 'EURUSD')
            
            # Prawidłowe obliczenie wartości pipsa dla danego symbolu i rozmiaru
            pip_value = self._calculate_pip_value(symbol, position_size)
            profit_pips = current_profit / pip_value if pip_value != 0 else 0
            
            # Dynamiczne progi na podstawie zmienności
            dynamic_sl = max(
                self.max_drawdown, 
                2 * signal.volatility * 10000  # Volatility w pipsach
            )
            dynamic_force_close = min(
                self.force_close_threshold,
                -dynamic_sl * 0.5
            )
            
            print(f"[DEBUG] Profit: {current_profit:.2f} USD → {profit_pips:.1f} pips | "
                f"Dynamic SL: {-dynamic_sl:.1f}pips, Force: {dynamic_force_close:.1f}pips")
            
            # Warunki zamknięcia w kolejności priorytetów
            if profit_pips <= dynamic_force_close:
                return f"FORCE CLOSE at {profit_pips:.1f}pips (threshold: {dynamic_force_close:.1f})"
                
            if profit_pips <= -dynamic_sl:
                return f"Stop loss at {profit_pips:.1f}pips (SL: {-dynamic_sl:.1f})"
                
            if profit_pips >= self.profit_target:
                return f"Take profit at {profit_pips:.1f}pips"
                
            if signal.warning and signal.confidence < 0.7:
                return f"Low confidence signal ({signal.confidence:.2f})"
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking close conditions: {str(e)}")
            return None

    def _calculate_pip_value(self, symbol: str, lot_size: float) -> float:
        """Oblicza wartość 1 pipsa dla danej pary walutowej i rozmiaru pozycji"""
        if 'JPY' in symbol:
            return lot_size * 0.1  # Dla par z JPY
        return lot_size * 0.0001  # Dla większości par
    
    async def _open_new_position(self, signal: TradingSignal) -> bool:
        try:
            if signal.action == 'BUY':
                response = await self.api.buy()
            else:
                response = await self.api.sell()
            
            if isinstance(response, dict):
                result = response
            else:
                result = response.json() if hasattr(response, 'json') else {}

            if result.get('status') == 'success':
                self._log_execution(signal)
                self.last_action = signal.action
                return True

            self.logger.error(f"Nie udało się wykonać zlecenia: {result.get('error', 'unknown')}")
            return False

        except Exception as e:
            self.logger.error(f"Błąd wykonania zlecenia: {str(e)}")
            return False

    async def _close_position(self) -> bool:
        """Zamyka aktualną pozycję"""
        try:
            result = await self.api.close_order()
            if isinstance(result, dict):
                if result.get('status') == 'success':
                    self.logger.info("✅ Zamknięto pozycję")
                    self.last_action = None
                    return True
            elif hasattr(result, 'status_code') and result.status_code == 200:
                self.logger.info("✅ Zamknięto pozycję (brak odpowiedzi JSON)")
                self.last_action = None
                return True
                
            self.logger.error(f"Błąd przy zamykaniu: {str(result)}")
            return False
        except Exception as e:
            self.logger.error(f"Błąd przy zamykaniu pozycji: {str(e)}")
            return False

    def _log_execution(self, signal: TradingSignal):
        """Loguje szczegóły wykonania zlecenia"""
        conf_level = self._get_confidence_level(signal.confidence)
        color = '🟢' if signal.action == 'BUY' else '🔴'
        symbol = {
            'high': '★★★',
            'medium': '★★',
            'low': '★'
        }.get(conf_level, '?')
        
        log_msg = (f"{color}{symbol} {signal.action} @ {signal.bid} | "
                  f"Conf: {signal.confidence:.2f} ({conf_level}) | "
                  f"Size: {signal.position_size:.2f}")
                  
        if signal.warning:
            log_msg += " ⚠️"
            
        self.logger.info(log_msg)

    def _get_confidence_level(self, confidence: float) -> str:
        """Klasyfikuje poziom pewności"""
        if confidence > 0.85:  # Wyższy próg dla V7
            return 'high'
        elif confidence > 0.75:
            return 'medium'
        return 'low'

class AccountManager:
    """Enhanced account manager with risk management and balance tracking"""
    def __init__(self, api_order_client, agent_logger):
        if api_order_client is None:
            raise ValueError("api_order_client cannot be None")
            
        self.api_client = api_order_client
        self.logger = agent_logger 
        self.account_info = {}
        self.open_positions = []
        self.trade_history = []
        self.balance_history = []
        self.profit = None
        
    async def initialize(self):
        """Inicjalizacja, która musi być wywołana po utworzeniu obiektu"""
        try:
            await self.refresh_data()
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    async def refresh_data(self) -> bool:
        """Refreshes account data from API and updates history"""
        try:
            account_response = await self.api_client.get_account_info()
            # Dodaj zabezpieczenie przed brakiem odpowiedzi lub błędną strukturą
            if not isinstance(account_response, dict) or 'result' not in account_response:
                self.logger.error(f"Invalid account response: {account_response}")
                return False
                
            self.account_info = account_response.get('result', {})
            
            positions_response = await self.api_client.get_open_positions()
            # Dodaj podobne zabezpieczenie dla pozycji
            if not isinstance(positions_response, dict):
                self.logger.error(f"Invalid positions response: {positions_response}")
                self.open_positions = []
            else:
                self.open_positions = positions_response.get('result', [])
                
            current_balance = self.get_account().get('balance', 0)
            self.balance_history.append({
                'timestamp': dt.now(),
                'balance': current_balance,
                'equity': self.get_account().get('equity', 0)
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error refreshing data: {str(e)}")
            return False
        
    async def get_position_profit(self) -> float:
        try:
            account_response = await self.api_client.get_account_info()
            if not account_response or 'result' not in account_response:
                return 0.0
                
            profit = account_response['result'].get('profit', 0.0)
            return float(profit)
        except Exception as e:
            return 0.0

    def get_account(self) -> dict:
        if not self.account_info:
            return {}
        
        return {
            'login': self.account_info.get('login', ''),
            'name': self.account_info.get('name', ''),
            'server': self.account_info.get('server', ''),
            'company': self.account_info.get('company', ''),
            'balance': float(self.account_info.get('balance', 0)),
            'equity': float(self.account_info.get('equity', 0)),
            'margin': float(self.account_info.get('margin', 0)),
            'margin_free': float(self.account_info.get('margin_free', 0)),
            'margin_level': float(self.account_info.get('margin_level', 0)),
            'currency': self.account_info.get('currency', ''),
            'leverage': int(self.account_info.get('leverage', 1)),
            'profit': float(self.account_info.get('profit', 0)),
            'trade_allowed': bool(self.account_info.get('trade_allowed', False)),
            'trade_expert': bool(self.account_info.get('trade_expert', False)),
            'credit': float(self.account_info.get('credit', 0))
        }

    async def get_open_position(self) -> dict:
        await self.refresh_data()
        
        if not self.open_positions or not isinstance(self.open_positions, list):
            return {}

        position = self.open_positions[0]
        
        if isinstance(position, dict):
            return {
                'ticket': position.get('ticket'),
                'type': 'BUY' if position.get('type') == 0 else 'SELL',
                'entry_time': position.get('time'),
                'entry_price': position.get('price_open'),
                'size': position.get('volume'),
                'stop_loss': position.get('sl'),
                'take_profit': position.get('tp'),
                'current_price': position.get('price_current'),
                'swap': position.get('swap'),
                'profit': position.get('profit', 0),
                'symbol': position.get('symbol'),
            }
        elif hasattr(position, '__getitem__'):
            try:
                return {
                    'ticket': position[0],
                    'type': 'BUY' if position[5] == 1 else 'SELL',
                    'entry_time': position[1],
                    'entry_price': position[10],
                    'size': position[9],
                    'stop_loss': position[12],
                    'take_profit': position[11],
                    'current_price': position[13],
                    'swap': position[14],
                    'profit': position[15],
                    'symbol': position[16],
                }
            except Exception as e:
                self.logger.error(f"Position parsing error: {str(e)}")
                return {}
        else:
            self.logger.error(f"Unknown position format: {type(position)}")
            return {}

    async def has_open_position(self) -> bool:
        await self.refresh_data()
        return len(self.open_positions) > 0

    async def get_profit(self) -> float:
        if await self.has_open_position():
            position = await self.get_open_position()
            return float(position.get('profit', 0))
        return float(self.get_account().get('profit', 0))

    async def get_account_metrics(self) -> dict:
        account = self.get_account()
        balance = account.get('balance', 0)
        equity = account.get('equity', 0)
        margin_free = account.get('margin_free', 0)
        
        drawdown = 0
        if self.balance_history:
            max_balance = max(h['balance'] for h in self.balance_history)
            drawdown = (max_balance - balance) / max_balance if max_balance > 0 else 0
            
        return {
            'balance': balance,
            'equity': equity,
            'margin_free': margin_free,
            'leverage': account.get('leverage', 1),
            'currency': account.get('currency', 'EUR'),
            'drawdown': drawdown,
            'risk_per_trade': min(0.02, 0.01 * equity / balance) if balance > 0 else 0.01,
            'daily_profit': await self._calculate_daily_performance(),
            'has_open_position': await self.has_open_position(),
            'current_profit': await self.get_profit()
        }
        
    async def _calculate_daily_performance(self) -> dict:
        today = dt.now().date()
        today_trades = [t for t in self.trade_history 
                       if t.get('exit_time', dt.datetime.now()).date() == today]
        
        profit = sum(t.get('profit', 0) for t in today_trades)
        win_rate = (sum(1 for t in today_trades if t.get('profit', 0) > 0) / len(today_trades)) if today_trades else 0
        
        return {
            'trades': len(today_trades),
            'profit': profit,
            'win_rate': win_rate,
            'avg_profit': profit / len(today_trades) if today_trades else 0
        }
        
        
class TrendAnalyzer:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.price_history = collections.deque(maxlen=window_size)
        self.trend = None
    
    def update(self, current_price):
        self.price_history.append(current_price)
        
        if len(self.price_history) >= self.window_size:
            avg = sum(self.price_history) / len(self.price_history)
            self.trend = 'BUY' if current_price > avg else 'SELL'
        
        return self.trend
        
    def should_favor_downtrend(self):
        if len(self.price_history) < self.window_size:
            return False
        downtrend_ratio = sum(1 for p in self.price_history if p < np.mean(self.price_history)) / len(self.price_history)
        return downtrend_ratio > 0.6

class Std(stats.Var):
    def get(self):
        return math.sqrt(super().get())

stats.Std = Std

DYNAMIC_CONFIDENCE_THRESHOLDS = {
    'high': 0.85,
    'medium': 0.75,
    'low': 0.65
}

class EnhancedEMA:
    """Rozszerzona implementacja EMA z adaptacyjnym współczynnikiem i śledzeniem skuteczności"""
    # Class-level performance tracking
    ema_performance = {'short': {'correct': 0, 'total': 0}, 
                      'long': {'correct': 0, 'total': 0}}
    
    TRACK_EMA_PERFORMANCE = True  # Class-level flag to control tracking

    def __init__(self, period: int = 5):
        self.period = period
        self.value = None
        self.alpha = 2 / (period + 1)
        self.volatility = stats.Var()
        self.performance = {'correct': 0, 'total': 0}  # Instance-level performance tracking

    def update(self, price: float, actual_direction: int = None) -> 'EnhancedEMA':
        if self.value is None:
            self.value = price
        else:
            # Dynamiczne alpha w zależności od zmienności
            self.volatility.update(price)
            std_dev = math.sqrt(self.volatility.get())
            vol_adjusted_alpha = self.alpha * (1 + 0.5 * math.tanh(std_dev * 10000))
            self.value = price * vol_adjusted_alpha + self.value * (1 - vol_adjusted_alpha)
            
            # Aktualizacja statystyk skuteczności jeśli podano rzeczywisty kierunek
            if actual_direction is not None:
                predicted_direction = 1 if self.value > price else 0
                self.performance['total'] += 1
                if predicted_direction == actual_direction:
                    self.performance['correct'] += 1
                
                # Update class-level performance if tracking enabled
                if self.TRACK_EMA_PERFORMANCE:
                    key = 'short' if self.period <= 10 else 'long'
                    EnhancedEMA.ema_performance[key]['total'] += 1
                    if predicted_direction == actual_direction:
                        EnhancedEMA.ema_performance[key]['correct'] += 1
                    
        return self
    
    def get_accuracy(self) -> float:
        return self.performance['correct'] / self.performance['total'] if self.performance['total'] > 0 else 0.5
    
    def get(self) -> float:
        return self.value if self.value is not None else 0

class RiverModelTrainer:
    def __init__(self, model_save_path="/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/snapshots/river_model_v7_2.pkl.gz"):
        self.logger = training_logger
        self.model_save_path = model_save_path
        self.model = self._load_or_init_model()
        self.metrics = {
            'accuracy': metrics.Accuracy(),
            'precision': metrics.Precision(),
            'recall': metrics.Recall()
        }
        self.drift_detector = drift.ADWIN(delta=0.002)
        self.trend_analyzer = TrendAnalyzer()
        self._setup_indicators()
        self._init_tracking()
        
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        
        self.hourly_performance = {h: {'correct': 0, 'total': 0} for h in range(24)}

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
        self.dynamic_pips_threshold = True
        self.volatility_multiplier = 1.5

    def _init_tracking(self):
        self.tick_count = 0
        self.save_interval = 100
        self.benchmark = {'correct': 0, 'total': 0}
        self.prev_bid = None
        self.prev_prev_bid = None

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
        volatility = X['volatility'] or 1e-8
        min_pips = self.min_pips_threshold
        if self.dynamic_pips_threshold:
            min_pips = max(min_pips, volatility * self.volatility_multiplier * 0.0001)
        
        conditions = [
            abs(X['momentum']) > (min_pips * 0.5),  # Używamy obliczonego min_pips zamiast stałej
            abs(X['bollinger']) < 2.5,
            (X['ema_ratio'] - 1) * (y_pred - 0.5) > -0.1,
            X['rsi'] < 80 if y_pred == 1 else X['rsi'] > 20,
            volatility < (X['bid'] * 0.01),  # Zwiększony limit zmienności
            # Możesz dodać dodatkowe warunki walidacji tutaj
        ]
        return all(conditions)

    def is_prediction_about_to_change(self, X: Dict[str, float], current_pred: int) -> bool:
        proba = self.model.predict_proba_one(X)
        current_conf = proba.get(current_pred, 0.5)
        opposite_conf = proba.get(1 - current_pred, 0.5)
        volatility_factor = 1 + (X['volatility'] * 10000)
        return opposite_conf > (current_conf - (self.warning_threshold / volatility_factor))

    def process_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            tick = {
                "time": tick_data["time"],
                "bid": float(tick_data["bid"]),
                "ask": float(tick_data["ask"])
            }

            # Aktualizacja trendu
            self.trend_analyzer.update(tick["bid"])

            for name, indicator in self.indicators.items():
                if isinstance(indicator, EnhancedEMA):
                    indicator.update(tick["bid"])
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

            # Wzmocnienie sygnałów zgodnych z trendem
            if (y_pred == 1 and self.trend_analyzer.trend == 'BUY') or \
               (y_pred == 0 and self.trend_analyzer.trend == 'SELL'):
                y_pred_proba = self.model.predict_proba_one(X)
                if y_pred in y_pred_proba:
                    y_pred_proba[y_pred] *= 1.2

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
            
            current_hour = datetime.now().hour
            self.hourly_performance[current_hour]['total'] += 1
            if y_true == y_pred:
                self.hourly_performance[current_hour]['correct'] += 1
                
            additional_metrics = {
                'hourly_accuracy': round(self.hourly_performance[current_hour]['correct'] / 
                                       (self.hourly_performance[current_hour]['total'] or 1), 4),
                'ema_short_accuracy': round(EnhancedEMA.ema_performance['short']['correct'] / 
                                          (EnhancedEMA.ema_performance['short']['total'] or 1), 4),
                'ema_long_accuracy': round(EnhancedEMA.ema_performance['long']['correct'] / 
                                         (EnhancedEMA.ema_performance['long']['total'] or 1), 4),
                'confidence_level': self._get_confidence_level(confidence),
                'trend': self.trend_analyzer.trend
            }

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
            self.logger.info(f"River-V7-DECISION: {json.dumps(log_entry)}")

            self.tick_count += 1
            if self.tick_count % self.save_interval == 0:
                self.save_model()

            self.prev_prev_bid = self.prev_bid
            self.prev_bid = tick["bid"]

            return {
                "action": "BUY" if y_pred == 1 else "SELL",
                **log_entry,
                "position_size": self._calculate_position_size(confidence, X['volatility']),
                "volatility": X['volatility']
            }

        except Exception as e:
            self.logger.error(f"Error processing tick: {str(e)}")
            return None
            
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= DYNAMIC_CONFIDENCE_THRESHOLDS['high']:
            return 'high'
        elif confidence >= DYNAMIC_CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        return 'low'
        
    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        base_size = 1.5  # Zwiększ bazowy rozmiar
        confidence_factor = {
            'high': 2.0,   # Zwiększ współczynniki
            'medium': 1.5,
            'low': 1.0
        }.get(self._get_confidence_level(confidence), 1.0)  # Dodane wywołanie _get_confidence_level
        
        volatility_adjustment = 1 / (0.5 + volatility * 10000)  # Mniejsza redukcja przez zmienność
        return round(base_size * confidence_factor * volatility_adjustment, 2)
    
    def save_model(self):
        with gzip.open(self.model_save_path, "wb") as f:
            pickle.dump(self.model, f)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"river_model_v7_{timestamp}.pkl.gz"
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
    uri = "ws://127.0.0.1:8666"
    model = RiverModelTrainer()
    api_client = ApiOrderClient()
    
    # Ustawienie strategii tylko SELL (można zmienić na 'buy' lub 'both')
    executor = OrderExecutor(api_client, trade_direction='sell')
    await executor.initialize() 

    total_signals = 0
    executed_signals = 0

    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
                print("🚀 River V7 connected to websocket")
                print("💡 Strategy: SELL only")
                print("💡 Ctrl+C to stop")

                while True:
                    try:
                        message = await websocket.recv()
                        tick_data = json.loads(message)
                        decision = model.process_tick(tick_data)
                        total_signals += 1

                        if decision:
                            signal = TradingSignal(
                                action=decision['action'],
                                confidence=decision['confidence'],
                                bid=decision['bid'],
                                pips=decision['pips'],
                                warning=decision['warning'],
                                volatility=decision['volatility']
                            )

                            conf_level = model._get_confidence_level(signal.confidence)
                            colors = {
                                'BUY': {'high': '🟢', 'medium': '🟡', 'low': '🟠'},
                                'SELL': {'high': '🔴', 'medium': '🟣', 'low': '🟤'}
                            }
                            symbols = {'high': '★★★', 'medium': '★★', 'low': '★'}

                            color = colors[signal.action].get(conf_level, '⚪')
                            symbol = symbols.get(conf_level, '?')
                            warning = " ⚠️" if signal.warning else ""
                            size = f"×{signal.position_size:.2f}" if signal.position_size != 1.0 else ""

                            print(f"\n{color}{symbol} {signal.action}{size} @ {signal.bid:.5f}")
                            print(f"   Pips: {signal.pips:.1f} | Conf: {signal.confidence:.2f}{warning}")
                            print(f"   Volatility: {signal.volatility*10000:.2f}pips | Trend: {decision['additional_metrics']['trend']}")

                            # if await executor.execute(signal):
                            #     executed_signals += 1
                            #     print(f"   ✅ Executed (Total: {executed_signals}/{total_signals})")
                            # else:
                            #     print(f"   ❌ Execution failed")

                    except (ConnectionClosedError, ConnectionClosedOK) as e:
                        print(f"❌ Połączenie zerwane: {e}")
                        break
                    except Exception as e:
                        print(f"⚠️ Błąd podczas odbioru wiadomości: {str(e)}")
                        continue

        except Exception as e:
            print(f"❗ Błąd podczas łączenia się z websocketem: {e}")

        print("🔄 Próba ponownego połączenia za 5 sekund...")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(river_websocket_client())
