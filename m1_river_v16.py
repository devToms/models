import functools
import collections
from river import compose, linear_model, preprocessing, metrics, drift, optim, stats
from pathlib import Path
import sys
import os
import pickle
import gzip
from datetime import datetime, timedelta
from datetime import datetime as dt
import time
import signal
import json
import asyncio
import websockets
import numpy as np
import math
from typing import Optional, Dict, Any, Deque
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from pathlib import Path
import sys
import os
import requests
# Dodanie ścieżki do systemu
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from helpers.csv_data_reader import CsvDataReader
from loggers.light_logger import LightLogger

from typing import List
import pandas as pd


training_log_file = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/training_logs/river_training_v16_log.txt"
training_logger = LightLogger(training_log_file, logger_name="training_logger")


training_log_file = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/training_logs/river_training_v16_api_log.txt"
training_api_logger = LightLogger(training_log_file, logger_name="training_api_logger")


training_log_file = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/training_logs/river_training_v16_order_log.txt"
training_order_logger = LightLogger(training_log_file, logger_name="training_order_logger")


training_log_file = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/training_logs/river_training_v16_account_log.txt"
training_account_logger = LightLogger(training_log_file, logger_name="training_account_logger")


training_log_file = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/training_logs/river_training_v16_websocket_log.txt"
training_websocket_logger = LightLogger(training_log_file, logger_name="training_websocket_logger")

logger = training_websocket_logger 


class TradingSignal:
    """Reprezentacja sygnału tradingowego z pełnymi danymi"""
    def __init__(self, action: str, confidence: float, bid: float, 
                 pips: float, warning: bool, volatility: float = 0.0,
                 trend: str = None, ema_ratio: float = 1.0):
        self.action = action.upper()  # Ensure uppercase
        self.confidence = confidence
        self.bid = bid
        self.pips = pips
        self.warning = warning
        self.volatility = volatility
        self.trend = trend.upper() if trend else None
        self.ema_ratio = ema_ratio
        self.timestamp = datetime.now()
        self.position_size = self.calculate_position_size()
        
    def calculate_position_size(self) -> float:
        """Unified position size calculation"""
        base_size = 1.5  # Standard base size
        confidence_factor = {
            'high': 2.0,
            'medium': 1.5,
            'low': 1.0
        }.get(self._get_confidence_level(), 1.0)
        
        volatility_adjustment = 1 / (0.5 + self.volatility * 10000)
        trend_factor = 1.5 if self.trend == self.action else 1.0
        ema_factor = 1.2 if ((self.action == 'BUY' and self.ema_ratio > 1) or 
                            (self.action == 'SELL' and self.ema_ratio < 1)) else 1.0
        
        return round(base_size * confidence_factor * volatility_adjustment * trend_factor * ema_factor, 2)
        
    def _get_confidence_level(self) -> str:
        """Unified confidence level calculation"""
        if self.confidence > 0.9:
            return 'high'
        elif self.confidence > 0.8:
            return 'medium'
        return 'low'
        
    def __str__(self):
        warning_str = " ⚠️" if self.warning else ""
        trend_str = f" | Trend: {self.trend}" if self.trend else ""
        return (f"{self.action}@{self.bid:.5f} (Conf: {self.confidence:.2f}, "
                f"Size: {self.position_size:.2f}, Pips: {self.pips:.1f}{warning_str}{trend_str})")

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
    """Order execution with risk management"""
    
    def __init__(self, api_client: ApiOrderClient, trade_direction: str = 'buy'):
        self.api = api_client
        self.account = AccountManager(api_client)
        self.logger = training_order_logger 
        
        # Configuration
        self.trade_direction = trade_direction.lower()
        self.min_pips_diff = 1.5
        self.profit_target = 3.0
        self.max_drawdown = 1.0
        self.min_confidence = 0.6
        self.force_close_threshold = -1.0
        self.min_time_between_trades = 60  # seconds
        self.max_trades_per_hour = 12
        self.trade_counter: Deque[datetime] = collections.deque(maxlen=self.max_trades_per_hour)
        
        # State
        self.last_action: Optional[str] = None
        self.last_action_time: Optional[datetime] = None
        
        self.max_eur_loss = 8.00  # Maksymalna strata w EUR
        self._monitor_task = None  # Referencja do taska monitorującego

    async def initialize(self) -> None:
        """Initialize account and load history"""
        if not await self.account.refresh_data():
            raise RuntimeError("Account initialization failed")
        self._load_trade_history()
        self._start_monitoring()

    def _load_trade_history(self) -> None:
        """Load trade history for rate limiting"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        self.trade_counter = collections.deque(
            t for t in self.trade_counter 
            if t > hour_ago
        )
        
        
    def _start_monitoring(self):
        """Uruchamia ciągłe monitorowanie pozycji"""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._continuous_monitor())
            self.logger.info("🔄 Monitoring started")  # Dodaj log potwierdzający
            
            
    def _calculate_pip_value(self, symbol: str, lot_size: float) -> float:
        """Poprawione obliczanie wartości pipsa"""
        if 'JPY' in symbol:
            return lot_size * 0.01  # 1 pip = 0.01 dla JPY
        return lot_size * 0.0001  # 1 pip = 0.0001 dla innych par
                
            
            
    async def _continuous_monitor(self):
        """Bezpieczne monitorowanie z synchronizacją"""
        while True:
            try:
                async with self.api.connection_lock:  # Używamy locka z ApiOrderClient
                    await self.account.refresh_data()
                    position = await self.account.get_open_position()
                    
                    if position:
                        current_profit = float(position['profit'])
                        symbol = position.get('symbol', 'EURUSD')
                        position_size = float(position['size'])
                        
                        pip_value = self._calculate_pip_value(symbol, position_size)
                        profit_pips = current_profit / pip_value if pip_value != 0 else 0
                        
                        # Poprawione logowanie (używamy abs() dla pipsów)
                        self.logger.info(
                            f"📊 Position: {position['type']} | "
                            f"Profit: {current_profit:.2f} EUR | "
                            f"Pips: {abs(profit_pips):.2f}"
                        )
                        
                        if current_profit <= -self.max_eur_loss:
                            self.logger.warning(f"🚨 Closing at LOSS: {current_profit:.2f} EUR")
                            await self._close_position()
                        
                        if abs(profit_pips) >= self.profit_target:
                            self.logger.info(f"🎯 Closing at PROFIT: {abs(profit_pips):.2f} pips")
                            await self._close_position()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {str(e)}")
                await asyncio.sleep(5)

    def _simple_close_check(self, position: dict, profit_pips: float) -> Optional[str]:
        """Uproszczona wersja sprawdzania warunków zamknięcia bez sygnału"""
        try:
            if profit_pips <= -self.max_drawdown:
                return f"Stop loss at {profit_pips:.2f} pips"
            
            if profit_pips >= self.profit_target:
                return f"Take profit at {profit_pips:.2f} pips"
                
            return None
        except Exception as e:
            self.logger.error(f"Close check error: {str(e)}")
            return None
            

    async def close(self):
        """Zamyka executor i task monitorujący"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        await super().close()
        
        
        
        
        
        
    

    async def _check_trade_limits(self) -> bool:
        """Check trade frequency limits"""
        now = datetime.now()
        
        # Time between trades check
        if (self.last_action_time and 
            (now - self.last_action_time).total_seconds() < self.min_time_between_trades):
            self.logger.debug("Skipping - too frequent trading")
            return False
            
        # Hourly trade limit check
        if len(self.trade_counter) >= self.max_trades_per_hour:
            self.logger.warning("Hourly trade limit reached")
            return False
            
        return True

    async def execute(self, signal: TradingSignal) -> bool:
        """Execute trade with risk management - SINGLE POSITION VERSION"""
        try:
            # Validate trade limits
            if not await self._check_trade_limits():
                return False
            
           
                
            # Validate trade direction
            if (self.trade_direction == 'buy' and signal.action != 'BUY') or \
            (self.trade_direction == 'sell' and signal.action != 'SELL'):
                self.logger.debug(f"Ignored {signal.action} (trading only {self.trade_direction.upper()})")
                return False
                
            # Validate confidence
            if signal.confidence < self.min_confidence:
                self.logger.debug(f"Skipping {signal} - low confidence")
                return False
                
            # Refresh account data
            if not await self.account.refresh_data():
                self.logger.error("Failed to refresh account data")
                return False

            # Get any open position (regardless of direction)
            any_position = await self.account.get_open_position()
            
            if any_position:
                # ALWAYS CLOSE existing position before opening new one
                self.logger.info(f"Closing existing {any_position['type']} position to open new {signal.action}")
                if not await self._close_position():
                    return False
                    
                # Add delay after closing position
                await asyncio.sleep(1)
                
            current_position = await self.account.get_open_position()
                
            if current_position:
                self.monitor_profit()
                return await self._handle_existing_position(signal, current_position)
            return await self._open_new_position(signal)

        except Exception as e:
            self.logger.error(f"Execution error: {str(e)}")
            return False
            
    async def _handle_existing_position(self, signal: TradingSignal, position: Dict[str, Any]) -> bool:
        """Handle case when position already exists"""
        if signal.action != position['type']:
            self.monitor_profit()
            print("_handle_existing)position: OrderExecutor")
            self.logger.info(f"Opposite signal ({signal.action}), closing {position['type']} position")
            return await self._close_position()

        close_reason = await self._should_close_position(signal, position)
        if close_reason:
            print("_handle_existing)position: OrderExecutor")
            self.logger.info(f"Closing condition: {close_reason}")
            return await self._close_position()

        self.logger.debug(f"Skipping {signal.action} - position already open")
        return False

    async def _should_close_position(self, signal: TradingSignal, position: Dict[str, Any]) -> Optional[str]:
        """Decide whether to close an open position based on loss, profit or signal quality."""
        try:
            position_size = float(position['size'])
            current_profit = float(position['profit'])  # in EUR
            symbol = position.get('symbol', 'EURUSD')
            
         
            
            
            pip_value = self._calculate_pip_value(symbol, position_size)
            profit_pips = current_profit / pip_value if pip_value != 0 else 0

            # Konfiguracja: możesz zmienić ten próg np. na self.force_loss_eur
            force_loss_eur = -2.00
            profit_target_pips = self.profit_target  # np. 5

            # 🔍 Debug log
            self.logger.debug(
                f"[CheckClose] Profit: {current_profit:.2f} EUR | "
                f"PipValue: {pip_value:.4f} | ProfitPips: {profit_pips:.2f} | "
                f"Conf: {signal.confidence:.2f}"
            )

            # 🔴 ZAMYKANIE NA STRACIE W EUR
            if current_profit <= force_loss_eur:
                return f"Force close: loss = {current_profit:.2f} EUR (threshold: {force_loss_eur} EUR)"
            
            if current_profit <= force_loss_eur:
                return f"Force close: loss = {current_profit:.2f} EUR (threshold: {force_loss_eur} EUR)"

            # ✅ ZAMYKANIE NA ZYSKU W PIPSACH
            if profit_pips >= profit_target_pips:
                return f"Take profit at {profit_pips:.1f} pips"

            # ⚠️ ZAMYKANIE NA NISKIM ZAUFANIU
            if signal.confidence < 0.6:
                return f"Low confidence ({signal.confidence:.2f})"

            return None

        except Exception as e:
            self.logger.error(f"Error in _should_close_position: {str(e)}")
            return None


    # def _calculate_pip_value(self, symbol: str, lot_size: float) -> float:
    #     """Calculate pip value for given symbol and lot size"""
    #     if 'JPY' in symbol:
    #         return lot_size * 0.1  # For JPY pairs
    #     return lot_size * 0.0001  # For most pairs
    
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
        """Close current position"""
        try:
            result = await self.api.close_order()
            
            if not isinstance(result, dict):
                self.logger.error(f"Invalid response type: {type(result)}")
                return False

            if result.get('status') == 'success':
                self.logger.info("Position closed successfully")
                self.last_action = None
                self.last_action_time = datetime.now()
                self.trade_counter.append(datetime.now())
                return True
                
            self.logger.error(f"Close failed: {result.get('error', 'unknown')}")
            return False
        except Exception as e:
            self.logger.error(f"Close error: {str(e)}")
            return False

    def _log_execution(self, signal: TradingSignal) -> None:
        """Log execution details"""
        conf_level = signal._get_confidence_level()
        color = '🟢' if signal.action == 'BUY' else '🔴'
        symbol = {
            'high': '★★★',
            'medium': '★★',
            'low': '★'
        }.get(conf_level, '?')
        
        log_msg = (f"{color}{symbol} {signal.action} @ {signal.bid:.5f} | "
                  f"Conf: {signal.confidence:.2f} ({conf_level}) | "
                  f"Size: {signal.position_size:.2f}")
                  
        if signal.warning:
            log_msg += " ⚠️"
            
        self.logger.info(log_msg)
        
    def monitor_profit(self):
        result = self.account.get_account()
        profit = result['profit']
            
        print("tu jest profit",profit)
            
        if profit <= -2.00:
                return  self._close_position()
        elif profit <= -2.00:
                return  self._close_position()
            # Validate trade limits
      
        return True
    


class AccountManager:
    """Account management with performance tracking"""
    
    def __init__(self, api_order_client: ApiOrderClient):
        if api_order_client is None:
            raise ValueError("api_order_client cannot be None")
            
        self.api_client = api_order_client
        self.logger = training_account_logger
        
        # Account data
        self.account_info: Dict[str, Any] = {}
        self.open_positions: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.balance_history: List[Dict[str, Any]] = []
        self.hourly_performance: Dict[int, Dict[str, int]] = {h: {'correct': 0, 'total': 0} for h in range(24)}
        
    async def refresh_data(self) -> bool:
        """Refresh account data from API"""
        try:
            # Get account info
            account_response = await self.api_client.get_account_info()
            if not isinstance(account_response, dict) or 'result' not in account_response:
                self.logger.error(f"Invalid account response: {account_response}")
                return False
                
            self.account_info = account_response['result']
            
            # Get open positions
            positions_response = await self.api_client.get_open_positions()
            if not isinstance(positions_response, dict):
                self.logger.error(f"Invalid positions response: {positions_response}")
                self.open_positions = []
            else:
                self.open_positions = positions_response.get('result', [])
                
            # Update balance history
            current_balance = self.get_account().get('balance', 0)
            self.balance_history.append({
                'timestamp': datetime.now(),
                'balance': current_balance,
                'equity': self.get_account().get('equity', 0)
            })
            
            # Clean old hourly performance data
            self._clean_old_performance_data()
            
            return True
        except Exception as e:
            self.logger.error(f"Refresh error: {str(e)}")
            return False
        
    def _clean_old_performance_data(self) -> None:
        """Clean performance data older than 24 hours"""
        now = datetime.now()
        if len(self.balance_history) > 24 * 60:  # Keep max 24 hours of minute data
            self.balance_history = self.balance_history[-24*60:]
            
        # Reset hourly performance at midnight
        if now.hour == 0 and now.minute == 0:
            self.hourly_performance = {h: {'correct': 0, 'total': 0} for h in range(24)}

    def get_account(self) -> Dict[str, Any]:
        """Get formatted account info"""
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
        
    async def get_open_positions(self, direction: str = None) -> List[Dict[str, Any]]:
        """Get all open positions, optionally filtered by direction"""
        await self.refresh_data()
        
        if not self.open_positions or not isinstance(self.open_positions, list):
            return []

        positions = []
        for position in self.open_positions:
            pos_data = {
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
            if direction is None or pos_data['type'] == direction:
                positions.append(pos_data)
        
        return positions
        
    async def has_open_position(self, direction: str = None) -> bool:
        """Check if position exists (optionally of specific direction)"""
        return len(await self.get_open_positions(direction)) > 0

    async def get_profit(self) -> float:
        """Get current profit"""
        if await self.has_open_position():
            position = await self.get_open_position()
            return float(position.get('profit', 0))
        return float(self.get_account().get('profit', 0))

    async def get_account_metrics(self) -> Dict[str, Any]:
        """Get comprehensive account metrics"""
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
        
    async def _calculate_daily_performance(self) -> Dict[str, Any]:
        """Calculate today's performance"""
        today = datetime.now().date()
        today_trades = [
            t for t in self.trade_history 
            if t.get('exit_time', datetime.now()).date() == today
        ]
        
        if not today_trades:
            return {
                'trades': 0,
                'profit': 0,
                'win_rate': 0,
                'avg_profit': 0
            }
        
        profit = sum(t.get('profit', 0) for t in today_trades)
        winning_trades = sum(1 for t in today_trades if t.get('profit', 0) > 0)
        win_rate = winning_trades / len(today_trades)
        
        return {
            'trades': len(today_trades),
            'profit': profit,
            'win_rate': win_rate,
            'avg_profit': profit / len(today_trades)
        }

class TrendAnalyzer:
    """Trend analysis with adaptive thresholds"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.price_history: Deque[float] = collections.deque(maxlen=window_size)
        self.trend: Optional[str] = None
        self.trend_strength: float = 0.0
        self.last_update: Optional[datetime] = None
    
    def update(self, current_price: float) -> Optional[str]:
        """Update with new price and return current trend"""
        self.price_history.append(current_price)
        self.last_update = datetime.now()
        
        if len(self.price_history) < self.window_size:
            return None
            
        avg = sum(self.price_history) / len(self.price_history)
        std_dev = np.std(list(self.price_history))
        
        # Calculate trend strength
        self.trend_strength = abs(current_price - avg) / (std_dev + 1e-8)
        
        if current_price > avg + std_dev * 0.5:
            self.trend = 'BUY'
        elif current_price < avg - std_dev * 0.5:
            self.trend = 'SELL'
        else:
            self.trend = None
        
        return self.trend
        
    def should_favor_downtrend(self) -> bool:
        """Check if market favors downtrend"""
        if len(self.price_history) < self.window_size:
            return False
            
        avg_price = np.mean(self.price_history)
        downtrend_ratio = sum(1 for p in self.price_history if p < avg_price) / len(self.price_history)
        return downtrend_ratio > 0.6
        
    def get_trend_strength(self) -> float:
        """Get current trend strength"""
        return self.trend_strength
        
    def is_trend_stale(self, threshold_minutes: int = 30) -> bool:
        """Check if trend data is stale"""
        if not self.last_update:
            return True
        return (datetime.now() - self.last_update).total_seconds() > threshold_minutes * 60

class Std(stats.Var):
    """Standard deviation extension for River"""
    def get(self) -> float:
        return math.sqrt(super().get())

stats.Std = Std

class EnhancedEMA:
    """Enhanced EMA with adaptive learning and performance tracking"""
    
    class_performance = {'short': {'correct': 0, 'total': 0}, 
                        'long': {'correct': 0, 'total': 0}}
    
    def __init__(self, period: int = 5):
        self.period = period
        self.value: Optional[float] = None
        self.alpha = 2 / (period + 1)
        self.volatility = stats.Var()
        self.performance = {'correct': 0, 'total': 0}
        self.last_update: Optional[datetime] = None

    def update(self, price: float, actual_direction: Optional[int] = None) -> 'EnhancedEMA':
        """Update EMA with new price"""
        if self.value is None:
            self.value = price
        else:
            std_dev = math.sqrt(self.volatility.get())
            vol_adjusted_alpha = min(0.5, self.alpha * (1 + 0.5 * math.tanh(std_dev * 10000)))
            self.value = price * vol_adjusted_alpha + self.value * (1 - vol_adjusted_alpha)
            
            if actual_direction is not None:
                predicted_direction = 1 if self.value > price else 0
                self._update_performance(predicted_direction, actual_direction)
        
        self.volatility.update(price)
        self.last_update = datetime.now()
        return self
    
    def _update_performance(self, predicted: int, actual: int) -> None:
        """Update performance metrics"""
        self.performance['total'] += 1
        if predicted == actual:
            self.performance['correct'] += 1
            
        key = 'short' if self.period <= 10 else 'long'
        self.class_performance[key]['total'] += 1
        if predicted == actual:
            self.class_performance[key]['correct'] += 1
    
    def get_accuracy(self) -> float:
        """Get prediction accuracy"""
        return (self.performance['correct'] / self.performance['total'] 
                if self.performance['total'] > 0 else 0.5)
    
    def get(self) -> float:
        """Get current EMA value"""
        return self.value if self.value is not None else 0.0
        
    def is_stale(self, threshold_minutes: int = 5) -> bool:
        """Check if EMA is stale"""
        if not self.last_update:
            return True
        return (datetime.now() - self.last_update).total_seconds() > threshold_minutes * 60

class RiverModelTrainer:
    """Online learning model for trading signals"""
    
    def __init__(self, model_save_path: str = "/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/river_models/river_model_v16.pkl.gz"):
        self.logger = training_logger
        self.model_save_path = Path(model_save_path)
        self.model = self._load_or_init_model()
        self._setup_metrics()
        self._setup_indicators()
        self._init_state()
        
        self.initialized = False
        self.initial_ticks_required = 100 
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        
    def _init_model(self) -> compose.Pipeline:
        """Initialize fresh model"""
        return compose.Pipeline(
            ('features', preprocessing.StandardScaler()),
            ('model', linear_model.LogisticRegression(
                optimizer=optim.SGD(0.015),
                l2=0.03,
                intercept_lr=0.2
            ))
        )

    def _setup_metrics(self) -> None:
        """Initialize performance metrics"""
        self.metrics = {
            'accuracy': metrics.Accuracy(),
            'precision': metrics.Precision(),
            'recall': metrics.Recall()
        }
        self.drift_detector = drift.ADWIN(delta=0.002)
        self.hourly_performance = {h: {'correct': 0, 'total': 0} for h in range(24)}

    def _setup_indicators(self) -> None:
        """Initialize technical indicators"""
        self.indicators = {
            'ema_short': EnhancedEMA(5),
            'ema_long': EnhancedEMA(20),
            'sma_short': stats.Mean(),
            'sma_long': stats.Mean(),
            'volatility': stats.Std(),
            'momentum': stats.Mean()
        }
        self.trend_analyzer = TrendAnalyzer()
        self.price_buffer: Deque[float] = collections.deque(maxlen=100)
        
        # Trading parameters
        self.min_pips_threshold = 0.00015
        self.warning_threshold = 0.15
        self.volatility_multiplier = 1.5
        self.required_trend_strength = 0.5
        self.dynamic_pips_threshold = True
        self.min_signal_interval = 60 

    def _init_state(self) -> None:
        """Initialize state variables"""
        self.tick_count = 0
        self.save_interval = 100
        self.prev_bid: Optional[float] = None
        self.prev_prev_bid: Optional[float] = None
        self.last_signal_time: Optional[datetime] = None
        self.last_signal: Optional[Dict[str, Any]] = None

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI without caching"""
        if len(self.price_buffer) < period:
            return 50.0
                
        deltas = np.diff(list(self.price_buffer)[-period:])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _generate_features(self, bid: float, ask: float) -> Dict[str, float]:
        """Generate feature vector for prediction"""
        if self.prev_bid is None:
            return {}
            
        spread = ask - bid
        momentum = bid - self.prev_bid
        
        ema_long = self.indicators['ema_long'].get() or 1e-8
        sma_long = self.indicators['sma_long'].get() or 1e-8
        volatility = self.indicators['volatility'].get() or 1e-8
        
        return {
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

    def _validate_signal(self, X: Dict[str, float], y_pred: int) -> bool:
        """Validate signal against trading rules"""
        if not X or self.prev_bid is None:
            return False
            
        volatility = X['volatility'] or 1e-8
        min_pips = max(
            self.min_pips_threshold, 
            volatility * self.volatility_multiplier * 0.0001
        ) if self.dynamic_pips_threshold else self.min_pips_threshold
        
        # Trading rules
        rules = [
            abs(X['momentum']) > (min_pips * 0.5),
            abs(X['bollinger']) < 2.5,
            (X['ema_ratio'] - 1) * (y_pred - 0.5) > -0.1,
            X['rsi'] < 80 if y_pred == 1 else X['rsi'] > 20,
            volatility < (X['bid'] * 0.01),
            self.trend_analyzer.get_trend_strength() > self.required_trend_strength,
            not (y_pred == 1 and self.trend_analyzer.trend == 'SELL' and X['confidence_weighted_ema'] < 0.5),
            not (y_pred == 0 and self.trend_analyzer.trend == 'BUY' and X['confidence_weighted_ema'] < 0.5)
        ]
        return all(rules)

    def is_prediction_about_to_change(self, X: Dict[str, float], current_pred: int) -> bool:
        """Check if prediction is likely to change"""
        proba = self.model.predict_proba_one(X)
        current_conf = proba.get(current_pred, 0.5)
        opposite_conf = proba.get(1 - current_pred, 0.5)
        volatility_factor = 1 + (X['volatility'] * 10000)
        return opposite_conf > (current_conf - (self.warning_threshold / volatility_factor))

    def process_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process new tick data and generate signals"""
        try:
            # Validate tick data
            if not all(k in tick_data for k in ['time', 'bid', 'ask']):
                self.logger.error("Invalid tick data format")
                return None
                
            tick = {
                "time": tick_data["time"],
                "bid": float(tick_data["bid"]),
                "ask": float(tick_data["ask"])
            }

            # Check minimum time between signals
            current_time = datetime.now()
            if (self.last_signal_time and 
                (current_time - self.last_signal_time).total_seconds() < self.min_signal_interval):
                return None

            # Update indicators
            for indicator in self.indicators.values():
                indicator.update(tick["bid"])
                
            self.trend_analyzer.update(tick["bid"])
            self.price_buffer.append(tick["bid"])

            # Skip first tick (need previous data)
            if self.prev_bid is None:
                self.prev_bid = tick["bid"]
                return None

            # Generate features and predict
            X = self._generate_features(tick["bid"], tick["ask"])
            y_true = 1 if tick["bid"] > self.prev_bid else 0
            y_pred = self.model.predict_one(X)
            
            # Validate signal
            if not self._validate_signal(X, y_pred):
                self._update_previous_prices(tick["bid"])
                return None

            # Learn from new data
            self.model.learn_one(X, y_true)
            
            # Update metrics
            for metric in self.metrics.values():
                metric.update(y_true, y_pred)

            # Check for concept drift
            error = 0 if y_true == y_pred else 1
            self.drift_detector.update(error)
            if self.drift_detector.drift_detected:
                self._handle_concept_drift()

            # Calculate signal properties
            pips = round((tick["bid"] - self.prev_bid) * 10000, 1)
            confidence = max(self.model.predict_proba_one(X).values(), default=0.5)
            warning = self.is_prediction_about_to_change(X, y_pred)
            
            # Update hourly performance
            current_hour = datetime.now().hour
            self.hourly_performance[current_hour]['total'] += 1
            if y_true == y_pred:
                self.hourly_performance[current_hour]['correct'] += 1
                
            # Prepare signal
            signal = {
                "time": tick["time"],
                "bid": tick["bid"],
                "prediction": y_pred,
                "confidence": confidence,
                "pips": pips,
                "warning": warning,
                "volatility": X['volatility'],
                "metrics": {k: round(v.get(), 4) for k, v in self.metrics.items()},
                "additional_metrics": {
                    'hourly_accuracy': round(
                        self.hourly_performance[current_hour]['correct'] / 
                        max(1, self.hourly_performance[current_hour]['total']), 
                        4
                    ),
                    'ema_short_accuracy': round(
                        EnhancedEMA.class_performance['short']['correct'] / 
                        max(1, EnhancedEMA.class_performance['short']['total']), 
                        4
                    ),
                    'ema_long_accuracy': round(
                        EnhancedEMA.class_performance['long']['correct'] / 
                        max(1, EnhancedEMA.class_performance['long']['total']), 
                        4
                    ),
                    'confidence_level': self._get_confidence_level(confidence),
                    'trend': self.trend_analyzer.trend
                }
            }

            self._log_signal(signal)
            self._update_model_state(tick["bid"], signal)
            
            return {
                "action": "BUY" if y_pred == 1 else "SELL",
                **signal,
                "position_size": self._calculate_position_size(confidence, X['volatility']),
                "volatility": X['volatility'],
                "ema_ratio": X.get('ema_ratio', 1.0)  # Dodaj to z domyślną wartością 1.0
            }

        except Exception as e:
            self.logger.error(f"Tick processing error: {str(e)}")
            return None
            
    def _update_previous_prices(self, current_bid: float) -> None:
        """Update price history"""
        self.prev_prev_bid = self.prev_bid
        self.prev_bid = current_bid

    def _update_model_state(self, current_bid: float, signal: Dict[str, Any]) -> None:
        """Update model state after signal generation"""
        self.tick_count += 1
        
        # Wyświetl postęp inicjalizacji
        if not self.initialized:
            if self.tick_count % 10 == 0 or self.tick_count == self.initial_ticks_required:
                remaining = max(0, self.initial_ticks_required - self.tick_count)
                self.logger.info(f"🔄 Inicjalizacja modelu: {self.tick_count}/{self.initial_ticks_required} ticków | Pozostało: {remaining}")
                
            if self.tick_count >= self.initial_ticks_required and not self.initialized:
                self.initialized = True
                self.logger.info("✅ Model w pełni zainicjalizowany! Rozpoczynam generowanie sygnałów")
        
        # Reszta istniejącej metody
        if self.tick_count % self.save_interval == 0:
            self.save_model()

        self._update_previous_prices(current_bid)
        self.last_signal = signal
        self.last_signal_time = datetime.now()

    def _handle_concept_drift(self) -> None:
            """Handle concept drift detection"""
            self.logger.warning("Concept drift detected! Resetting model...")
            self.model = self._init_model()
            self.drift_detector = drift.ADWIN(delta=0.002)
            
            # Reset indicators that might be affected by drift
            for indicator in self.indicators.values():
                if isinstance(indicator, (stats.Mean, stats.Std)):
                    indicator = type(indicator)()  # Reset statistical indicators

    def _get_confidence_level(self, confidence: float) -> str:
        """Classify confidence level"""
        if confidence >= 0.9:
            return 'high'
        elif confidence >= 0.8:
            return 'medium'
        return 'low'
        
    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """Calculate position size (now unified with TradingSignal)"""
        signal = TradingSignal(
            action="BUY",  # Temporary, actual action doesn't affect size
            confidence=confidence,
            bid=0,  # Not used
            pips=0,  # Not used
            warning=False,
            volatility=volatility
        )
        return signal.position_size
    
    def _log_signal(self, signal: Dict[str, Any]) -> None:
        """Log signal details"""
        log_entry = {
            "time": signal["time"],
            "bid": signal["bid"],
            "prediction": signal["prediction"],
            "confidence": signal["confidence"],
            "pips": signal["pips"],
            "warning": signal["warning"],
            "metrics": signal["metrics"],
            "additional_metrics": signal["additional_metrics"]
        }
        self.logger.info(f"River-V15-DECISION: {json.dumps(log_entry)}")

    def save_model(self) -> None:
        """Save model to disk"""
        try:
            with gzip.open(self.model_save_path, "wb") as f:
                pickle.dump(self.model, f)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.model_save_path.parent / f"river_model_v16_{timestamp}.pkl.gz"
            with gzip.open(backup_path, "wb") as f:
                pickle.dump(self.model, f)
                
        except Exception as e:
            self.logger.error(f"Model save failed: {str(e)}")

    def _load_or_init_model(self) -> compose.Pipeline:
        """Load existing model or initialize new one"""
        if self.model_save_path.exists():
            try:
                with gzip.open(self.model_save_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"Model load failed: {str(e)}")
        return self._init_model()
        
    def _handle_exit(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals"""
        self.logger.info("Shutting down gracefully...")
        self.save_model()
        sys.exit(0)

async def train_on_historical_data():
    """Optimized training loop with increased signal frequency"""
    data_path = "/home/tomasz/.wine/drive_c/projects/"
    csv_file = os.path.join(data_path, "m1_data.csv")
    
    # Initialize with more sensitive parameters
    model = RiverModelTrainer()
    model.min_pips_threshold = 0.00010  # Lower threshold for signals
    model.warning_threshold = 0.20  # More tolerant warning level
    model.required_trend_strength = 0.3  # Lower trend strength requirement
    
    api_client = ApiOrderClient()
    executor = OrderExecutor(api_client, trade_direction='both')  # Allow both buy/sell
    
    # Load and prepare data
    reader = CsvDataReader(csv_file)
    historical_data = reader.get_historical_data("2025-07-28 09:44:24")
    
    if historical_data is None:
        logger.error("Failed to load historical data")
        return

    historical_data['time'] = pd.to_datetime(historical_data['time'])
    historical_data = historical_data.sort_values('time').reset_index(drop=True)
    
    logger.info(f"Starting training on {len(historical_data)} records from {historical_data['time'].iloc[0]} to {historical_data['time'].iloc[-1]}")
    
    total_signals = 0
    executed_signals = 0
    last_time = None
    initialization_ticks = 150  # Reduced warmup period

    for index, row in historical_data.iterrows():
        current_time = row['time']
        
        # Faster simulation (0.01s delay max)
        if last_time is not None:
            time_diff = (current_time - last_time).total_seconds()
            await asyncio.sleep(min(time_diff, 0.01))
        last_time = current_time

        tick_data = {
            "time": current_time.strftime('%Y-%m-%d %H:%M:%S'),
            "bid": float(row['bid']),
            "ask": float(row['ask'])
        }
        
        print(tick_data)
        await asyncio.sleep(2) 
        # Process tick
        decision = model.process_tick(tick_data)
        
        # Initial warmup period
        if index < initialization_ticks:
            if index == initialization_ticks - 1:
                logger.info("Model warmup completed! Starting signal generation...")
            continue
            
        if not decision:
            continue
            
        total_signals += 1
        signal = TradingSignal(
            action=decision['action'],
            confidence=decision['confidence'],
            bid=decision['bid'],
            pips=decision['pips'],
            warning=decision['warning'],
            volatility=decision['volatility'],
            trend=decision['additional_metrics']['trend'],
            ema_ratio=decision['ema_ratio']
        )

        # Enhanced output formatting
        conf_level = signal._get_confidence_level()
        color = '🟢' if signal.action == 'BUY' else '🔴'
        symbol = '★' * (3 if conf_level == 'high' else 2 if conf_level == 'medium' else 1)
        size_info = f"×{signal.position_size:.2f}" if signal.position_size != 1.0 else ""
        
        logger.info(f"\n{color}{symbol} {signal.action}{size_info} @ {signal.bid:.5f}")
        logger.info(f"   Pips: {signal.pips:.1f} | Conf: {signal.confidence:.2f}{' ⚠️' if signal.warning else ''}")
        logger.info(f"   Volatility: {signal.volatility*10000:.2f}pips | Trend: {signal.trend}")

        # Execute with retry logic
        try:
            if await executor.execute(signal):
                executed_signals += 1
                logger.info(f"   ✅ Executed (Total: {executed_signals}/{total_signals})")
            else:
                logger.info("   ❌ Execution skipped (rules not met)")
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            
     
        # Progress every 100 signals or 10% of data
        if total_signals % 100 == 0 or index % (len(historical_data)//10) == 0:
            progress = index/len(historical_data)*100
            logger.info(f"\n=== Progress: {progress:.1f}% ===")
            logger.info(f"Last tick: {current_time}")
            logger.info(f"Signals: {total_signals} | Executed: {executed_signals}")
            logger.info(f"Execution rate: {executed_signals/max(1,total_signals)*100:.1f}%\n")

    # Final report
    model.save_model()
    logger.info(f"\n=== FINAL REPORT ===")
    logger.info(f"Total ticks processed: {len(historical_data)}")
    logger.info(f"Signals generated: {total_signals}")
    logger.info(f"Trades executed: {executed_signals}")
    logger.info(f"Execution rate: {executed_signals/max(1,total_signals)*100:.1f}%")
    logger.info(f"First signal: {historical_data['time'].iloc[initialization_ticks]}")
    logger.info(f"Last signal: {last_time}")
 

if __name__ == "__main__":
    asyncio.run(train_on_historical_data())