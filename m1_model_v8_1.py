import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import os
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys

# Dodanie ścieżki do systemu
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Importy z Twojego projektu
from helpers.csv_data_reader import CsvDataReader
from helpers.multi_Interval_processor import MultiIntervalProcessor
from helpers.feature_engineer import FeatureEngineer
from managers.multi_universal_scaler import MultiUniversalScaler
from helpers.multi_interval_sequencer import MultiIntervalSequencer
from loggers.light_logger import LightLogger

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# Loggers (pozostają bez zmian)
training_log_file = "training_v8_log.txt"
training_logger = LightLogger(training_log_file, logger_name="training_logger")

evaluation_log_file = "evaluation_v8_log.txt"
evaluation_logger = LightLogger(evaluation_log_file, logger_name="evaluation_logger")

data_flow_log_file = "data_flow_v8_log.txt"
data_flow_logger = LightLogger(data_flow_log_file, logger_name="data_flow_logger")

feature_logger = LightLogger(
    "feature_stats_v8_log.txt",
    logger_name="feature_logger"
)

target_logger = LightLogger(
    "arget_stats_v8_log.txt",
    logger_name="target_logger"
)

sequence_logger = LightLogger(
    "sequence_stats_v8_log.txt",
    logger_name="sequence_logger"
)

class BackupCleaner(Callback):
    def __init__(self, model_instance, max_backups=3):
        super().__init__()
        self._model_instance = model_instance  # Zmieniamy nazwę na _model_instance
        self.max_backups = max_backups
        self.backup_dir = os.path.dirname(self._model_instance.model_path)
        self.base_name = os.path.basename(self._model_instance.model_path).replace('.keras', '')
    
    def on_epoch_end(self, epoch, logs=None):
        backup_files = sorted(
            [f for f in os.listdir(self.backup_dir) 
             if f.startswith(f"{self.base_name}_backup_epoch_") 
             and f.endswith('.keras')],
            key=lambda x: int(x.split('_')[-1].split('.')[0]),
            reverse=True
        )
        
        if len(backup_files) > self.max_backups:
            for old_file in backup_files[self.max_backups:]:
                os.remove(os.path.join(self.backup_dir, old_file))
                data_flow_logger.info(f"Usunięto stary backup: {old_file}")

class M1ModelV8:
    def __init__(
        self,
        sequence_length,
        num_features,
        num_targets,
        model_path="/home/tomasz/projekty/python/app_market_bot/app_market_bot/model_manager/neural_network_models/new_model_keras_files/m1_model_v8.keras",
        max_backups=3
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_targets = num_targets
        self.model_path = model_path
        self.max_backups = max_backups
        self.best_val_loss = float('inf')
        self.model = self.build_model()
        
    def _get_backup_path(self, epoch):
        base_dir = os.path.dirname(self.model_path)
        base_name = os.path.basename(self.model_path).replace('.keras', '')
        return os.path.join(base_dir, f"{base_name}_backup_epoch_{epoch:03d}.keras")

    def build_model(self) -> tf.keras.Model:
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.num_features)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_targets, activation='linear'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mae']
        )
        return model

    def train(self, X, y, X_val, y_val, batch_size=None, epochs=None, patience=10, initial_training=True):
        try:
            early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
            
            callbacks = [early_stop, reduce_lr]
            
            if initial_training:
                epochs = epochs or 250
                checkpoint = ModelCheckpoint(
                    self.model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(checkpoint)
                
                backup_pattern = os.path.join(
                    os.path.dirname(self.model_path),
                    f"{os.path.basename(self.model_path).replace('.keras', '')}_backup_epoch_{{epoch:03d}}.keras"
                )
                
                backup_callback = ModelCheckpoint(
                    filepath=backup_pattern,
                    save_freq=10 * X.shape[0] // batch_size if batch_size else 10,
                    save_best_only=False,
                    verbose=1
                )
                callbacks.append(backup_callback)
                callbacks.append(BackupCleaner(self, max_backups=self.max_backups))
            else:
                epochs = epochs or 50
                self.model.optimizer.learning_rate.assign(0.0001)
            
            history = self.model.fit(
                X, y,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            self._log_training_results(history)
            return history
        except Exception as e:
            print(f"Błąd podczas treningu: {str(e)}")
            raise

    def continue_training(self, X, y, X_val, y_val, batch_size=None, epochs=50, patience=5):
        """Metoda do dotrenowywania istniejącego modelu"""
        if not os.path.exists(self.model_path):
            raise ValueError("Model nie istnieje. Użyj metody train() do początkowego treningu.")
        
        self.model = tf.keras.models.load_model(self.model_path)
        return self.train(
            X, y, X_val, y_val,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            initial_training=False
        )

    def _log_training_results(self, history):
        for epoch in range(len(history.history['loss'])):
            log_message = (
                f"Epoch {epoch + 1}/{len(history.history['loss'])} - "
                f"loss: {history.history['loss'][epoch]:.4f} - "
                f"mae: {history.history['mae'][epoch]:.4f} - "
                f"val_loss: {history.history['val_loss'][epoch]:.4f} - "
                f"val_mae: {history.history['val_mae'][epoch]:.4f}"
            )
            training_logger.info(log_message)

    def evaluate_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {
            'loss': results[0],
            'mae': results[1]
        }

    def predict(self, X):
        try:
            return self.model.predict(X, verbose=0)
        except Exception as e:
            print(f"Błąd predykcji: {str(e)}")
            raise

    def load_or_create_model(self):
        if os.path.exists(self.model_path):
            try:
                model = tf.keras.models.load_model(self.model_path)
                print(f"Załadowano istniejący model z {self.model_path}")
                return model
            except Exception as e:
                print(f"Błąd ładowania modelu: {str(e)}. Tworzę nowy model.")
                return self.build_model()
        return self.build_model()

def log_data_stats(data, name, logger):
    if isinstance(data, (np.ndarray, pd.DataFrame)):
        if len(data.shape) == 1:
            stats = {
                'min': np.min(data),
                'max': np.max(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'nan_count': np.isnan(data).sum()
            }
        else:
            stats = {
                'shape': data.shape,
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0),
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'nan_count': np.isnan(data).sum(axis=0)
            }
        logger.info(f"{name} stats: {stats}")
    elif isinstance(data, dict):
        for key, value in data.items():
            log_data_stats(value, f"{name}.{key}", logger)

def prepare_data():
    data_path = "/home/tomasz/.wine/drive_c/projects/"
    readers = {
        'm1': CsvDataReader(os.path.join(data_path, "m1_data.csv"))
    }
    
    data_flow_logger.info("=== Rozpoczęcie przygotowywania danych ===")
    
    raw_data = {
        'm1': readers['m1'].get_latest_data(count=8000)
    }
    
    raw_data['m1']['time'] = raw_data['m1']['time'].astype(str).str[:19]

    processor = MultiIntervalProcessor()
    processed_data = processor.process(raw_data)
    
    data_flow_logger.info(f"Przykładowe znaczniki czasu: {processed_data.index[:5]}")
    log_data_stats(processed_data, "processed_data", data_flow_logger)
    
    engineer = FeatureEngineer()
    featured_data = engineer.add_features(processed_data)
    
    time_features = ['hour_sin_m1', 'hour_cos_m1', 'weekday_sin_m1', 'weekday_cos_m1']
    for feat in time_features:
        if feat not in featured_data.columns:
            raise ValueError(f"Brak cechy czasowej: {feat}")
    
    log_data_stats(featured_data, "featured_data", data_flow_logger)
    
    all_features = [
        'mid_m1', 'spread_m1', 'atr_m1', 'rsi_m1', 'sma10_m1', 'sma50_m1',
        'hour_sin_m1', 'hour_cos_m1', 'weekday_sin_m1', 'weekday_cos_m1'
    ]
    
    missing_features = [f for f in all_features if f not in featured_data.columns]
    if missing_features:
        raise ValueError(f"Brakujące cechy: {missing_features}")
    
    target_cols = ['bid_m1', 'ask_m1']
    
    feature_groups = {
        'prices': ['mid_m1'],
        'spreads': ['spread_m1'],
        'indicators': ['atr_m1', 'rsi_m1', 'sma10_m1', 'sma50_m1']
    }
    
    scaler = MultiUniversalScaler(
        feature_groups=feature_groups,
        target_columns=target_cols,
        scaler_type='robust'
    )
    
    X = featured_data[all_features]
    y = featured_data[target_cols]
    
    scaler.fit(X, y)
    X_scaled, y_scaled = scaler.transform(X, y)
    
    scaler_path= "m1_model_v8.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler.save(scaler_path)
    
    data_for_sequences = pd.concat([
        pd.DataFrame(X_scaled, columns=all_features, index=featured_data.index),
        pd.DataFrame(y_scaled, columns=target_cols, index=featured_data.index)
    ], axis=1)
    
    sequencer = MultiIntervalSequencer(
        sequence_length=30,
        prediction_horizon=1,
        target_columns=target_cols,
        feature_columns=all_features,
        normalize=False,
        include_time_features=True
    )
    
    X_seq, y_seq = sequencer.create_sequences(data_for_sequences)
    
    if X_seq.shape[2] != len(all_features):
        raise ValueError(f"Niespójność liczby cech. Oczekiwano {len(all_features)}, otrzymano {X_seq.shape[2]}")
    
    data_flow_logger.info("=== Zakończenie przygotowywania danych ===")
    
    return X_seq, y_seq

def main_initial_training():
    """Funkcja do początkowego treningu modelu"""
    try:
        X_seq, y_seq = prepare_data()
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, shuffle=False)

        data_flow_logger.info("=== Data split statistics ===")
        data_flow_logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        data_flow_logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        data_flow_logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        sequence_logger.info("=== Training data sample ===")
        sequence_logger.info(f"First X_train sample stats - min: {np.min(X_train[0])}, max: {np.max(X_train[0])}, mean: {np.mean(X_train[0])}")
        sequence_logger.info(f"First y_train sample: {y_train[0]}")
        
        sequence_logger.info("=== dane dla modelu ===")
        sequence_logger.info(f"sequence_length=X_train.shape[1]: {X_train.shape[1]}")
        sequence_logger.info(f"num_features=X_train.shape[2]: {X_train.shape[2]}")
        sequence_logger.info(f" num_targets=y_train.shape[1]: { y_train.shape[1]}")

        model = M1ModelV8(
            sequence_length=X_train.shape[1],
            num_features=X_train.shape[2],
            num_targets=y_train.shape[1],
            max_backups=3
        )
        
        print("\n=== Rozpoczynam trening ===")
        data_flow_logger.info("=== Rozpoczęcie treningu ===")

        history = model.train(
            X_train, y_train, X_val, y_val,
            batch_size=1024,
            epochs=250
        )

        test_metrics = model.evaluate_model(X_test, y_test)
        print("\n=== Wyniki testowe ===")
        evaluation_logger.info("=== Wyniki testowe ===")
        evaluation_logger.info(f"Loss: {test_metrics['loss']:.4f}")
        evaluation_logger.info(f"MAE: {test_metrics['mae']:.4f}")

        data_flow_logger.info("=== Zakończenie treningu ===")

    except Exception as e:
        print(f"Błąd w głównym procesie: {str(e)}")
        data_flow_logger.error(f"Błąd w głównym procesie: {str(e)}")

def main_continue_training():
    """Funkcja do dotrenowywania istniejącego modelu"""
    try:
        X_seq, y_seq = prepare_data()
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, shuffle=False)

        model = M1ModelV8(
            sequence_length=X_train.shape[1],
            num_features=X_train.shape[2],
            num_targets=y_train.shape[1],
            max_backups=3
        )
        
        print("\n=== Rozpoczynam dotrenowywanie ===")
        data_flow_logger.info("=== Rozpoczęcie dotrenowywania ===")

        history = model.continue_training(
            X_train, y_train, X_val, y_val,
            batch_size=1024,
            epochs=50
        )

        test_metrics = model.evaluate_model(X_test, y_test)
        print("\n=== Wyniki testowe po dotrenowaniu ===")
        evaluation_logger.info("=== Wyniki testowe po dotrenowaniu ===")
        evaluation_logger.info(f"Loss: {test_metrics['loss']:.4f}")
        evaluation_logger.info(f"MAE: {test_metrics['mae']:.4f}")

        data_flow_logger.info("=== Zakończenie dotrenowywania ===")

    except Exception as e:
        print(f"Błąd w procesie dotrenowywania: {str(e)}")
        data_flow_logger.error(f"Błąd w procesie dotrenowywania: {str(e)}")

if __name__ == "__main__":
    # Wybierz czy chcesz przeprowadzić początkowy trening czy dotrenowywanie
    main_initial_training()  # Odkomentuj dla początkowego treningu
    # main_continue_training()  # Odkomentuj dla dotrenowywania