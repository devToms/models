import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

import time

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 


training_log_file = "training_v8_1_log.txt"
training_logger = LightLogger(training_log_file, logger_name="training_logger")

evaluation_log_file = "/evaluation_v8_1_log.txt"
evaluation_logger = LightLogger(evaluation_log_file, logger_name="evaluation_logger")


# Nowe loggery dla danych
data_flow_log_file = "data_flow_v8_1_log.txt"
data_flow_logger = LightLogger(data_flow_log_file, logger_name="data_flow_logger")





sequence_logger = LightLogger(
    "sequence_stats_v8_1_log.txt",
    logger_name="sequence_logger"
)






class M1ModelV8:
    def __init__(
        self,
        sequence_length,
        num_features,
        num_targets,
        model_path="m1_model_v8_1.keras"
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_targets = num_targets
        self.model_path = model_path
        self.best_val_loss = float('inf')
        self.model = self.build_model()
        
        # NEW: Callback do zapisywania najlepszego modelu podczas fine-tuningu
        self.checkpoint = ModelCheckpoint(
                filepath=self.model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
        )

    def build_model(self) -> tf.keras.Model:
        model = Sequential()
        model.add(Input(shape=(self.sequence_length, self.num_features)))
        
        # NEW: Lepsza inicjalizacja i regularyzacja
        model.add(Conv1D(
            filters=64,  # Zwiększona liczba filtrów
            kernel_size=3,
            activation='relu',
            padding='causal',
            kernel_regularizer=regularizers.l2(0.001)  # NEW: Regularyzacja L2
        ))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))  # NEW: Zwiększony dropout
        
        # NEW: Recurrent dropout w LSTM
        model.add(LSTM(
            64,
            return_sequences=True,
            recurrent_dropout=0.2,  # NEW: Zapobiega overfittingowi
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(Dropout(0.3))
        
        model.add(LSTM(
            32,
            recurrent_dropout=0.1  # NEW
        ))
        model.add(Dropout(0.4))  # NEW: Większy dropout
        
        # NEW: Dodatkowa warstwa i regularyzacja
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(self.num_targets, activation='linear'))
        
        # NEW: Optymalizator z clipvalue i dynamicznym LR
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipvalue=0.5,  # NEW: Zapobiega eksplozji gradientów
            beta_1=0.9,
            beta_2=0.999
        )
        
        # NEW: Huber loss jest bardziej odporny na outliers niż MAE
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Zamiast 'mae'
            metrics=['mae', 'mse']  # NEW: Dodano MSE dla lepszego śledzenia
        )
        return model

    def train(self, X, y, X_val, y_val, batch_size=None, epochs=None, patience=10):
        try:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                min_delta=0.0001  # NEW: Wymaga znaczącej poprawy
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,  # NEW: Mniejszy minimalny learning rate
                verbose=1
            )
            
            

            history = self.model.fit(
                X, y,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, reduce_lr, self.checkpoint],  # NEW: Użyj zapisanego checkpointa
                verbose=1,
                shuffle=False  # NEW: Ważne dla danych czasowych
            )
            self._log_training_results(history)
            return history
        except Exception as e:
            print(f"Błąd podczas treningu: {str(e)}")
            raise

    # NEW: Metoda do łatwego dotrenowywania
    def fine_tune(self, X_new, y_new, epochs=5, batch_size=32):
        try:
            # Sprawdź czy model jest gotowy
            if not hasattr(self.model, 'optimizer'):
                print("Model nie ma optymalizatora, kompiluję...")
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='huber',
                    metrics=['mae', 'mse']
                )
            
            # Tymczasowo wyłącz checkpoint podczas debugowania
            history = self.model.fit(
                X_new, y_new,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[self.checkpoint],  # Puste na czas debugowania
                verbose=1,
                shuffle=False
            )
            
            # Ręczne zapisanie modelu po treningu
            # self.model.save(self.model_path)
            
            return history
        except Exception as e:
            print(f"Błąd podczas fine-tune: {type(e).__name__}: {str(e)}")
            raise


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
    """Helper function to log statistics about data"""
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
    # 1. Wczytanie danych
    data_path = "/home/tomasz/.wine/drive_c/projects/"
    readers = {
        'm1': CsvDataReader(os.path.join(data_path, "m1_data.csv"))
    }
    
    data_flow_logger.info("=== Rozpoczęcie przygotowywania danych ===")
    
    # Logowanie surowych danych
    raw_data = {
        'm1': readers['m1'].get_latest_data(count=4000)
    }
    
    
    raw_data['m1']['time'] = raw_data['m1']['time'].astype(str).str[:19]
    


    # data_flow_logger.info(f"Przykładowe znaczniki czasu przed przetwarzaniem: {sample_times}")
    # log_data_stats(raw_data, "raw_data", data_flow_logger)

    # 2. Przetwarzanie danych - upewnij się, że kolumna czasowa jest właściwie przetwarzana
    processor = MultiIntervalProcessor()
    processed_data = processor.process(raw_data)
    
    # Dodatkowe logowanie czasu
    data_flow_logger.info(f"Przykładowe znaczniki czasu: {processed_data.index[:5]}")
    log_data_stats(processed_data, "processed_data", data_flow_logger)
    
    # 3. Feature Engineering - upewnij się, że cechy czasowe są dodawane
    engineer = FeatureEngineer()
    featured_data = engineer.add_features(processed_data)
    
    # Weryfikacja dodanych cech czasowych
    time_features = ['hour_sin_m1', 'hour_cos_m1', 'weekday_sin_m1', 'weekday_cos_m1']
    for feat in time_features:
        if feat not in featured_data.columns:
            raise ValueError(f"Brak cechy czasowej: {feat}")
    
    log_data_stats(featured_data, "featured_data", data_flow_logger)
    
    # 4. Przygotowanie danych do skalowania
    all_features = [
        'mid_m1', 'spread_m1', 'atr_m1', 'rsi_m1', 'sma10_m1', 'sma50_m1',
        'hour_sin_m1', 'hour_cos_m1', 'weekday_sin_m1', 'weekday_cos_m1'
    ]
    
    # Weryfikacja dostępności wszystkich cech
    missing_features = [f for f in all_features if f not in featured_data.columns]
    if missing_features:
        raise ValueError(f"Brakujące cechy: {missing_features}")
    
    target_cols = ['bid_m1', 'ask_m1']
    
    # 5. Skalowanie danych
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
    
    scaler_path= "m1_model_v8_1.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)  # Utwórz folder jeśli nie istnieje
    scaler.save(scaler_path)
    
    # 6. Tworzenie sekwencji
    data_for_sequences = pd.concat([
        pd.DataFrame(X_scaled, columns=all_features, index=featured_data.index),
        pd.DataFrame(y_scaled, columns=target_cols, index=featured_data.index)
    ], axis=1)
    
    sequencer = MultiIntervalSequencer(
        sequence_length=60,
        prediction_horizon=5,
        target_columns=target_cols,
        feature_columns=all_features,
        normalize=False,
        include_time_features=True
    )
    
    X_seq, y_seq = sequencer.create_sequences(data_for_sequences)
    
    # Weryfikacja końcowych kształtów
    if X_seq.shape[2] != len(all_features):
        raise ValueError(f"Niespójność liczby cech. Oczekiwano {len(all_features)}, otrzymano {X_seq.shape[2]}")
    
    data_flow_logger.info("=== Zakończenie przygotowywania danych ===")
    
    return X_seq, y_seq
   
    
### main do dotrenowania    
# def main():
#     try:
#         # 1. Ładowanie istniejącego modelu
#         model = M1ModelV8(
#             sequence_length=30,  # Musi zgadzać się z sequence_length używanym wcześniej
#             num_features=10,    # Liczba cech (mid_m1, spread_m1, etc.)
#             num_targets=2       # Liczba targetów (bid_m1, ask_m1)
#         )
        
#         # 2. Załaduj wytrenowany model
#         model.model = model.load_or_create_model()
#         print("Model załadowany pomyślnie")
        
#         # 3. PRZYGOTOWANIE NOWYCH DANYCH do dotrenowania
#         new_raw_data = {
#             'm1': CsvDataReader(os.path.join("/home/tomasz/.wine/drive_c/projects/", "m1_data.csv"))
#                 .get_latest_data(count=100)  # Tylko nowe rekordy!
#         }
#         new_raw_data['m1']['time'] = new_raw_data['m1']['time'].astype(str).str[:19]
        
#         # 4. Przetwarzanie NOWYCH danych
#         processor = MultiIntervalProcessor()
#         processed_new = processor.process(new_raw_data)
        
#         engineer = FeatureEngineer()
#         featured_new = engineer.add_features(processed_new)
        
#         # 5. Wczytanie skalera
#         scaler = MultiUniversalScaler.load(
#             "m1_model_v8_1.pkl"
#         )
        
#         # 6. Przygotowanie danych
#         all_features = [
#             'mid_m1', 'spread_m1', 'atr_m1', 'rsi_m1', 'sma10_m1', 'sma50_m1',
#             'hour_sin_m1', 'hour_cos_m1', 'weekday_sin_m1', 'weekday_cos_m1'
#         ]
#         target_cols = ['bid_m1', 'ask_m1']
        
#         # Walidacja danych
#         missing_features = [f for f in all_features if f not in featured_new.columns]
#         if missing_features:
#             raise ValueError(f"Brakujące cechy: {missing_features}")
        
#         # Skalowanie danych
#         X_new, y_new = scaler.transform(featured_new[all_features], featured_new[target_cols])
        
#         # 7. Tworzenie sekwencji
#         data_for_new_sequences = pd.concat([
#             pd.DataFrame(X_new, columns=all_features, index=featured_new.index),
#             pd.DataFrame(y_new, columns=target_cols, index=featured_new.index)
#         ], axis=1)
        
#         sequencer = MultiIntervalSequencer(
#             sequence_length=30,
#             prediction_horizon=1,
#             target_columns=target_cols,
#             feature_columns=all_features,
#             normalize=False
#         )
#         X_new_seq, y_new_seq = sequencer.create_sequences(data_for_new_sequences)
        
#         # 8. WALIDACJA DANYCH PRZED FINE-TUNE
#         print("\n=== Dane do dotrenowania ===")
#         print("Liczba sekwencji:", len(X_new_seq))
#         print("Kształt X_new_seq:", X_new_seq.shape)
#         print("Kształt y_new_seq:", y_new_seq.shape)
#         print("Przykładowe dane X:", X_new_seq[0])
#         print("Przykładowe dane y:", y_new_seq[0])
        
#         # 9. DOTREnowanie
#         print("\n=== Rozpoczynam dotrenowywanie ===")
#         model.fine_tune(X_new_seq, y_new_seq, epochs=5, batch_size=32)
        
#         print("\n=== Dotrenowywanie zakończone pomyślnie ===")

#     except Exception as e:
#         print(f"\n!!! Błąd w głównym procesie: {str(e)} !!!")
#         import traceback
#         traceback.print_exc()
#         data_flow_logger.error(f"Błąd w głównym procesie:{str(e)}")
    
##main do treningu

def main():
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

        # --- TU: przekazujemy rozmiary sekwencji i cech do klasy! ---
        model = M1ModelV8(
            sequence_length=X_train.shape[1],
            num_features=X_train.shape[2],
            num_targets=y_train.shape[1]
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

if __name__ == "__main__":
    
    main()
    # primaldata_log_file = "primal_data_v11_log.txt"
    # primal_data_logger = LightLogger(primaldata_log_file, logger_name="primal_logger")



    # reader = CsvDataReader(os.path.join("/home/tomasz/.wine/drive_c/projects/", "m1_data.csv"))
    
    # while True:
    #     try:
    #         print("\n=== Sprawdzanie nowych danych ===")
    #         # Sprawdź, czy są nowe dane (np. porównaj z poprzednim stanem)
    #         latest_data = reader.get_latest_data_by_row_check(200)
    #         if latest_data is not None:
    #             print("Znaleziono nowe dane, rozpoczynam przetwarzanie...")
    #             main()
    #             primal_data_logger.info(f"Przetworzono nowe dane: {latest_data}")
    #         else:
    #             print("Brak nowych danych.")
            
    #         # Odczekaj 60 sekund przed kolejną iteracją
    #         time.sleep(60)
          
    #     except Exception as e:
    #         print(f"Błąd w głównej pętli: {str(e)}")
    #         primal_data_logger.error(f"Błąd w głównej pętli: {str(e)}")
    #         time.sleep(5)