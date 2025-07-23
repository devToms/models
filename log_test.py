import pandas as pd
import json
from datetime import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_log_line(line):
    try:
        if 'DECISION' not in line:
            return None
            
        json_part = re.search(r'\{.*\}', line)
        if not json_part:
            return None
            
        data = json.loads(json_part.group())
        data['time'] = datetime.strptime(data['time'], '%Y-%m-%d %H:%M:%S')
        return data
    except Exception as e:
        print(f"Błąd parsowania linii: {line}\nError: {str(e)}")
        return None

print("Ładowanie danych rynkowych...")
market_data = pd.read_csv(
    '/home/tomasz/.wine/drive_c/projects/m1_data.csv',
    parse_dates=['time'],
    names=['time', 'bid', 'ask', 'open', 'high', 'low', 'close'],
    header=0
)
print(f"Wczytano {len(market_data)} rekordów rynkowych")

print("\nŁadowanie logów predykcji...")
predictions = []
with open('river_training_v5_log.txt', 'r') as f:
    for line in f:
        if any(x in line for x in ['River-V8-DECISION', 'River-V6-DECISION']):
            parsed = parse_log_line(line)
            if parsed:
                predictions.append(parsed)

predictions_df = pd.DataFrame(predictions)
predictions_df['time'] = pd.to_datetime(predictions_df['time'])

combined = pd.merge_asof(
    predictions_df.sort_values('time'),
    market_data.sort_values('time'),
    on='time',
    direction='nearest',
    tolerance=pd.Timedelta('5s'),
    suffixes=('_pred', '_market')
)

# Obliczenia
combined['time_diff'] = combined['time'].diff().dt.total_seconds().fillna(0)
combined['next_bid'] = combined['bid_market'].shift(-1)
combined['price_change'] = combined['next_bid'] - combined['bid_market']
combined = combined.dropna(subset=['price_change'])

combined['actual_direction'] = combined['price_change'] > 0
combined['correct_prediction'] = combined['prediction'] == combined['actual_direction']
combined['pips_gain'] = np.where(
    combined['correct_prediction'],
    abs(combined['price_change']) * 10000,
    -abs(combined['price_change']) * 10000
)

# Obliczanie trendu
combined['sma10'] = combined['bid_market'].rolling(10, min_periods=1).mean()
combined['trend'] = np.where(
    combined['bid_market'] > combined['sma10'], 
    'Uptrend', 
    'Downtrend'
)

# Metryki
metrics = {
    'accuracy': accuracy_score(combined['actual_direction'], combined['prediction']),
    'precision': precision_score(combined['actual_direction'], combined['prediction'], zero_division=0),
    'recall': recall_score(combined['actual_direction'], combined['prediction'], zero_division=0),
    'f1': f1_score(combined['actual_direction'], combined['prediction'], zero_division=0)
}

# Analiza skuteczności
combined['confidence_bin'] = pd.cut(combined['confidence'], bins=5)
confidence_analysis = combined.groupby('confidence_bin', observed=True)['correct_prediction'].mean()

# Analiza trendów
trend_analysis = combined.groupby('trend').agg({
    'correct_prediction': 'mean',
    'pips_gain': 'sum'
}).rename(columns={
    'correct_prediction': 'accuracy',
    'pips_gain': 'total_pips'
})

# Metryki finansowe
combined['cumulative_pips'] = combined['pips_gain'].cumsum()
max_drawdown = (combined['cumulative_pips'].cummax() - combined['cumulative_pips']).max()
profit_factor = combined[combined['pips_gain'] > 0]['pips_gain'].sum() / abs(combined[combined['pips_gain'] < 0]['pips_gain'].sum())

# Generowanie raportu
report = f"""
ANALIZA SKUTECZNOŚCI MODELU
---------------------------
Okres analizy: {combined['time'].min()} - {combined['time'].max()}
Liczba predykcji: {len(combined)}

METRYKI PODSTAWOWE:
- Accuracy: {metrics['accuracy']:.2%}
- Precision: {metrics['precision']:.2%}
- Recall: {metrics['recall']:.2%}
- F1-score: {metrics['f1']:.2%}

WYNIK FINANSOWY:
- Całkowity wynik: {combined['pips_gain'].sum():.1f} pipsów
- Średnio na transakcję: {combined['pips_gain'].mean():.1f} pipsów
- Profit Factor: {profit_factor:.2f}
- Maks. drawdown: {max_drawdown:.1f} pipsów

ANALIZA TRENDÓW:
Uptrend
- Skuteczność: {trend_analysis.loc['Uptrend', 'accuracy']:.2%}
- Wynik: {trend_analysis.loc['Uptrend', 'total_pips']:.1f} pipsów

Downtrend
- Skuteczność: {trend_analysis.loc['Downtrend', 'accuracy']:.2%}
- Wynik: {trend_analysis.loc['Downtrend', 'total_pips']:.1f} pipsów

ANALIZA PEWNOŚCI:
{confidence_analysis.to_string()}
"""

# Wykresy
plt.figure(figsize=(15, 12))

# Wykres cen i sygnałów
plt.subplot(3, 1, 1)
plt.plot(combined['time'], combined['bid_market'], label='Cena', alpha=0.7)
plt.plot(combined['time'], combined['sma10'], label='SMA 10', linestyle='--', alpha=0.7)
plt.scatter(
    combined[combined['prediction']].time,
    combined[combined['prediction']].bid_market,
    color='green', label='Long', marker='^', s=100
)
plt.scatter(
    combined[~combined['prediction']].time,
    combined[~combined['prediction']].bid_market,
    color='red', label='Short', marker='v', s=100
)
plt.title('Sygnály transakcyjne na tle ceny')
plt.legend()

# Wykres zysków skumulowanych
plt.subplot(3, 1, 2)
plt.plot(combined['time'], combined['cumulative_pips'], label='Zysk skumulowany')
plt.fill_between(
    combined['time'],
    combined['cumulative_pips'].cummax(),
    combined['cumulative_pips'],
    where=(combined['cumulative_pips'] < combined['cumulative_pips'].cummax()),
    color='red', alpha=0.3, label='Drawdown'
)
plt.title('Wynik finansowy (pipsy)')
plt.legend()

# Wykres skuteczności wg poziomu pewności
plt.subplot(3, 1, 3)
confidence_analysis.plot(kind='bar', rot=45)
plt.title('Skuteczność wg poziomu pewności predykcji')
plt.ylabel('Skuteczność')
plt.xlabel('Przedział pewności')

plt.tight_layout()
plt.savefig('analysis_results.png')
plt.close()

# Zapis wyników
combined.to_csv('predictions_evaluation_detailed_v?.csv', index=False)
with open('prediction_performance_report_v?.txt', 'w') as f:
    f.write(report)

print(report)
print("\nWykresy zapisano do: analysis_results_v?.png")
print("Szczegółowe dane zapisano do: predictions_evaluation_detailed_v?.csv")