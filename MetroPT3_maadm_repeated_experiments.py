"""Repeated-seed MAADM experiment for MetroPT3.

Dependencies: numpy, pandas, scipy, scikit-learn, tensorflow, xgboost, matplotlib.
Run: python maadm_repeated_experiments.py
"""
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Dense
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import wilcoxon

DATASET_FILE = 'MetroPT3.csv'
TARGET_COLUMN = 'Reservoirs'
DROP_COLUMNS = ['timestamp']
SAMPLE_SIZE = 50000
SAMPLE_SEED = 5
LABEL_THRESHOLD = 9
K_FEATURES = 10
ENCODING_DIM = 32
SPARSITY_LAMBDA = 1e-4
EPOCHS = 50
BATCH_SIZE = 256
CONTAMINATION = 0.1
TEST_SIZE = 0.2
SEEDS = [42, 21, 123, 456, 789]


def load_data():
    data = pd.read_csv(DATASET_FILE)
    data = data.sample(n=min(SAMPLE_SIZE, len(data)), random_state=SAMPLE_SEED)
    data = data.drop(columns=DROP_COLUMNS)
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    y = np.where(data[TARGET_COLUMN].to_numpy() > LABEL_THRESHOLD, 1, 0)
    X = data.drop(columns=[TARGET_COLUMN])
    if K_FEATURES > X.shape[1]:
        raise ValueError(f'k={K_FEATURES} exceeds {X.shape[1]} available predictors')
    if len(np.unique(y)) < 2:
        raise ValueError('Binary target contains only one class')
    return X, y


def build_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(ENCODING_DIM, activation='relu',
                    activity_regularizer=regularizers.l1(SPARSITY_LAMBDA))(inputs)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=inputs, outputs=decoded)
    encoder = Model(inputs=inputs, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder


def run_maadm(X, y, seed):
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # CPU/GPU determinism can still depend on platform and TensorFlow operations.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Scaling before the sigmoid reconstruction output avoids fitting
    # an unbounded target with a bounded activation. This is a deliberate
    # preprocessing correction relative to the original implementation.
    from sklearn.preprocessing import MinMaxScaler
    ae_scaler = MinMaxScaler()
    X_train_ae = ae_scaler.fit_transform(X_train_selected)
    X_test_ae = ae_scaler.transform(X_test_selected)

    autoencoder, encoder = build_autoencoder(X_train_ae.shape[1])
    autoencoder.fit(X_train_ae, X_train_ae, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    X_train_encoded = encoder.predict(X_train_ae, verbose=0)
    X_test_encoded = encoder.predict(X_test_ae, verbose=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    iso_forest = IsolationForest(contamination=CONTAMINATION,
                                 n_estimators=100, random_state=seed)
    normal_mask = iso_forest.fit_predict(X_train_scaled) == 1
    X_train_normal = X_train_scaled[normal_mask]
    y_train_normal = y_train[normal_mask]
    if np.unique(y_train_normal).size < 2:
        raise ValueError(f'Seed {seed}: filtered training labels have only one class')

    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                        eval_metric='logloss', random_state=seed,
                        n_jobs=1)
    xgb.fit(X_train_normal, y_train_normal)
    preds = xgb.predict(X_test_scaled)
    return {
        'seed': seed,
        'precision': precision_score(y_test, preds, average='weighted', zero_division=0),
        'recall': recall_score(y_test, preds, average='weighted', zero_division=0),
        'f1': f1_score(y_test, preds, average='weighted', zero_division=0),
    }


def summarize(results):
    df = pd.DataFrame(results).sort_values('seed')
    for metric in ['precision', 'recall', 'f1']:
        print(f'{metric}: {df[metric].mean():.3f} ± {df[metric].std(ddof=1):.3f}')
    return df


def paired_wilcoxon(results_a, results_b, metric='f1'):
    """Use only for two genuinely implemented, independently evaluated configurations."""
    a = results_a.set_index('seed')[metric]
    b = results_b.set_index('seed')[metric]
    if set(a.index) != set(b.index) or not a.index.is_unique or not b.index.is_unique:
        raise ValueError('Each configuration must have the same unique seeds')
    differences = a.sort_index() - b.sort_index()
    if np.allclose(differences, 0):
        return {'statistic': 0.0, 'p_value': 1.0}
    stat, p_value = wilcoxon(differences, alternative='two-sided',
                             zero_method='wilcox', method='auto')
    return {'statistic': float(stat), 'p_value': float(p_value)}


if __name__ == '__main__':
    X, y = load_data()
    results = []
    for seed in SEEDS:
        print(f'Running seed {seed}...', flush=True)
        results.append(run_maadm(X, y, seed))
    df = summarize(results)
    df.to_csv('maadm_metropt3_seed_results.csv', index=False)
    print('Saved maadm_metropt3_seed_results.csv')
    print('A mirrored-pair Wilcoxon test requires results from a second configuration.')
