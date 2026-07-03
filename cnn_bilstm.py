## CNN-BiLSTM with Enron dataset

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Concatenate

# 1. Loading New Dataset Structure
try:
    df = pd.read_csv(‘datasets/Enron.csv', engine='python', on_bad_lines='skip', encoding='latin1')
    print(f'Dataset loaded. Shape: {df.shape}')
except Exception as e:
    print(f'Error loading dataset: {e}')
    df = pd.DataFrame(columns=['subject', 'body', 'label'])

# 2. Preprocessing & Feature Engineering
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|\S+\.com\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(re.findall(r'\b\w+\b', text))

if not df.empty:
    for col in ['subject', 'body', 'label']:
        if col not in df.columns:
            df[col] = ''

    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')

    # Combine subject and body for text analysis
    df['combined_text'] = df['subject'] + " " + df['body']
    df['preprocessed_message'] = df['combined_text'].apply(preprocess_text)

    # Map labels first to filter consistent rows
    if df['label'].dtype == object:
        df['label'] = df['label'].str.lower().map({'spam': 1, 'ham': 0, '1': 1, '0': 0})

    df = df.dropna(subset=['label'])

    # =========================
    # FEATURE ENGINEERING
    # =========================

    df['msg_len'] = df['body'].astype(str).apply(len)

    df['word_count'] = df['preprocessed_message'].apply(
        lambda x: len(x.split())
    )

    # =========================
    # PREPARE FEATURES
    # =========================

    X_text = df['preprocessed_message']

    X_meta = df[['msg_len', 'word_count']]

    y = df['label']

    # =========================
    # TRAIN TEST SPLIT FIRST
    # =========================

    X_train_text, X_val_text, X_train_meta, X_val_meta, y_train, y_val = train_test_split(
        X_text,
        X_meta,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Labels:")
    print(pd.Series(y_train).value_counts())

    print("\nValidation Labels:")
    print(pd.Series(y_val).value_counts())

    # =========================
    # TOKENIZER
    # FIT ONLY ON TRAIN DATA
    # =========================

    max_len = 500
    max_words = 20000

    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token='<unk>'
    )

    tokenizer.fit_on_texts(X_train_text)

    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)

    X_val_seq = tokenizer.texts_to_sequences(X_val_text)

    # Padding
    X_train = pad_sequences(
        X_train_seq,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )

    X_val = pad_sequences(
        X_val_seq,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )

    vocab_size = min(len(tokenizer.word_index) + 1, max_words)

    # =========================
    # SCALE META FEATURES
    # FIT ONLY ON TRAIN DATA
    # =========================

    scaler = StandardScaler()

    m_train = scaler.fit_transform(X_train_meta)

    m_val = scaler.transform(X_val_meta)

    # The previous line `X_train, X_val, m_train, m_val, y_train, y_val = train_test_split(
    #     padded_X, meta_data, y, test_size=0.2, random_state=42, stratify=y
    # )` was incorrect as 'padded_X' and 'meta_data' are not defined.
    # The data has already been appropriately split and processed into X_train, X_val, m_train, m_val, y_train, y_val.
    # Therefore, this redundant and erroneous line is removed.

    # 5. Build and Train Model
    text_input = Input(shape=(max_len,), name='text_input')
    embedding = Embedding(input_dim=min(vocab_size, 20000), output_dim=128)(text_input)
    cnn = Conv1D(128, 5, activation='relu')(embedding)
    pool = MaxPooling1D(pool_size=2)(cnn)
    bilstm = Bidirectional(LSTM(64))(pool)

    meta_input = Input(shape=(2,), name='meta_input')
    combined = Concatenate()([bilstm, meta_input])

    dense = Dense(64, activation='relu')(combined)
    drop = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(drop)

    model = Model(inputs=[text_input, meta_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('Starting training on aligned samples...')
    model.fit(
        x={'text_input': X_train, 'meta_input': m_train},
        y=y_train,
        epochs=3,
        batch_size=64,
        validation_data=({'text_input': X_val, 'meta_input': m_val}, y_val)
    )

    # 6. Evaluation
    decision_threshold = 0.4 # You can change this value
    y_pred_proba = model.predict({'text_input': X_val, 'meta_input': m_val})
    y_pred = (y_pred_proba > decision_threshold).astype(int)

    print('\nClassification Report:')
    print(classification_report(y_val, y_pred))

    print(f'ROC-AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}')

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate (0)', 'Phishing (1)'], yticklabels=['Legitimate (0)', 'Phishing (1)'])
    plt.title('CNN-BiLSTM with Enron - Confusion Matrix')
    plt.show()
else:
    print("Dataset is empty or loading failed.")
