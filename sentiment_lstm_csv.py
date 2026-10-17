"""
Predict Sentiment from Movie Reviews using LSTM Networks
--------------------------------------------------------
Variant: raw-text CSV dataset + apni text preprocessing pipeline + stacked Bidirectional LSTM.

Dataset : IMDB Dataset of 50K Movie Reviews (Kaggle) -> "IMDB Dataset.csv"
          columns: review, sentiment (positive / negative)
          https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Install : pip install tensorflow pandas scikit-learn matplotlib seaborn
Run     : python sentiment_lstm_csv.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------------
DATA_PATH = "IMDB Dataset.csv"
MAX_WORDS = 15000
MAX_LEN = 250
EMBED_DIM = 100
EPOCHS = 8
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df["target"] = (df["sentiment"].str.lower() == "positive").astype(int)

print("Dataset shape:", df.shape)
print(df["sentiment"].value_counts(), "\n")

# ------------------------------------------------------------------
# 3. Text preprocessing
# ------------------------------------------------------------------
def clean_review(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags (<br /> etc.)
    text = re.sub(r"[^a-z\s']", " ", text)        # punctuation / numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["review"].apply(clean_review)

# ------------------------------------------------------------------
# 4. Split: 80% train, 20% test
# ------------------------------------------------------------------
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df["clean"], df["target"], test_size=0.2, random_state=SEED, stratify=df["target"]
)

# ------------------------------------------------------------------
# 5. Tokenize + pad
# ------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_txt)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_txt), maxlen=MAX_LEN, padding="post", truncating="post")
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_txt), maxlen=MAX_LEN, padding="post", truncating="post")
y_train, y_test = np.array(y_train), np.array(y_test)

# ------------------------------------------------------------------
# 6. Stacked Bidirectional LSTM model
# ------------------------------------------------------------------
model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(32)),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.build(input_shape=(None, MAX_LEN))
model.summary()

# ------------------------------------------------------------------
# 7. Train
# ------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ------------------------------------------------------------------
# 8. Evaluate
# ------------------------------------------------------------------
y_pred = (model.predict(X_test).ravel() > 0.5).astype(int)
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_csv.png", dpi=150)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, m, title in zip(ax, ["accuracy", "loss"], ["Accuracy", "Loss"]):
    a.plot(history.history[m], label="train")
    a.plot(history.history["val_" + m], label="val")
    a.set_title(title)
    a.set_xlabel("Epoch")
    a.legend()
plt.tight_layout()
plt.savefig("training_curves_csv.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 9. Predict sentiment of new reviews
# ------------------------------------------------------------------
def predict_sentiment(review: str) -> str:
    seq = tokenizer.texts_to_sequences([clean_review(review)])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(pad, verbose=0)[0][0])
    label = "Positive" if prob > 0.5 else "Negative"
    return f"{label}  (confidence = {max(prob, 1 - prob):.2%})"

samples = [
    "An absolute masterpiece. The acting, music and direction were brilliant!",
    "Dull, predictable and way too long. I wanted my money back.",
    "Not bad, but the second half dragged a lot.",
]
print("\n--- Custom predictions ---")
for s in samples:
    print(f"{s}\n  -> {predict_sentiment(s)}\n")

model.save("sentiment_lstm_csv.keras")
print("Model saved as sentiment_lstm_csv.keras")
