
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
import os

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")

# Ensure 'title' column exists
if 'title' not in df.columns:
    raise ValueError("Dataset must contain a 'title' column")

# Text preprocessing
titles = df['title'].astype(str).tolist()  # Convert all titles to string

# Tokenizer setup
tokenizer = Tokenizer()
tokenizer.fit_on_texts(titles)  # Learn word indices

# Save tokenizer for later use
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert text to sequences
token_sequences = tokenizer.texts_to_sequences(titles)

# Prepare input-output sequences
input_sequences = []
for sequence in token_sequences:
    if len(sequence) > 1:  # Ensure sequence has at least two elements
        for i in range(1, len(sequence)):
            seq = sequence[:i+1]  # Generate incremental sequences
            input_sequences.append(seq)

# Padding sequences
max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Splitting into input (X) and output (y)
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Convert target (y) to categorical
y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# Model architecture
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_len-1),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)

# ✅ Save model in `.keras` format (Recommended for Streamlit)
model.save("mymodel.keras")

# Print final accuracy
final_accuracy = history.history["accuracy"][-1] * 100
print(f"Model training complete. Final accuracy: {final_accuracy:.2f}%")

# Ensure model directory exists before saving in TensorFlow format
if not os.path.exists("saved_model"):
    os.makedirs("saved_model")

# ✅ Save model in TensorFlow's `SavedModel` format (Alternative option)
model.save("saved_model/mymodel")


