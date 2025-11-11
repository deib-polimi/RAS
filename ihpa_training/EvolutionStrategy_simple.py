import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
import os
import argparse

def load_data(data_folder="training_data", mode='train'):
    """
    Loads the training or testing data from the specified folder.
    Now loads data with response_time as target instead of CPU utilization.
    """
    print(f"Loading {mode} data with response_time as target...")
    
    file_suffix = '_test' if mode == 'test' else ''
    
    # All data files now use response_time as target by default (no special suffix needed)
    X_recurrent_path = os.path.join(data_folder, f'X_recurrent{file_suffix}.npy')
    X_feedforward_path = os.path.join(data_folder, f'X_feedforward{file_suffix}.npy')
    y_target_path = os.path.join(data_folder, f'y_target{file_suffix}.npy')

    # Check if files exist
    for path in [X_recurrent_path, X_feedforward_path, y_target_path]:
        if not os.path.exists(path):
            print(f"Error: Data file not found at {path}")
            print("Please run the data generation script first (e.g., `generate_training_data.py --mode test`).")
            return None, None, None

    X_recurrent = np.load(X_recurrent_path)
    X_feedforward = np.load(X_feedforward_path)
    y_target = np.load(y_target_path)
    
    print("Data loaded successfully.")
    print(f"X_recurrent shape: {X_recurrent.shape}")
    print(f"X_feedforward shape: {X_feedforward.shape}")
    print(f"y_target shape: {y_target.shape}")
    
    return X_recurrent, X_feedforward, y_target

def build_double_tower_model(recurrent_input_shape, feedforward_input_shape):
    """
    Builds the Double Tower (LSTM + Dense) Keras model.
    """
    # --- Recurrent Tower (LSTM) for time-series data ---
    recurrent_input = Input(shape=recurrent_input_shape, name='recurrent_input')
    # Using two LSTM layers is a common pattern.
    lstm_layer = LSTM(100, activation='tanh', return_sequences=True)(recurrent_input)
    lstm_layer = LSTM(50, activation='tanh')(lstm_layer)
    recurrent_output = Flatten()(lstm_layer) # Flatten the output for concatenation

    # --- Feed-forward Tower (Dense) for state data ---
    feedforward_input = Input(shape=feedforward_input_shape, name='feedforward_input')
    dense_layer = Dense(50, activation='relu')(feedforward_input)
    dense_layer = Dropout(0.2)(dense_layer) # Dropout for regularization
    feedforward_output = Dense(25, activation='relu')(dense_layer)

    # --- Concatenate and final layers ---
    concatenated = concatenate([recurrent_output, feedforward_output])
    final_dense = Dense(50, activation='relu')(concatenated)
    output = Dense(1, activation='linear', name='output')(final_dense) # Linear activation for regression

    model = Model(inputs=[recurrent_input, feedforward_input], outputs=output)
    
    return model

# --- Learning Rate Scheduler ---
# Definiamo le costanti per il learning rate circolare
BASE_LR = 0.001
MAX_LR = 0.006  # 6x il learning rate di base
STEP_SIZE = 20  # Numero di epoche per metà ciclo

def triangular_cyclical_lr(epoch):
    """Funzione per il learning rate circolare con andamento triangolare."""
    cycle = np.floor(1 + epoch / (2 * STEP_SIZE))
    x = np.abs(epoch / STEP_SIZE - 2 * cycle + 1)
    lr = BASE_LR + (MAX_LR - BASE_LR) * np.maximum(0, (1 - x))
    return lr

def run_training_pipeline(model_save_path):
    """
    Runs the full model training pipeline.
    """
    # 1. Load data
    X_recurrent, X_feedforward, y_target = load_data(mode='train')
    if X_recurrent is None:
        return # Stop if data loading failed

    # 2. Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_rec_train, X_rec_val, X_ff_train, X_ff_val, y_train, y_val = train_test_split(
        X_recurrent, X_feedforward, y_target, test_size=0.2, random_state=42
    )
    print("Data split complete.")

    # 3. Build model
    print("Building model...")
    recurrent_shape = (X_rec_train.shape[1], X_rec_train.shape[2]) # (timesteps, features)
    ff_shape = (X_ff_train.shape[1],) # (features,)
    model = build_double_tower_model(recurrent_shape, ff_shape)
    
    # 4. Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.summary()

    # 5. Train model
    print("\nStarting model training...")
    # Pazienza aumentata e scheduler per il learning rate circolare
    early_stopping = EarlyStopping(monitor='val_loss', patience=10000, restore_best_weights=True, verbose=1)
    lr_scheduler = LearningRateScheduler(triangular_cyclical_lr)
    
    history = model.fit(
        [X_rec_train, X_ff_train], y_train,
        epochs=2000,
        batch_size=64,
        validation_data=([X_rec_val, X_ff_val], y_val),
        callbacks=[early_stopping, lr_scheduler],
        verbose=2 
    )
    print("Model training finished.")

    # 6. Evaluate model on validation data
    print("\nEvaluating model on validation data...")
    loss, mae = model.evaluate([X_rec_val, X_ff_val], y_val, verbose=0)
    print(f"Validation Loss (MSE): {loss:.4f}")
    print(f"Validation Mean Absolute Error: {mae:.4f}")

    # 7. Save the trained model
    output_dir = os.path.dirname(model_save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # 8. Plot and save training history
    history_plot_path = 'training_history.png'
    print(f"Saving training history plot to {history_plot_path}...")
    
    plt.figure(figsize=(18, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss (MSE) Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    # Plot Learning Rate
    plt.subplot(1, 3, 3)
    # Ricalcoliamo il learning rate per le epoche eseguite
    epochs_ran = len(history.history['loss'])
    lrs = [triangular_cyclical_lr(epoch) for epoch in range(epochs_ran)]
    plt.plot(lrs, label='Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(history_plot_path)
    print("Plot saved.")


def plot_predictions_vs_actual(y_true, y_pred, plot_path='test_set_predictions_vs_actual.png'):
    """
    Generates and saves a plot comparing true values and predictions.
    """
    print(f"Saving prediction plot to {plot_path}...")
    plt.figure(figsize=(15, 6))

    # Line plot to see the trend over samples
    plt.subplot(1, 2, 1)
    plt.plot(y_true, label='Actual Values', color='b', alpha=0.7, marker='o', linestyle='-')
    plt.plot(y_pred, label='Predicted Values', color='r', alpha=0.7, marker='x', linestyle='--')
    plt.title('Comparison of Actual vs. Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Response Time (seconds)')
    plt.legend()
    plt.grid(True)

    # Scatter plot to evaluate correlation
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    # Add a line for perfect predictions (y=x)
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Perfect Prediction')
    plt.title('Scatter Plot: Actual vs. Predicted')
    plt.xlabel('Actual Response Time (seconds)')
    plt.ylabel('Predicted Response Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print("Plot saved.")


def run_evaluation_pipeline(model_path):
    """
    Loads a pre-trained model and evaluates it on the test set.
    """
    print(f"--- Running Evaluation on Test Set ---")
    
    # 1. Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please train the model first using --mode train")
        return

    # 2. Load test data
    X_rec_test, X_ff_test, y_test = load_data(mode='test')
    if X_rec_test is None:
        return # Stop if data loading failed

    # 3. Load the pre-trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    model.summary()

    # 4. Evaluate the model on the test data
    print("\nEvaluating model on the held-out test set...")
    loss, mae = model.evaluate([X_rec_test, X_ff_test], y_test, verbose=1)
    predictions = model.predict([X_rec_test, X_ff_test])

    print("\n--- Test Set Evaluation Results ---")
    print(f"Model: {model_path}")
    print(f"Test Loss (MSE): {loss:.4f}")
    print(f"Test Mean Absolute Error: {mae:.4f}")
    print("-----------------------------------")

    # 5. Generate and save a plot of predictions vs. actual values
    plot_predictions_vs_actual(y_test, predictions)


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the Double Tower model.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test'],
        default='train',
        help="Set to 'train' to train a new model, or 'test' to evaluate a model on the test set."
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/model_simple.h5',
        help="Path to save the trained model or load the model for testing."
    )
    args = parser.parse_args()

    if args.mode == 'train':
        run_training_pipeline(args.model_path)
    elif args.mode == 'test':
        run_evaluation_pipeline(args.model_path)


if __name__ == "__main__":
    main()