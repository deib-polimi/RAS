import pandas as pd
import numpy as np
import os
import argparse # Aggiunto per gestire gli argomenti da riga di comando
import random # Aggiunto per campionamento casuale
from pathlib import Path
import time

# Importiamo Application1 invece delle funzioni LQN
from application1 import Application1

# --- 1. CONFIGURAZIONE ---
# Cartella dove salveremo il dataset finale
OUTPUT_DIR = Path("./training_data")

# Parametri per la generazione del dataset
USER_START = 1
USER_END = 500
USER_STEP = 10
CORE_VALUES = np.arange(1, 30, 1) # Valori di core da testare

def generate_dataset_point(num_users: int, num_cores: int) -> dict:
    """
    Genera un singolo punto dati utilizzando Application1 per una data combinazione di utenti e core.
    """
    try:
        # Creiamo un'istanza di Application1
        app = Application1(sla=0.4,init_cores=1)  # SLA di 100ms come nel codice originale
        app.cores = num_cores
        
        # Calcoliamo il response time usando il modello Application1
        response_time = app.__computeRT__(req=num_users, t=0)
        
        return {
            "users": float(num_users),
            "cores": float(num_cores),
            "response_time": float(response_time),
        }
    except Exception as e:
        print(f"Errore nel calcolo per {num_users} utenti e {num_cores} core: {e}")
        raise


def create_sliding_windows(df: pd.DataFrame, window_size: int):
    """
    Converte un DataFrame di serie temporali in un dataset supervisionato
    per un modello a doppia torre, ora con 3 features: users, cores, response_time.
    """
    X_recurrent, X_feedforward, y_target = [], [], []

    # Per rendere il processo robusto, ordiniamo i dati per core e poi per utenti
    # Questo assicura che le 'sliding windows' abbiano un senso logico di progressione.
    df_sorted = df.sort_values(by=['cores', 'users']).reset_index(drop=True)

    for i in range(window_size, len(df_sorted)):
        # La finestra storica ora contiene 3 features: users, cores, response_time
        recurrent_window = df_sorted[['users', 'cores', 'response_time']].iloc[i - window_size:i].values
        X_recurrent.append(recurrent_window)

        # L'input "feed forward" contiene lo stato attuale (utenti e core)
        feedforward_input = df_sorted[['users', 'cores']].iloc[i].values
        X_feedforward.append(feedforward_input)

        # L'obiettivo è predire il response time
        target = df_sorted['response_time'].iloc[i]
        y_target.append(target)

    return np.array(X_recurrent), np.array(X_feedforward), np.array(y_target)


def main(mode: str, num_simulations: int):
    """
    Funzione principale che orchestra la generazione del dataset.
    Il parametro 'mode' determina se generare 'train' o 'test' data.
    """
    start_time = time.time()
    print(f"Inizio generazione dataset in modalità '{mode}' con {num_simulations} simulazioni...")

    # Impostiamo il suffisso per i file di output
    file_suffix = '_test' if mode == 'test' else ''

    raw_data = []
    
    # Ciclo di campionamento casuale per generare i dati
    print(f"Avvio di {num_simulations} simulazioni con parametri casuali...")
    for i in range(num_simulations):
        # Estraiamo una configurazione casuale
        users = random.randint(USER_START, USER_END)
        cores = random.choice(CORE_VALUES)

        if (i + 1) % 50 == 0:
            print(f"  ... simulazione {i + 1}/{num_simulations} (utenti: {users}, core: {cores})")

        try:
            data_point = generate_dataset_point(users, cores)
            raw_data.append(data_point)
        except Exception as e:
            print(f"ERRORE: Impossibile generare il punto dati per {users} utenti e {cores} core. Ignoro e continuo.")
            print(f"  Dettagli errore: {e}")

    if not raw_data:
        print("Nessun dato generato. Controllare i parametri e il modello Application1.")
    else:
        # Creiamo un DataFrame con i dati grezzi
        raw_df = pd.DataFrame(raw_data)
        print("\\n--- Dati Grezzi Raccolti ---")
        print(raw_df.head())
        print("---------------------------\\n")

        # Creiamo il dataset per il machine learning con una finestra storica di 5 step
        WINDOW_SIZE = 5
        X_rec, X_ff, y = create_sliding_windows(raw_df, window_size=WINDOW_SIZE)

        # Salviamo gli array finali con il suffisso corretto
        OUTPUT_DIR.mkdir(exist_ok=True)
        np.save(OUTPUT_DIR / f'X_recurrent{file_suffix}.npy', X_rec)
        np.save(OUTPUT_DIR / f'X_feedforward{file_suffix}.npy', X_ff)
        np.save(OUTPUT_DIR / f'y_target{file_suffix}.npy', y)
        
        end_time = time.time()
        print(f"Dataset '{mode}' creato con successo in {end_time - start_time:.2f} secondi.")
        print(f"Dati salvati in '{OUTPUT_DIR.absolute()}'")
        print(f"  - X_recurrent{file_suffix}.npy shape: {X_rec.shape}")
        print(f"  - X_feedforward{file_suffix}.npy shape: {X_ff.shape}")
        print(f"  - y_target{file_suffix}.npy shape: {y.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genera dati di training o di test per il modello di autoscaling.")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'test'], 
        default='train',
        help="Specifica se generare il dataset di 'train' o di 'test'. Default: 'train'."
    )
    parser.add_argument(
        '--num-simulations',
        type=int,
        default=1000,
        help="Numero di simulazioni casuali da eseguire per generare i dati."
    )
    args = parser.parse_args()
    
    main(args.mode, args.num_simulations) 