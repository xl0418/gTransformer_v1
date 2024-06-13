from pathlib import Path

def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 10,
        'lr': 0.001,
        'seq_len': 256,
        'embedding_size': 768,
        'model_folder': "tmodels",
        'model_name': "model_",
        'pre_load': "latest",
        "data_file": "data/GR3_ribosomal_maxg_expanded_cut_50.tsv",
        "vocab_file": "data/tokens_frequencies.json",
        "pre_tokenized": "data/vocab.json",
        "experiment_name": "runs/transformer",
        "vocab_size": 5006,
        "num_encoderblocks": 2,
        "expansion_factor": 4,
        "n_heads": 8
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_name']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_name']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])