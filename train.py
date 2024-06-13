from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import pandas as pd
from config import get_config, get_weights_file_path, latest_weights_file_path
import json
from BPEgenome import genomeBPE
from datasets import grDataset
from gTransformer import Transformer, Embedding, PositionEmbedding, MultiHeadAttention, EncoderBlock, TransformerEncoder

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def get_ds(config):
    sequence_data = pd.read_csv(config['data_file'], sep='\t')
    training_set = pd.read_csv(config['training_set_file'], sep=',', header=None)

    all_IDs = sequence_data['genomeid'].unique()
    train_list_IDs = training_set[0].values.tolist()
    val_list_IDs = [x for x in all_IDs if x not in train_list_IDs]

    train_dataset = grDataset(train_list_IDs, sequence_data, config['seq_len'])
    val_dataset = grDataset(val_list_IDs, sequence_data, config['seq_len'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # check if the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_ds(config)
    model = Transformer(config['seq_len'], config["vocab_size"], config['embedding_size'], config['num_encoderblocks'],
                        config['expansion_factor'], config['n_heads'])

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['pre_load']
    model_name = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload)

    if model_name is not None:
        print(f"Loading weights from {model_name}")
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
    else:
        print("No weights file found. Training from scratch.")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader,
                              desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            seq = batch[0].to(device)
            gr = batch[1].to(device)

            optimizer.zero_grad()
            output = model(seq)
            loss = loss_fn(output.squeeze(), gr)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (seq, gr) in enumerate(val_dataloader):
                seq = seq.to(device)
                gr = gr.to(device)

                output = model(seq)
                loss = loss_fn(output.squeeze(), gr)
                total_loss += loss.item()

            avg_loss = total_loss / len(val_dataloader)
            print(f"Epoch: {epoch}, Validation Loss: {avg_loss}")
            writer.add_scalar('Loss/val', avg_loss, epoch)

        # Save the model
        model_name = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_name)


if __name__ == '__main__':
    config = get_config()
    k_clade = 10
    kk = 1
    config['training_set_file'] = f'data/GR3subtrees/Accession_{k_clade}_clades_train_{kk}.csv'

    # train_dataloader, val_dataloader = get_ds(config)
    train_model(config)

