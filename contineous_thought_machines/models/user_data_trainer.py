import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision
from PIL import Image
import numpy as np
from contineous_thought_machines.models.ctm_components import HierarchicalCTM, EnhancedCTMConfig, DynamicEntropyPatcher
from training import autocast, GradScaler  # Assuming these are defined in training.py

# Configuration
USER_TRAIN_FOLDER = 'contineous_thought_machines/data/user_training/'
USER_EVAL_FOLDER = 'contineous_thought_machines/data/user_evaluation/'
BATCH_SIZE = 1000  # As per the hierarchical model's capability
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UserDataDataset(Dataset):
    def __init__(self, folder_path, patcher=None):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.patcher = patcher
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.txt', '.md']:
            with open(file_path, 'rb') as f:
                data = f.read()
        elif file_ext in ['.wav', '.mp3']:
            waveform, sr = torchaudio.load(file_path)
            data = waveform.numpy().tobytes()
        elif file_ext in ['.mp4', '.avi']:
            video = torchvision.io.read_video(file_path)
            data = video[0].numpy().tobytes()  # Video frames
        elif file_ext in ['.jpg', '.png']:
            img = Image.open(file_path)
            data = np.array(img).tobytes()
        else:
            with open(file_path, 'rb') as f:
                data = f.read()  # Treat as binary
        
        # Convert to torch tensor of bytes
        byte_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8))
        
        if self.patcher:
            # Use patcher to get patch indices
            _, patch_indices, _ = self.patcher(byte_tensor.unsqueeze(0))
            chunks = [byte_tensor[start:end+1] for start, end in patch_indices[0]]
        else:
            chunks = [byte_tensor]
        
        return chunks
    
def collate_fn(batch):
    all_chunks = []
    for item in batch:
        all_chunks.extend(item)
    
    if not all_chunks:
        return torch.tensor([])
    
    lengths = [len(c) for c in all_chunks]
    max_l = max(lengths)
    padded = torch.zeros(len(all_chunks), max_l, dtype=torch.uint8)
    for i, c in enumerate(all_chunks):
        padded[i, :len(c)] = c
    
    return padded

def train_user_data(model, dataloader, optimizer, epochs=NUM_EPOCHS):
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)  # Now inputs is (effective_B, max_len)
                loss = outputs['total_loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"Train Epoch {epoch+1}/{epochs} completed.")

def evaluate_user_data(model, dataloader):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = outputs['total_loss']
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Evaluation Average Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    config = EnhancedCTMConfig()  # Use default or load custom
    
    # Instantiate DynamicEntropyPatcher with parameters from config
    patcher = DynamicEntropyPatcher(
        embedding_dim=config.patch_embedding_dim,
        patch_cnn_channels=config.patch_encoder_cnn_channels,
        patching_mode=config.entropy_patcher_threshold_type,
        global_threshold=config.entropy_patcher_global_threshold,
        relative_threshold=config.entropy_patcher_relative_threshold,
        min_patch_size=config.entropy_patcher_min_patch_size,
        max_patch_size=config.entropy_patcher_max_patch_size,
        entropy_byte_vocab_size=config.entropy_model_byte_vocab_size,
        entropy_embedding_dim=config.entropy_model_embedding_dim,
        entropy_hidden_dim=config.entropy_model_hidden_dim,
        entropy_num_layers=config.entropy_model_num_layers,
        entropy_dropout=config.entropy_model_dropout
    )
    
    model = HierarchicalCTM(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train_dataset = UserDataDataset(USER_TRAIN_FOLDER, patcher)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    eval_dataset = UserDataDataset(USER_EVAL_FOLDER, patcher)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    train_user_data(model, train_dataloader, optimizer)
    evaluate_user_data(model, eval_dataloader)

    '''
    Evaluation Notes

    The evaluation function in user_data_trainer.py does not perform any training on the test data.
    It runs the model in evaluation mode (no gradients, no parameter updates) on data from the evaluation folder to compute the average loss,
    which measures how well the model's learned knowledge reconstructs or processes the test data compared to the training dataset's patterns.
    This helps benchmark the model's generalization to new, unseen data without modifying the model.
    '''

    '''
    Dataset Training Notes

    The UserDataDataset now uses DynamicEntropyPatcher for chunking both audio and text files into byte-based patches.
    This allows faster, interruptible training and generation for both modalities, enabling the model to sync text and voice generation while handling interruptions gracefully.
    Non-audio/text files are treated as single chunks.
    The collate_fn pads all chunks to the maximum length in the batch for efficient processing.
    '''