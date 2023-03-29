# %%
import torch
import pandas as pd
import os, sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %%
data_tain_path = './dataset/train_set.csv'
data_test_path = './dataset/test_set.csv'
batch_size = 32

# %%
data_train = pd.read_csv(data_tain_path)
data_test = pd.read_csv(data_test_path)

# %%
data_train

# %%
data_train_formatted = []
for pd_data_row in data_train.iloc:
    row = {'sentence': pd_data_row['Job_offer'], 'label': pd_data_row['Label']}
    data_train_formatted.append(row)

data_test_formatted = []
for pd_data_row in data_test.iloc:
    row = {'sentence': pd_data_row['Job_offer'], 'label': pd_data_row['Label']}
    data_test_formatted.append(row)

# %%
data_train_labels = sorted(list(set([r["label"] for r in data_train_formatted])))
data_test_labels = sorted(list(set([r["label"] for r in data_test_formatted])))

# %%
data_train_labels

# %%
data_test_labels

# %%
id_to_label = data_train_labels
label_to_id = {l:i for i,l in enumerate(id_to_label)}

# %%
class JDataset():
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
    def create_collate_fn(self):
        def collate_fn(batch):
            batch_formatted = {}
            batch_formatted['sentence'] = [sample['sentence'] for sample in batch]
            batch_formatted['label'] = [sample['label'] for sample in batch]
            return batch_formatted
        return collate_fn

# %%
dataset_train = JDataset(data_train_formatted)
dataset_test = JDataset(data_test_formatted)

# %%
from torch.utils.data import DataLoader

dataloader_train = DataLoader(
    dataset_train,
    batch_size=batch_size,
    collate_fn=dataset_train.create_collate_fn(),
    shuffle=True,
)

dataloader_dev = DataLoader(
    dataset_test,
    batch_size=batch_size,
    collate_fn=dataset_train.create_collate_fn(),
    shuffle=False,
)

# %%
import torch.optim as optim

loss_function = torch.nn.CrossEntropyLoss()

# %%
from code_files.models.transformer_classifier import TClassifier
from code_files.utils.Trainer_nec import Trainer_nec

# %%
model = TClassifier(
    loss_fn = loss_function,
    hparams = {
        'transformer_name':"xlm-roberta-base",
        'id_to_label': data_train_labels
    },
    fine_tune_transformer = True
)

# %%
def print_summary(model, short = False):
    """prints the summary for a model

    Args:
        model (any): The torch model
        short (bool, optional): If the print must be synthetic. Defaults to False.
    """
    if not short:
        print(model)
        print('----------------------')
    p = sum(p.numel() for p in model.parameters())
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ntp = p - tp
    print('parameters:', f'{p:,}')
    print('trainable parameters:', f'{tp:,}')
    print('non-trainable parameters:', f'{ntp:,}')

# %%
optimizer_pid = optim.SGD(model.parameters(), lr=0.0016, momentum=0.9)

# %%
print_summary(model, short = True)

# %%
for e in dataloader_dev:
    break

# %%
e

# %%
history = {}

# %%
trainer = Trainer_nec()

history = trainer.train(
    model, optimizer_pid, dataloader_train, dataloader_dev,
    epochs=100, device=device,
    save_best=True, 
    min_score=0.8,
    save_path_name=os.path.join('checkpoints/transformer_classifier/', f'transformer.pth'),
    saved_history=history
)

# %%



