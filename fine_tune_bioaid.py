import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import torch
from torch_geometric.typing import SparseTensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
import polars as pl
import pandas as pd
import numpy as np

from tqdm import tqdm
import wandb

from utils.BulkFormer import BulkFormer
from model.config import model_params

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

from IPython.display import display

import datetime as dt

device = 'cuda'

def load_model(file):
    graph_path = 'data/G_gtex.pt'
    weights_path = 'data/G_gtex_weight.pt'
    gene_emb_path = 'data/esm2_feature_concat.pt'

    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    weights = torch.load(weights_path, map_location='cpu', weights_only=False)
    graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)
    gene_emb = torch.load(gene_emb_path, map_location='cpu', weights_only=False)
    model_params['graph'] = graph
    model_params['gene_emb'] = gene_emb

    model = BulkFormer(**model_params).to(device)
    ckpt_model = torch.load(file, weights_only=False)
    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model

def main_gene_selection(X_df, gene_list):
    # fills columns (genes) that are in the gene list but not in our data
    # with -10
    # Returns df containing gene_list values (with -10 filling), columns that
    # were filled, and var, df indicating which columns are masked (to fill)

    to_fill_columns = list(set(gene_list) - set(X_df.columns))


    padding_df = pd.DataFrame(np.full((X_df.shape[0], len(to_fill_columns)), -10), 
                            columns=to_fill_columns, 
                            index=X_df.index)

    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]

    return X_df, to_fill_columns,var

def load_data(file, genes_only=True, return_df=False):
    # genes only indicates whether file contains genes only or other combined data
    # return df indicates whether to return torch tensor or output of main_gene
    # main_gene used for generating new embeddings i.e inference not training

    df = pd.read_parquet(file)
    bulkformer_gene_info = pd.read_csv('data/bulkformer_gene_info.csv')
    bulkformer_gene_info = bulkformer_gene_info[bulkformer_gene_info['ensg_id'] != '35991']
    bulkformer_gene_list = list(bulkformer_gene_info["gene_symbol"])

    if not genes_only:
        df = df.loc[:, "5S_rRNA":]

    input_df, to_fill_columns, var = main_gene_selection(X_df=df,gene_list=bulkformer_gene_list)
    if return_df:
        return input_df, to_fill_columns, var
    else:
        data = torch.tensor(input_df.values, dtype=torch.float32)
        return data

MASK_PROB = 0.15
MASK_VALUE = -10.0

# 2. Masking function
def mask_inputs(x, mask_prob=MASK_PROB):
    """
    x: (batch_size, 20010) tensor
    Returns:
        masked_x: input with some values masked (set to MASK_VALUE)
        labels: target values (same shape), with unmasked positions = MASK_VALUE (so we can ignore them in loss)
        mask: boolean mask of which positions were masked
    """
    # Do not mask already missing positions
    valid_mask = (x != MASK_VALUE)
    rand = torch.rand_like(x)
    mask = (rand < mask_prob) & valid_mask

    masked_x = x.clone()
    masked_x[mask] = MASK_VALUE

    labels = torch.full_like(x, MASK_VALUE)
    labels[mask] = x[mask]

    return masked_x, labels, mask

def get_validation_metrics(X, y):
    # TODO: add CV/not using test set later
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, penalty='l1', solver='saga') 

    # TODO: add feature scaling if needed
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) 
    y_prob = clf.predict_proba(X_test) [:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    return {
        "val/accuracy": accuracy,
        "val/roc_auc": roc_auc ,
        "val/precision": precision,
        "val/recall": recall,
        "val/f1": f1,
    }


def get_validation_metrics_from_embeddings(
        file="~/UCLThesis/data/BIOAID_UCL_Oxford_361_combined.parquet", 
        debug=True):

    if debug:
        file = "~/UCLThesis/data/BIOAID_UCL_Oxford_10_labelled_debug.parquet"

    embeddings = generate_embeddings(
        model,
        file=file,
        high_var_genes_only=False,
    )

    X = pd.DataFrame(embeddings.numpy(), columns=[f"col_{i}" for i in range(model.dim)])
    label_col = "micro_diagnosis"
    
    # process targets
    y = pd.read_parquet(file)[label_col]
    y = y.fillna(value="None")

    # In future, pd won't auto downcast. In our case, we manually ensure the type
    # so we can just ignore the warning.
    with pd.option_context('future.no_silent_downcasting', True):
        y.replace({"Bacterial": 0, "None": 0, "Bacterial & Viral": 1, "Viral": 1}, inplace=True)
    y = y.astype(int)

    return get_validation_metrics(X, y)

def train(model, dataloader, num_epochs=10, lr=1e-4, device="cuda", accumulation_steps=8, wandb_project="Thesis", debug=False):
    # 🟢 Initialize wandb
    wandb.init(project=wandb_project, config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "accumulation_steps": accumulation_steps,
        "batch_size": dataloader.batch_size,
        "model": model.__class__.__name__,
    },
    tags=["BulkFormer"],
    name=f"{model.__class__.__name__}_{dt.datetime.now()}"
    )

    model = model.to(device) 
    # model = torch.compile(model)  # not compatible with torch sparse
    optimizer = optim.Adam(model.parameters(), lr=lr, fused=True)
    loss_fn = nn.MSELoss(reduction="none")
    scaler = torch.amp.GradScaler()

    model.train()
    global_step = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        running_loss = 0.0
        step = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Batches, Epoch {epoch+1}", leave=False, position=1)

        for i, (batch,) in pbar:
            batch = batch.to(device)
            masked_x, labels, mask = mask_inputs(batch)
            masked_x = masked_x.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            with torch.amp.autocast(device_type=device):
                preds = model(masked_x)
                loss_matrix = loss_fn(preds, labels)
                masked_loss = loss_matrix[mask].mean() / accumulation_steps

            scaler.scale(masked_loss).backward()
            running_loss += masked_loss.item()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                avg_loss = running_loss
                wandb.log({
                    "train/loss": avg_loss, 
                    "train/step": global_step}
                    )
                pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

                running_loss = 0.0
                step += 1
                global_step += 1

            del batch, masked_x, labels, mask, preds, loss_matrix, masked_loss
            torch.cuda.empty_cache()

        # Final gradient step flush
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            wandb.log({"train/loss": running_loss, "train/step": global_step})
            global_step += 1

        # measure performance on the training set from labelled dataset as validation 

        val_metrics = get_validation_metrics_from_embeddings(
            file="~/UCLThesis/data/BIOAID_UCL_Oxford_361_combined.parquet",
            debug=debug) # remember debug = True uses different dataset and ignores file

        wandb.log(val_metrics)

    wandb.finish()

def extract_feature(model,
                    expr_array, 
                    high_var_gene_idx,
                    feature_type,
                    aggregate_type,
                    device,
                    batch_size,
                    return_expr_value = False,
                    esm2_emb = None,
                    valid_gene_idx = None):

    expr_tensor = torch.tensor(expr_array,dtype=torch.float32,device=device)
    mydataset = TensorDataset(expr_tensor)
    myloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False) 
    model.eval()

    all_emb_list = []
    all_expr_value_list = []


    with torch.no_grad():
        if feature_type == 'transcriptome_level':
            for (X,) in tqdm(myloader, total=len(myloader), leave=True, desc="\033[94mGenerating Embeddings\033[0m"):
                X = X.to(device)
                output, emb = model(X, [2])
                # output shape [batch, n_genes]
                # emb shape [batch, n_genes, proj_dim]

                all_expr_value_list.append(output.detach().cpu().numpy())
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:,high_var_gene_idx,:]
     
                if aggregate_type == 'max':
                    final_emb =np.max(emb_valid, axis=1)
                elif aggregate_type == 'mean':
                    final_emb =np.mean(emb_valid, axis=1)
                elif aggregate_type == 'median':
                    final_emb =np.median(emb_valid, axis=1)
                elif aggregate_type == 'all':
                    max_emb =np.max(emb_valid, axis=1)
                    mean_emb =np.mean(emb_valid, axis=1)
                    median_emb =np.median(emb_valid, axis=1)
                    final_emb = max_emb+mean_emb+median_emb

                all_emb_list.append(final_emb)
            result_emb = np.vstack(all_emb_list)
            result_emb = torch.tensor(result_emb,device='cpu',dtype=torch.float32)

        elif feature_type == 'gene_level':
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)
                output, emb = model(X, [2])
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:,valid_gene_idx,:]
                all_emb_list.append(emb_valid)
                all_expr_value_list.append(output.detach().cpu().numpy())
            all_emb = np.vstack(all_emb_list)
            all_emb_tensor = torch.tensor(all_emb,device='cpu',dtype=torch.float32)
            esm2_emb_selected = esm2_emb[valid_gene_idx]
            esm2_emb_expanded = esm2_emb_selected.unsqueeze(0).expand(all_emb_tensor.shape[0], -1, -1)  # [B, N, D]
            esm2_emb_expanded = esm2_emb_expanded.to('cpu')

            result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1)
    
    if return_expr_value:
        return np.vstack(all_expr_value_list)
    
    else:
        return result_emb

def generate_embeddings(
        model, 
        file="~/UCLThesis/data/BIOAID_UCL_Oxford_361_combined.parquet", 
        high_var_genes_only=False):
    input_df, _, var = load_data(file, genes_only=False, return_df=True)

    var.reset_index(inplace=True)
    valid_gene_idx = list(var[var['mask'] == 0].index)

    # decide whether to use high-variance genes only for embeddings
    if high_var_genes_only:
        high_var_gene_idx = torch.arange(20010)
    else:
        high_var_gene_idx = torch.load('data/high_var_gene_list.pt',weights_only=False)

    embeddings = extract_feature(
        model,
        expr_array= input_df.values,
        high_var_gene_idx=high_var_gene_idx,
        feature_type='transcriptome_level',
        aggregate_type='max',
        device=device, # TODO: may need to change to cpu if OOM
        batch_size=8,
        return_expr_value=False,
        esm2_emb=model_params['gene_emb'],
        valid_gene_idx=valid_gene_idx
    )

    return embeddings 


if __name__ == "__main__":
    WANDB = True
    DEBUG = False
    if not WANDB:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    torch.manual_seed(42)

    # TODO: Remove this if need higher precision
    torch.set_float32_matmul_precision('high')
    
    # for faster conv layers, no noticeable difference
    torch.backends.cudnn.benchmark = True

    training = True
    if training:
        print("Loading model...")
        model = load_model("model/Bulkformer_ckpt_epoch_29.pt")

        print("Loading data...")
        DEBUG_TRAINING_SAMPLES = 10 if DEBUG else 10000
        data = load_data("../UCLThesis/data/BIOAID_combined_tpm_PC0.001_log2_genesymbol_dedup.parquet")
        dataloader = DataLoader(TensorDataset(data[:DEBUG_TRAINING_SAMPLES]), batch_size=1, shuffle=True, num_workers=16)

        print("\033[94mStarting Training\033[0m")
        train(model, dataloader, num_epochs=5, debug=DEBUG)
        torch.save(model.state_dict(), f"fine-tuned-bulkformer-{dt.datetime.now()}.pt")

    else:
        # generate the embeddings
        model = load_model(file="fine-tuned-bulkformer.pt")
        embeddings = generate_embeddings(model)

        # about 1 minute batch size 16 for 1100, time seems around same with batch size 4
        embeddings = pd.DataFrame(embeddings.numpy(), columns=[f"col_{i}" for i in range(640)])
        display(embeddings)
        embeddings.to_parquet("../UCLThesis/data/BIOAID_361_embeddings_all_genes_fine_tuned.parquet", index=False)



