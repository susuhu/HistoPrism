"""
Preprocess WSIs into patch-level embeddings using pathology foundation models (PFM)
with corresponding barcode in a python dictionary, which can be matched to gene expression data.
---------
There are some weird graph data and model hanlding because we also experimented with
graph neural networks which did not yield better results. But we keep the code for
potential future use.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import h5py
import scanpy as sc

import torch
from torch_geometric.data import Data


from utils.hest_utils import (
    read_assets_from_h5,
    load_adata,
    create_spatial_knn_edges,
)
from utils.other_utils import get_nested
from utils.hest_utils import normalize_adata


def decode_barcode(b):
    # If it's a numpy array with one element, extract it
    if isinstance(b, np.ndarray):
        b = b.item()
    if isinstance(b, bytes):
        return b.decode()
    elif isinstance(b, np.bytes_):
        return b.astype(str)
    elif isinstance(b, np.str_):
        return str(b)
    elif isinstance(b, str) and b.startswith("[b'") and b.endswith("']"):
        # Handle stringified list of bytes: "[b'AAACAACGAATAGTTC-1']"
        return b[3:-2]
    else:
        return str(b)


def onehot_oncotree(code, all_codes):
    """
    Returns a one-hot torch tensor for a single code given the full list of all_codes.
    Example:
        all_codes = ['BRCA', 'LUAD', 'healthy', 'unknown']
        code = 'LUAD'
        onehot = onehot_oncotree_torch(all_codes, code)
        # onehot: tensor([0., 1., 0., 0.])
    """
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    idx = code_to_idx.get(code, -1)
    onehot = torch.zeros(len(all_codes), dtype=torch.float32)
    if idx >= 0:
        onehot[idx] = 1.0
    return onehot

    
class PrecomputedEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, config, split, gene_list, single_oncotree_code=None, stpath_gene_dict=None, DEBUG=False):
        split_path = os.path.join(get_nested(config,["paths_config","splits_path"]), f"{split}_split.csv")
        df = pd.read_csv(split_path, header=0)
        if single_oncotree_code:
            print(f"[INFO custom datset] Load only oncotree {single_oncotree_code}.")
            df = df[df["oncotree_code"] == single_oncotree_code]
        self.sample_ids = df["sample_id"].tolist()
        self.oncotree_code = df["oncotree_code"].tolist()
        self.embeddings_path = os.path.join("/mnt/cluster/datasets/HEST1k/", config["PFM_name"])
        if DEBUG:
            self.sample_ids = self.sample_ids[:5]
            # self.oncotree_code = self.oncotree_code[:2]
        self.gene_list = gene_list
        self.patches_path = get_nested(config,["paths_config","patches_path"])
        self.st_path = get_nested(config,["paths_config","st_path"])
        self.oncotree_list_path = os.path.join(get_nested(config,["paths_config","splits_path"]), "oncotree_code_list.txt")
        with open(self.oncotree_list_path, "r") as f:
            self.all_codes = [line.strip() for line in f]
        self.stpath_gene_dict = stpath_gene_dict

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        oncotree_code = self.oncotree_code[idx]
        oncotree_onehoted = onehot_oncotree(oncotree_code, self.all_codes).unsqueeze(0)

        # Path to the pre-processed file
        embedding_file = os.path.join(self.embeddings_path, f"{sample_id}_embeddings.pt")
        
        # Load the pre-computed data (fast I/O)
        patches_transformed_dict = torch.load(embedding_file, weights_only=True)
        
        # Load the gene data
        try:
            if self.stpath_gene_dict:
                gene_array = get_gene_array_efficient(os.path.join(self.st_path, f"{sample_id}.h5ad"),
                        patches_transformed_dict,
                        self.gene_list,
                        self.stpath_gene_dict,)
            else:
                gene_array = get_gene_array(
                    os.path.join(self.st_path, f"{sample_id}.h5ad"),
                    patches_transformed_dict,
                    self.gene_list,
                )
            gene_tensor = torch.tensor(gene_array, dtype=torch.float32)
        except:
            raise ValueError(f"sample {sample_id} has wrong barcodes")
        
        embeddings = torch.stack(list(patches_transformed_dict.values())).squeeze(1)

        # Load the asset from the HDF5 file
        asset_path = os.path.join(self.patches_path, f"{sample_id}.h5")
        if not os.path.exists(asset_path):
            print(asset_path)
            raise FileNotFoundError(
                f"[ERROR_dataset:]Asset file {asset_path} does not exist."
            )
        with h5py.File(asset_path, "r") as f:
            # Assuming the asset is stored in a dataset named 'asset'
            assets, _ = read_assets_from_h5(asset_path)
        coords = assets["coords"]

        return embeddings, coords, gene_tensor, oncotree_onehoted, sample_id


class HESTGraphDataset(PrecomputedEmbeddingDataset):
    def __init__(self, *args, knn=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.knn = knn

    def __getitem__(self, idx):
        # Use parent class to load base data
        embeddings, coords, gene_tensor, onco_onehot, sample_id = super().__getitem__(
            idx
        )

        # Create graph edges
        edge_index = create_spatial_knn_edges(coords, k=self.knn)
        edge_index = edge_index.detach().clone().long()

        # Wrap everything in a PyG Data object
        data = Data(
            x=embeddings,  # node features
            edge_index=edge_index,  # edges
            y=gene_tensor,  # target gene expression
        )
        data.sample_id = sample_id
        data.onco = onco_onehot  # additional global info

        return data


def get_gene_array(exp_path, embeddings_dict, gene_list):
    # Fix barcode decoding
    barcodes = []
    for b in embeddings_dict.keys():
        if isinstance(b, (list, np.ndarray)) and len(b) == 1:
            b = b[0]
        barcodes.append(decode_barcode(b))

    # Load AnnData
    gene_df = load_adata(
        exp_path, genes=None, barcodes=barcodes, normalize=True
    )  # Load all genes
    
    # Prepare output array: [n_barcodes, n_genes]
    n_barcodes = len(barcodes)
    n_genes = len(gene_list)

    # gene_array = np.zeros((n_barcodes, n_genes), dtype=np.float32)
    gene_array = np.full((n_barcodes, n_genes), np.nan, dtype=np.float32)

    # Map gene names to columns in AnnData
    adata_genes = list(gene_df.columns)
    gene_to_idx = {g: i for i, g in enumerate(adata_genes)}

    for j, gene in enumerate(gene_list):
        if gene in gene_to_idx:
            # Get the column index for the current gene from the loaded data
            source_idx = gene_to_idx[gene]
            # Copy the expression values for this gene into the correct column of our output array
            gene_array[:, j] = gene_df.iloc[:, source_idx].to_numpy(dtype=np.float32)
        # else: leave as np.nan
    return gene_array


def get_gene_array_efficient(exp_path, embeddings_dict, gene_list, stpath_gene_dict):
    """
    Maps gene expression data from original gene names to target symbol names,
    handling many-to-one mappings by summing expression values.
    """
    # --- 1. Barcode Loading and Alignment ---
    barcodes = []
    for b in embeddings_dict.keys():
        if isinstance(b, (list, np.ndarray)) and len(b) == 1:
            b = b[0]
        barcodes.append(decode_barcode(b))

    gene_df_original = load_adata(exp_path, genes=None, barcodes=barcodes)
    print(f"Original DataFrame loaded with shape: {gene_df_original.shape}")

    if not gene_df_original.index.equals(pd.Index(barcodes)):
        print("[INFO] Barcode order mismatch detected. Re-indexing DataFrame to align with embeddings.")
        gene_df_original = gene_df_original.reindex(barcodes, fill_value=0.0)

    # --- 2. Gene Name Mapping and Aggregation (The Fix) ---

    # Create a Series for mapping, which is needed for the groupby operation
    # This filters the mapping dict to only include genes present in our data
    mapper = pd.Series(stpath_gene_dict)
    mapper = mapper[mapper.index.isin(gene_df_original.columns)]

    # Group the original DataFrame by the NEW names (the values in the mapper)
    # The .T transposes the DataFrame so we can group by columns, then we transpose back.
    gene_df_mapped = gene_df_original.T.groupby(mapper).sum().T
    
    # --- 3. Final Gene Selection and Zero-Filling ---
    
    # Reindex the columns to match the final `gene_list`. This is now safe
    # because `gene_df_mapped` is guaranteed to have unique column names.
    final_df = gene_df_mapped.reindex(columns=gene_list, fill_value=0.0)

    # --- 4. Convert to NumPy Array ---
    gene_array = final_df.to_numpy(dtype=np.float32)
    
    # --- Logging and Validation ---
    found_genes_count = final_df.columns.isin(gene_df_mapped.columns).sum()
    missing_gene_count = len(gene_list) - found_genes_count
    
    print(f"[INFO] Finished processing. Final array shape: {gene_array.shape}")
    print(f"[INFO] Found data for {found_genes_count} out of {len(gene_list)} target genes.")
    if missing_gene_count > 0:
        print(f"[INFO] {missing_gene_count} target genes were not in the source data after mapping and were zero-filled.")
        
    return gene_array


def load_adata_gene_only(expr_path, genes=None, barcodes=None, normalize=True):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        # drop duplicated genes
        adata = adata[:, genes]
    adata.var_names_make_unique()
    if normalize:
        adata = normalize_adata(adata)
    return adata.to_df()

