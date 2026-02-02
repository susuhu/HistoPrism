"""
HEST1k data ujtils
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import json
import yaml
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

import torch
import torch.nn.functional as F
import scanpy as sc

from utils.other_utils import get_nested


# from HEST1K repository
def save_hdf5(
    output_fpath, asset_dict, attr_dict=None, mode="a", auto_chunk=True, chunk_size=None
):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f:  # if key does not exist, create dataset
                data_type = val.dtype
                if data_type == np.object_:
                    data_type = h5py.string_dtype(encoding="utf-8")
                if auto_chunk:
                    chunks = True  # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(
                        key,
                        shape=data_shape,
                        chunks=chunks,
                        maxshape=(None,) + data_shape[1:],
                        dtype=data_type,
                    )
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")

            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0] :] = val

    return output_fpath


# from HEST1K repository
def read_assets_from_h5(h5_path, keys=None, skip_attrs=False, skip_assets=False):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, "r") as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            if not skip_assets:
                assets[key] = f[key][:]
            if not skip_attrs:
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
    return assets, attrs


# from HEST1K repository
def normalize_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    # print(adata.obs)
    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row["array_row"])
            col = int(df_row["array_col"])
            neighbors_index = adata.obs[
                (
                    (adata.obs["array_row"] >= row - 1)
                    & (adata.obs["array_row"] <= row + 1)
                )
                & (
                    (adata.obs["array_col"] >= col - 1)
                    & (adata.obs["array_col"] <= col + 1)
                )
            ].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)

            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg

    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)

    return filtered_adata


# from HEST1K repository
def load_adata(expr_path, genes=None, barcodes=None, normalize=True):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        # drop duplicated genes
        adata.var_names_make_unique()
        adata = adata[:, genes]
    if normalize:
        adata = normalize_adata(adata)
    return adata.to_df()


# my code
def build_knn_edge_index(img_emb, k=5):
    """
    DEPRECIATED
    feature similariy edges
    """
    # img_emb: [num_nodes, emb_dim]
    num_nodes = img_emb.size(0)
    normed = F.normalize(img_emb, dim=1)
    sim = torch.matmul(normed, normed.t())  # [batch, batch]
    _, idx = torch.topk(sim, k=k + 1, dim=-1)  # self included
    edge_index = []
    for i in range(num_nodes):
        for j in idx[i][1:]:  # skip self
            edge_index.append([i, j.item()])
    edge_index = (
        torch.tensor(edge_index, device=img_emb.device).t().contiguous()
    )  # [2, num_edges]
    return edge_index


def create_spatial_knn_edges(coords, k=8):
    """
    Spatial edges
    """
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)  # k+1 to include self

    # Remove self-loops (index 0)
    src_nodes = np.repeat(np.arange(coords.shape[0]), k)
    dst_nodes = indices[:, 1:].reshape(-1)

    edge_index = torch.tensor(np.stack([src_nodes, dst_nodes]), dtype=torch.long)
    return edge_index


# for inference
def get_top50var_genes(config, split="test", sample_id=None,stpath_gene_flag=False):
    """
    get top50var genes names, and gene ground truth from hest_bench, for metrics calculation
    """
    if split == "test":
        split_df = pd.read_csv(os.path.join(config["paths_config"]["splits_path"], "test_split.csv"))
        oncotree_code = split_df[split_df["sample_id"] == sample_id]["oncotree_code"].iloc[0]
        if stpath_gene_flag:
            top50_gene_path = os.path.join(
                "data/hest_bench/",
                oncotree_code,
                f"var_50genes_mapped.json",
            )
        else: 
            top50_gene_path = os.path.join(
                "data/hest_bench/",
                oncotree_code,
                f"var_50genes.json",
            )
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'test'.")

    with open(top50_gene_path, "r") as f:
        genes = json.load(f)
    top50_genes = genes["genes"]
    assert len(top50_genes) == 50, "[ERROR] Wrong number of genes!"

    with open(config["paths_config"]["gene_list_path"], "r") as f:
        gt_genes = yaml.safe_load(f)
    gt_genes = gt_genes["genes"]
    top50_indices_in_genes_gt = [
        gt_genes.index(gene) for gene in top50_genes if gene in gt_genes
    ]
    top50_genes_filtered = [gene for gene in top50_genes if gene in gt_genes]
    return top50_genes_filtered, top50_indices_in_genes_gt


def get_target_genes_idx(train_gene_list, target_genes):
    """
    get target genes names, and gene ground truth from hest_bench, for metrics calculation
    """
    # get indexes
    target_genes_indices_in_genes_gt = [
        train_gene_list.index(gene) for gene in target_genes if gene in train_gene_list
    ]
    target_genes_filtered = [gene for gene in target_genes if gene in train_gene_list]
    return target_genes_filtered, target_genes_indices_in_genes_gt


def get_top50_HVG_joint():
    """
    Get the joint of top50 HVG from hest bench dataset
    """
    data_path = "data/hest_bench/"
    cancer_types = os.listdir(data_path)
    all_genes = []
    for cancer_type in cancer_types:
        if cancer_type.startswith("."):
            continue
        file_path = os.path.join(data_path, cancer_type, "var_50genes.json")
        assert os.path.exists(file_path) is True
        with open(file_path, "r") as f:
            genes = json.load(f)
        genes = genes["genes"]
        all_genes.extend(genes)
    all_genes = list(set(all_genes))
    all_genes.sort()
    return all_genes


def load_gene_list(config):
    with open(get_nested(config,["paths_config","gene_list_path"])) as f:
        gene_list = yaml.safe_load(f)["genes"]
    return gene_list


def load_metadata(config_path):
    # Load the metadata
    config = yaml.safe_load(open(config_path))
    return config



def get_gt_genes_df(config_path,split="test"):
    """
    get a ground truth df of gene epxressions with sample ids and oncotree code for UMAP plot
    
    """
    config = load_metadata(config_path)
    gene_list = load_gene_list(config)
    # get ids
    splits_path = config["paths_config"]["splits_path"]
    split_csv_path = os.path.join(splits_path, split+"_split.csv")
    print("split from:", split_csv_path)

    split_df = pd.read_csv(split_csv_path)
    try:
        sample_ids = split_df["sample_id"].tolist()
    except:
        sample_ids = split_df["id"].tolist()

    all_df_list = []
    # Process each sample
    for sample_id in tqdm(sample_ids):
        exp_path = os.path.join(config["paths_config"]["st_path"], f"{sample_id}.h5ad")
        sample_adata_gene_df = load_adata(expr_path = exp_path, normalize=True)

        # Add missing columns with 0
        for gene in gene_list:
            if gene not in sample_adata_gene_df.columns:
                # Set it to zero, meaning there is no expression of this gene
                sample_adata_gene_df[gene] = 0
        sample_adata_gene_df_filtered = sample_adata_gene_df[gene_list]
        sample_adata_gene_df_filtered["sample_id"] = sample_id
        try:
            sample_adata_gene_df_filtered["oncotree_code"] = split_df.loc[split_df["sample_id"] == sample_id, "oncotree_code_filled"].values[0]
        except:
            sample_adata_gene_df_filtered["oncotree_code"] = split_df.loc[split_df["id"] == sample_id, "oncotree_code"].values[0]
        all_df_list.append(sample_adata_gene_df_filtered)

    all_df = pd.concat(all_df_list,ignore_index=True)
    return all_df



def load_stpath_gene_dict():
    with open("data/gene_panel/STPath_genes.json", "r") as f:
        gene_dict = json.load(f)
    return gene_dict


def rename_genes(df_ori,stpath_gene_dict):
    df_filtered = df_ori.loc[:, df_ori.columns.intersection(stpath_gene_dict.keys())]
    # Rename columns to symbols
    df_renamed = df_filtered.rename(columns=stpath_gene_dict)
    return df_renamed
