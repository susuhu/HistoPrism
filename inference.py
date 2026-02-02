import os
import sys
import yaml
import json
import pickle
import argparse
import gc

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import torch
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.data import Batch as GraphBatch  # Import with an alias

from utils.hest_utils import load_gene_list,load_stpath_gene_dict
from utils.training_utils import get_emb_dim, get_model, load_checkpoint
from utils.custom_dataset import PrecomputedEmbeddingDataset, HESTGraphDataset
from utils.other_utils import get_nested,seed_everything
from utils.hest_utils import get_top50var_genes, load_gene_list, get_target_genes_idx


def reduce_batch_dim(input_var):
    if input_var.ndim == 3 and input_var.shape[0] == 1:
        return input_var.squeeze(0)
    else:
        return input_var

def get_pcc(Inference_dir, test_split_path):
    # 1. Loop over sampling directory to get average_pcc across all sample ids
    assert os.path.exists(
        Inference_dir
    ), f"Inference directory does not exist: {Inference_dir}"
    average_pccs = {}
    for fname in os.listdir(Inference_dir):
        if fname.endswith(f"_gene_pcc.json"):
            if fname != f"all_samples_gene_pcc.json":
                sample_id = fname.replace(f"_gene_pcc.json", "")
                with open(os.path.join(Inference_dir, fname), "r") as f:
                    data = json.load(f)
                    if "average_pcc" in data:
                        average_pccs[sample_id] = data["average_pcc"]
                    else:
                        print(
                            f"Warning: 'average_pcc' not found in {fname}. Skipping this file."
                        )

    # 2. Load test split dataframe
    df = pd.read_csv(test_split_path)

    # 3. Aggregate gene expression data by sample id, group by organ
    df["sample_id"] = df["sample_id"].astype(str)
    df["average_pcc"] = df["sample_id"].map(average_pccs)

    # Report on any samples that did not have a corresponding PCC value
    unmatched_ids = df[df["average_pcc"].isnull()]
    if not unmatched_ids.empty:
        print(
            f"[WARNING_SAMPLING]: {len(unmatched_ids)} samples from the test split did not have a matching PCC file."
        )

    # Group by oncotree_code and aggregate average_pcc, skipping empty codes
    oncotree_pcc = (
        df[
            df["oncotree_code"].notnull()
            & (df["oncotree_code"].astype(str).str.strip() != "")
        ]
        .groupby("oncotree_code")["average_pcc"]
        .mean()
        .reset_index()
    )
    print("Average PCC per oncotree_code (non-empty):")
    print(oncotree_pcc)
    save_filename = os.path.join(Inference_dir, f"oncotree_average_pcc.csv")
    oncotree_pcc.to_csv(save_filename, index=False)
    print(f"\nSuccessfully saved results to {save_filename}")


if __name__ == "__main__":
    print("[INFO_INFERENCE]Starting sampling...")
    seed_everything(seed=2025)

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", "-l", type=str)  # training logging dir 
    # mapped hallmark and GO gene pickle files bwtween ESN codes and gene names
    parser.add_argument("--hallmark_gene_pickle", "-hg", type=str, default="data/Gene_Ontology/final_hallmark_genes_mapped.pkl")
    parser.add_argument("--GO_gene_pickle", "-g", type=str, default="data/Gene_Ontology/final_GO_genes_mapped.pkl")
    parser.add_argument("--save_pred_csv", "-s", action="store_true")  # THIS TAKES A LONG TIME SINCE THE .csv FILE IS HUGE
    args = parser.parse_args()
    log_dir = args.log_dir
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."
    
    hallmark_pickle_path = args.hallmark_gene_pickle
    GO_pickle_path = args.GO_gene_pickle

    # load config
    with open(os.path.join(log_dir, "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO_sampling] Using device: {device}")

    train_gene_list = load_gene_list(config)
    gene_dim = len(train_gene_list)

    # stpath_gene_flag for mapping between ESN codes and gene names
    if config["paths_config"]["gene_list_path"] =="data/gene_panel/stpath_genes.yaml":
        stpath_gene_dict = load_stpath_gene_dict()
        stpath_gene_flag = True
    else:
        stpath_gene_dict = None
        stpath_gene_flag = False
    print("[INFO] Using STPath dict", stpath_gene_flag)

    # create dataloader
    # I have graph conv option. That's why there is graph loader. But graoh conv does not yield better results
    if config["model"].lower() =="histoprism": 
        print(f"[INFO_INFERENCE] Using graph dataloader for model {config['model']}")
        test_dataset = HESTGraphDataset(
            config,
            split="test",
            gene_list=train_gene_list,
            single_oncotree_code=None,
            DEBUG=config["debug"],
            knn=get_nested(config=config, keys=["graph_config", "knn"], default=8),
            stpath_gene_dict=stpath_gene_dict,
        )
        test_loader = torch_geometric.loader.DataLoader(
            test_dataset, batch_size=1, shuffle=True
        )
    else:
        print(f"[INFO_INFERENCE] Using standard dataloader for model {config['model']}")
        test_dataset = PrecomputedEmbeddingDataset(
            config,
            split="test",
            gene_list=train_gene_list,
            single_oncotree_code=None,
            DEBUG=config["debug"],
            stpath_gene_dict=stpath_gene_dict,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    emb_dim = get_emb_dim(config)

    # get model
    model = get_model(config, gene_dim, emb_dim, device)
    model.to(device)
    print(f"[INFO_INFERENCE] Flow model initialized")

    checkpoint_path = os.path.join(log_dir, "ckpt", "best_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"[INFO_FLOW_ST] Loading model from {checkpoint_path}")
        start_epoch, best_val_loss, extra_info = load_checkpoint(
            path= checkpoint_path,
            model=model,
            # optimizer=optimizer,
            # scheduler=scheduler,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}. Please check the path.")
    model.eval()
    print(f"[INFO_INFERENCE] Loaded checkpoint: {checkpoint_path}")

    # log dirs
    inference_log_dir = os.path.join(
        log_dir, f"inference_epoch"+str(start_epoch)
    ) 
    os.makedirs(inference_log_dir, exist_ok=True)
    # Create a unique subfolder in sampling for each run
    log_path = os.path.join(inference_log_dir, "inference_run.txt")
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout


    inference_log_dir_all = os.path.join(inference_log_dir,"all_genes")
    inference_log_dir_top50 = os.path.join(inference_log_dir,"top50")
    inference_log_dir_hallmark = os.path.join(inference_log_dir,"hallmark")
    inference_log_dir_GO = os.path.join(inference_log_dir,"GO")
    os.makedirs(inference_log_dir_all, exist_ok=True)
    os.makedirs(inference_log_dir_top50, exist_ok=True)
    os.makedirs(inference_log_dir_hallmark, exist_ok=True)
    os.makedirs(inference_log_dir_GO, exist_ok=True)

    # test loss
    loss_fn = torch.nn.MSELoss()
    pred_df = pd.DataFrame(columns=["sample_id"] + train_gene_list)
    gene_pcc_across_samples = {gene: [] for gene in train_gene_list}
    rows = []
    # Inference/generation loop
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, GraphBatch):
                batch = batch.to(device)
                genes_gt = batch.y
                sample_id = batch.sample_id[0]  # it's a list

                # graph only models
                if config["model"].lower() =="histoprism":
                    outdict = model._get_predictions(emb = batch.x,
                                                          edges=batch.edge_index,
                                                          onco_onehot=batch.onco,
                                                          batch=batch.batch,) 
                    predicted_genes = outdict["gene_pred"]
                else:
                    raise NotImplementedError(
                        f"Model {config['model']} not implemented for graph batches."
                    )
            # non graph batch for MLP
            else: 
                # baseline MLP models not included in public repo
                # not graph batch, no edges
                raise NotImplementedError(
                    f"Model {config['model']} not implemented for non-graph batches."
                )

            # Compute MSE between generated and real gene expression
            test_loss = loss_fn(predicted_genes, genes_gt)

            predicted_genes_np = (
                predicted_genes.detach().cpu().numpy()
            )  # n_patches x n_genes
            genes_gt_np = genes_gt.detach().cpu().numpy()
            predicted_genes_np = reduce_batch_dim(predicted_genes_np)
            genes_gt_np = reduce_batch_dim(genes_gt_np)

            for n_patch in range(predicted_genes_np.shape[0]):
                rows.append([sample_id] + predicted_genes_np[n_patch].tolist())
            
            # ================================================
            # calculte all genes pcc
            all_pearson_dict = {}
            all_pearson_values = []

            all_sample_json_path = os.path.join(
                inference_log_dir_all, f"{sample_id}_gene_pcc.json"
            )
            if not os.path.isfile(all_sample_json_path):
                for idx, gene_name in enumerate(train_gene_list):
                    # PCC
                    try:
                        r, _ = pearsonr(predicted_genes_np[:, idx], genes_gt_np[:, idx])
                        r = float(r)
                        all_pearson_dict[gene_name] = r
                        gene_pcc_across_samples[gene_name].append(r)
                        all_pearson_values.append(r)
                    except Exception:
                        all_pearson_dict[gene_name] = None
                        gene_pcc_across_samples[gene_name].append(None)
                avg_pearson = (
                    float(np.nanmean([v for v in all_pearson_values if v is not None]))
                    if all_pearson_values
                    else float("nan")
                )

                all_sample_json_data = {
                    "pearson_corr": all_pearson_dict,
                    "average_pcc": avg_pearson,
                }
  
                with open(all_sample_json_path, "w") as f:
                    json.dump(all_sample_json_data, f, indent=2)

                try:
                    print(
                        f"[INFO_INFERENCE] sample_id={sample_id} | "
                        f"MSE={test_loss.item():.4f} | "
                        f"PCC={avg_pearson:.4f} | "
                        f"PCC saved: {all_sample_json_path}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"EXCEPTION in print block: {e}", flush=True)
            else:
                print(f"[INFO]{sample_id}'s all gene pcc has already been processed.")

                with open(all_sample_json_path, "r") as f:
                    all_sample_json_data = json.load(f)
            # ================================================
            all_sample_pcc_dict = all_sample_json_data["pearson_corr"]


            # ================================================
            # Only look up PCC for top 50 variable genes
            # Save per-sample Pearson to JSON
            top50_sample_json_path = os.path.join(
                inference_log_dir_top50, f"{sample_id}_gene_pcc.json"
            )
            if not os.path.isfile(top50_sample_json_path):
                topk_gene_names, topk_indices = get_top50var_genes(
                    config=config, split="test", sample_id=sample_id, stpath_gene_flag = stpath_gene_flag
                )
 
                # avoid ESN to gene names mismatch
                for idx, topk_idx in enumerate(topk_indices):
                    assert list(all_sample_pcc_dict.keys())[topk_idx] == topk_gene_names[idx]

                top50_pearson_dict = {list(all_sample_pcc_dict.keys())[i]: list(all_sample_pcc_dict.values())[i] for i in topk_indices}
                top50_avg_pearson = np.nanmean(list(top50_pearson_dict.values()))

                top50_sample_json_data = {
                    "pearson_corr": top50_pearson_dict,
                    "average_pcc":  top50_avg_pearson,
                }
       
                with open(top50_sample_json_path, "w") as f:
                    json.dump(top50_sample_json_data, f, indent=2)

                try:
                    print(
                        f"[INFO_INFERENCE] sample_id={sample_id} | "
                        f"MSE={test_loss.item():.4f} | "
                        f"PCC={top50_avg_pearson:.4f} | "
                        f"PCC saved: {top50_sample_json_path}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"EXCEPTION in print block: {e}", flush=True)
            else:
                print(f"[INFO]{sample_id}'s top50 pcc has already been processed.")
            # ================================================

            # ================================================
            # Only look up PCC for hallmark pathway genes
            with open(hallmark_pickle_path, 'rb') as f:
                hallmark_gene_path_dict = pickle.load(f)

            for pathway_name, pathway_gene_list in hallmark_gene_path_dict.items():
                inference_log_dir_hallmark_subpathway = os.path.join(inference_log_dir_hallmark, pathway_name)
                os.makedirs(inference_log_dir_hallmark_subpathway,exist_ok=True)
                hallmark_sample_json_path = os.path.join(
                    inference_log_dir_hallmark_subpathway, f"{sample_id}_gene_pcc.json"
                )
                if not os.path.isfile(hallmark_sample_json_path):
                    gene_names, gene_idxes = get_target_genes_idx(train_gene_list, pathway_gene_list)
                    # avoid ESN to gene names mismatch
                    for idx, gene_idx in enumerate(gene_idxes):
                        assert list(all_sample_pcc_dict.keys())[gene_idx] == gene_names[idx]

                    hallmark_pearson_dict = {list(all_sample_pcc_dict.keys())[i]: list(all_sample_pcc_dict.values())[i] for i in gene_idxes}
                    hallmark_avg_pearson = np.nanmean(list(hallmark_pearson_dict.values()))

                    hallmark_sample_json_data = {
                        "pearson_corr": hallmark_pearson_dict,
                        "average_pcc":  hallmark_avg_pearson,
                    }
        
                    with open(hallmark_sample_json_path, "w") as f:
                        json.dump(hallmark_sample_json_data, f, indent=2)

                    try:
                        print(
                            f"[INFO_INFERENCE] sample_id={sample_id} | "
                            f"{pathway_name} |"
                            f"PCC={hallmark_avg_pearson:.4f} | "
                            f"PCC saved: {hallmark_sample_json_path}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"EXCEPTION in print block: {e}", flush=True)
                else:
                    print(f"[INFO]{sample_id}'s hallmark pcc has already been processed.")
            # ================================================


            # ================================================
            # Only look up PCC for GO pathway genes
            with open(GO_pickle_path, 'rb') as f:
                GO_gene_path_dict = pickle.load(f)

            for pathway_name, pathway_gene_list in GO_gene_path_dict.items():
                inference_log_dir_GO_subpathway = os.path.join(inference_log_dir_GO, pathway_name)
                os.makedirs(inference_log_dir_GO_subpathway,exist_ok=True)
                GO_sample_json_path = os.path.join(
                    inference_log_dir_GO_subpathway, f"{sample_id}_gene_pcc.json"
                )
                if not os.path.isfile(GO_sample_json_path):
                    gene_names, gene_idxes  = get_target_genes_idx(train_gene_list, pathway_gene_list)
                    # avoid ESN to gene names mismatch
                    for idx, gene_idx in enumerate(gene_idxes):
                        assert list(all_sample_pcc_dict.keys())[gene_idx] == gene_names[idx]

                    GO_pearson_dict = {list(all_sample_pcc_dict.keys())[i]: list(all_sample_pcc_dict.values())[i] for i in gene_idxes}
                    GO_avg_pearson = np.nanmean(list(GO_pearson_dict.values()))

                    GO_sample_json_data = {
                        "pearson_corr": GO_pearson_dict,
                        "average_pcc":  GO_avg_pearson,
                    }
        
                    with open(GO_sample_json_path, "w") as f:
                        json.dump(GO_sample_json_data, f, indent=2)

                    try:
                        print(
                            f"[INFO_INFERENCE] sample_id={sample_id} | "
                            f"{pathway_name} | "
                            f"PCC={GO_avg_pearson:.4f} | "
                            f"PCC saved: {GO_sample_json_path}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"EXCEPTION in print block: {e}", flush=True)
                else:
                    print(f"[INFO]{sample_id}'s GO pcc has already been processed.")
            # ================================================

            del predicted_genes_np
            del genes_gt_np
            gc.collect()

    # per oncotree_code average pcc on top50 HVG
    split_path = os.path.join(config["paths_config"]["splits_path"], "test_split.csv")
    get_pcc(inference_log_dir_top50, split_path)
    
    # THIS TAKES LONG TIME SINCE IT'S HUGE
    if args.save_pred_csv:
        # Save all predicted genes to CSV
        print(f"[INFO_INFERENCE] saving Predicted genes csv...This takes a long time.")
        pred_df = pd.DataFrame(rows, columns=pred_df.columns)
        pred_df.to_csv(os.path.join(inference_log_dir, "predicted_genes.csv"), index=False)
        print(f"[INFO_INFERENCE] Predicted genes saved to: predicted_genes.csv")

    print("[INFO_INFERENCE] Inferencecomplete.")


