import os
import sys
import datetime
import argparse

import torch
from torch.utils.data import DataLoader as StandardDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from transformers import get_linear_schedule_with_warmup
from torch.amp import GradScaler

from utils.hest_utils import load_gene_list, load_stpath_gene_dict
from utils.training_utils import (
    get_emb_dim,
    get_model,
    handle_graph_batch,
    handle_tensor_batch,
    unified_train_epoch,
    evaluate_epoch,
    print_model_params,
    save_checkpoint,
    load_checkpoint,
)
from utils.custom_dataset import PrecomputedEmbeddingDataset, HESTGraphDataset
from utils.other_utils import get_nested,seed_everything

import wandb
from ruamel.yaml import YAML


if __name__ == "__main__":
    print("[INFO_HISTOPRISM_TRAIN]Starting training scirpt...")
     # Set seed for reproducibility
    seed_everything(seed=2025) 

    # for better yaml logging
    yaml = YAML()
    yaml.preserve_quotes = True  # enables quote preservation

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Mutually exclusive group for config_path and log_dir
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config_path", "-f", type=str, help="Path to config file")
    group.add_argument("--log_dir", "-l", type=str, help="Logging directory")

    # Other args
    parser.add_argument("--continue_training", "-c", action="store_true",
                        help="Flag to continue training or not")

    args = parser.parse_args()
    continuous_training = args.continue_training
    log_dir = args.log_dir

    # If not continuing training, create a new log directory
    if not continuous_training:
        # Create a unique subfolder in training_log for each run
        run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("training_log", f"run_{run_time}")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, "training_run.txt")
        # Redirect stdout and stderr to the log file
        sys.stdout = open(log_path, "w")
        sys.stderr = sys.stdout

        # load the training configuration
        with open(args.config_path, "r") as f:
            config = yaml.load(f)

        # Save config to log_dir for reproducibility
        with open(os.path.join(log_dir, "config.yml"), "w") as f:
            yaml.dump(config, f)

    # If continuing training, use the existing log directory
    else:
        assert os.path.exists(
            log_dir
        ), f"[ERROR_HISTOPRISM_TRAIN] Log directory {log_dir} does not exist. Please check the path."
        # load the training configuration
        with open(os.path.join(log_dir, "config.yml"), "r") as f:
            config = yaml.load(f)
        log_path = os.path.join(log_dir, "training_run.txt")
        # Redirect stdout and stderr to the log file (append mode)
        sys.stdout = open(log_path, "a")
        sys.stderr = sys.stdout

    # Create directories for checkpoints and bad batches
    bad_batches_dir = os.path.join(log_dir, "bad_batches")
    os.makedirs(bad_batches_dir, exist_ok=True)

    # Initialize wandb
    # wandb project name
    wandb_project_name = config["wandb_project_name"]
    wandb.login()
    wandb.init(
        project=wandb_project_name,  # Change to your project name
        name=f"run_{wandb.util.generate_id()}",
        config=config,
    )
    # Save the run id for later
    with open(os.path.join(log_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb.run.id)

    # wandb when resuming (continuous_training)
    if continuous_training:
        with open(os.path.join(log_dir, "wandb_run_id.txt")) as f:
            wandb_run_id = f.read().strip()
        wandb.init(
            project=wandb_project_name,
            id=wandb_run_id,
            resume="allow",
            config=config,
        )
    else:
        wandb.init(
            project=wandb_project_name,
            name=f"run_{wandb.util.generate_id()}",
            config=config,
        )

    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO_HISTOPRISM_TRAIN] Using device: {device}")

    # Load gene list
    gene_list = load_gene_list(config)
    gene_dim = len(gene_list)
    wandb.config.update({"gene_dim": gene_dim})  # Log gene_dim to wandb

    emb_dim = get_emb_dim(config)
    wandb.config.update({"emb_dim": emb_dim})  # Log emb_dim to wandb

    if config["paths_config"]["gene_list_path"] =="data/gene_panel/stpath_genes.yaml":
        stpath_gene_dict = load_stpath_gene_dict()
    else:
        stpath_gene_dict = None

    # dataloader setup
    if config["model"].lower() =="histoprism": 
        print(f"[INFO_HISTOPRISM_TRAIN] Using graph dataloader for model {config['model']}")
        train_dataset = HESTGraphDataset(
            config,
            split="train",
            gene_list=gene_list,
            DEBUG=config["debug"],
            knn=get_nested(config=config, keys=["graph_config", "knn"], default=8),
            stpath_gene_dict=stpath_gene_dict,
        )
        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        val_dataset = HESTGraphDataset(
            config,
            split="val",
            gene_list=gene_list,
            DEBUG=config["debug"],
            knn=get_nested(config=config, keys=["graph_config", "knn"], default=8),
            stpath_gene_dict=stpath_gene_dict,
        )
        val_loader = GraphDataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        batch_handler = handle_graph_batch
    else:
        print(f"[INFO_HISTOPRISM_TRAIN] Using standard dataloader for model {config['model']}")
        train_dataset = PrecomputedEmbeddingDataset(
            config,
            split="train",
            gene_list=gene_list,
            DEBUG=config["debug"],
        )
        train_loader = StandardDataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        val_dataset = PrecomputedEmbeddingDataset(
            config,
            split="val",
            gene_list=gene_list,
            DEBUG=config["debug"],
        )
        val_loader = StandardDataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        batch_handler = handle_tensor_batch

    print(f"[INFO_HISTOPRISM_TRAIN] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset) if val_loader is not None else 'N/A'}")
    
    # instantiate an affine path object
    model = get_model(config, gene_dim, emb_dim, device)
    print(f"[INFO_HISTOPRISM_TRAIN] Model initialized")
    print_model_params(model)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # debug mode: set num_epochs to 200
    if config["debug"] == True:
        config["num_epochs"] = 5

    #  CREATE THE LEARNING RATE SCHEDULER ---
    # First, calculate the total number of training steps.
    num_epochs = config["num_epochs"]
    num_training_steps = num_epochs * len(train_loader)

    # Choose a warmup period. A common choice is 5-10% of total steps.
    warmup_steps = int(0.05 * num_training_steps)

    print(f"[INFO_HISTOPRISM_TRAIN] Total training steps: {num_training_steps}")
    print(f"[INFO_HISTOPRISM_TRAIN] Warmup steps: {warmup_steps}")

    # Create the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    # This is essential for preventing underflow with half-precision gradients.
    scaler = GradScaler("cuda")

    # load the model state for continuing training
    if continuous_training:
        # Load the model state if continuing training
        checkpoint_path = os.path.join(log_dir, "ckpt", "checkpoint_latest.pt")
        if os.path.exists(checkpoint_path):
            print(f"[INFO_HISTOPRISM_TRAIN] Loading model from {checkpoint_path}")
            start_epoch, best_val_loss, extra_info = load_checkpoint(
                path= checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            raise FileNotFoundError(
                f"[ERROR_HISTOPRISM_TRAIN] Checkpoint file {checkpoint_path} does not exist. Cannot continue training."
            )
    else:
        print(f"[INFO_HISTOPRISM_TRAIN] Starting training from scratch.")
        start_epoch = 0
        best_val_loss = float("inf")

    # Training prep
    best_epoch = start_epoch
    early_stop_counter = 0
    early_stop_patience = config[
        "early_stop_patience"
    ]  # Stop if no improvement for 20 epochs
    print(f"[INFO_HISTOPRISM_TRAIN] Early stop patience is {early_stop_patience} epochs")

    # _, topk_indices = get_top50var_genes(config, split="train")
    for epoch in range(start_epoch, config["num_epochs"]):
        # Training phase
        model.train()
        train_loss_epoch = 0
        train_batches = 0
        # TODO differnet dataloader for graph and non-graph models
        epoch_summary = unified_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            batch_handler_fn=batch_handler,
            epoch=epoch,
            bad_batch_dir = bad_batches_dir,
            save_bad_batches=True,  # Save bad batches for debugging
        )

        # Log average train loss for the epoch
        average_train_loss = epoch_summary.get("train/total_loss")

        wandb.log(
            {
                "train/epoch_loss": average_train_loss,
                "epoch": epoch,
            }
        )

        # Save checkpoint EVERY epochs
        ckpt_dir = os.path.join(log_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        # model, optimizer, scheduler, epoch, path, **kwargs
        checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_latest.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            path=checkpoint_path,
        )
        print(f"[INFO_HISTOPRISM_TRAIN] Saved checkpoint: {checkpoint_path}")

        # Validation phase (optional)
        if val_loader is not None:
            # Validation phase
            with torch.no_grad():
                val_epoch_summary = evaluate_epoch(
                    model=model,
                    dataloader=val_loader,
                    batch_handler_fn=batch_handler,
                    epoch=epoch,
                    description="val",
                )
                average_val_loss = val_epoch_summary.get("val/gene_loss", 0.0)

                wandb.log(
                    {
                        "val/epoch_loss": average_val_loss,
                        "epoch": epoch,
                    }
                )
        else:
            raise NotImplementedError("Val loader should not be empty!")
        # Save best model based on validation loss if val_loader exists,
        # otherwise use training loss as the criterion
        improved = False
        if val_loader is not None:
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_epoch = epoch + 1
                best_model_path = os.path.join(ckpt_dir, "best_model.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=best_epoch,
                    path=best_model_path,
                    best_val_loss=best_val_loss,
                )
                print(
                    f"[INFO_HISTOPRISM_TRAIN] Saved new best model at epoch {best_epoch} with val_loss {best_val_loss:.4f}"
                )
                improved = True
        else:
            # Use training loss as the criterion for saving the best model
            if average_train_loss < best_val_loss:
                best_val_loss = average_train_loss
                best_epoch = epoch + 1
                best_model_path = os.path.join(ckpt_dir, "best_model.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=best_epoch,
                    path=best_model_path,
                    best_val_loss=best_val_loss,
                )
                print(
                    f"[INFO_HISTOPRISM_TRAIN] Saved new best model at epoch {best_epoch} with train_loss {best_val_loss:.4f}"
                )
                wandb.save(best_model_path)
                improved = True

        wandb.log(
            {
                "best_epoch": best_epoch,
            }
        )

        # Early stopping logic
        if improved:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"[INFO_HISTOPRISM_TRAIN] No improvement for {early_stop_counter} epoch(s).")
            if early_stop_counter >= early_stop_patience:
                print(f"[INFO_HISTOPRISM_TRAIN] Early stopping triggered at epoch {epoch+1}.")
                break

        # Log average validation loss for the epoch
        if val_loader is not None:
            print(
                f"[INFO_HISTOPRISM_TRAIN] Epoch {epoch}, Training Loss - total:{average_train_loss:.4f},\nValidation Loss - total: {average_val_loss:.4f},  gene:{average_val_loss:.4f}"
            )
        else:
            print(
                f"[INFO_HISTOPRISM_TRAIN] Epoch {epoch}, Training Loss - total:{average_train_loss:.4f}"
            )

    wandb.finish()
    print("[INFO_HISTOPRISM_TRAIN] Training completed.")
