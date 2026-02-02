"""
HistoPrism model.
-----------------
There are some weird graph data and model hanlding because we also experimented with
graph neural networks which did not yield better results. But we keep the code for
potential future use.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.other_utils import get_nested
from torch_geometric.nn import GCNConv, GATConv, GINConv


class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=4, dropout_rate=0.1):
        super().__init__()
        # PyTorch's built-in MultiheadAttention is perfect for this.
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, # The dimension of the final output
            kdim=context_dim,    # The dimension of the keys (from context)
            vdim=context_dim,    # The dimension of the values (from context)
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True     # This is crucial for our data shapes
        )
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(self, query, context):
        """
        Args:
            query (torch.Tensor): The tensor that is "asking" the questions.
                                  Shape: [batch_size, num_queries, query_dim]
            context (torch.Tensor): The tensor providing the information.
                                    Shape: [batch_size, num_context_items, context_dim]
        
        Returns:
            torch.Tensor: The attended query tensor.
                          Shape: [batch_size, num_queries, query_dim]
        """
        # The MHA layer returns the attention output and the attention weights.
        # We only need the output.
        attn_output, _ = self.attention(
            query=query,
            key=context,
            value=context
        )
        # Add & Norm is a standard pattern in transformers
        return self.norm(query + attn_output)


class GraphConvLayer(nn.Module):
    def __init__(self, emb_dim, conv_type="gcn"):
        super().__init__()
        self.emb_dim = emb_dim
        if conv_type == "gcn":
            # Use GCNConv for image embeddings
            self.gconv = GCNConv(emb_dim, emb_dim)
        elif conv_type == "gat":
            self.gconv = GATConv(emb_dim, emb_dim)
        elif conv_type == "gin":
            self.gconv = GINConv(nn.Linear(emb_dim, emb_dim))
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}. Use 'gcn', 'gin', or 'gat'.")

    def forward(self, input, edges):
        # GCNConv expects [num_nodes, emb_dim]
        out = self.gconv(input, edges)  # [num_nodes, out_dim]
        return out


class HistoPrism(nn.Module):
    def __init__(self, config, gene_input_dim, img_emb_dim): # Removed time_dim and flow_path
        super().__init__()

        # --- Core Architectural Components (largely unchanged) ---
        self.img_emb_dim = img_emb_dim
        self.hidden_dim = config.get("hidden_dim", 256)
        self.n_conv_layers = config["graph_config"]["n_conv_layers"]
        self.conv_type = get_nested(config, ["graph_config", "conv_type"], "gat")
        self.n_heads = config.get("transformer_heads", 2)
        self.n_layers = config.get("transformer_layers", 2)
        self.gene_loss_weight = get_nested(config, ["loss_config", "gene_loss_weight"], 1.0)
        self.rnd_predictor_loss_weight = get_nested(config, ["loss_config", "rnd_predictor_loss_weight"], 0.01)
        self.exploration_reward_weight = get_nested(config, ["loss_config", "exploration_reward_weight"], 0.01)
        self.gene_emb_dim = get_nested(config,["gene_emb_dim"],False)
        self.gene_input_dim = gene_input_dim
        self.dropout_rate = config.get("dropout_rate", 0.1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.onco_onehot_dim = config.get("onco_onehot_dim", False)
        self.conditioning_strategy = config.get("conditioning_strategy",None).lower()
        self.cross_attn_heads = config.get("cross_attn_heads",2)

        # --- Graph Convolution Layers ---
        self.img_conv = GraphConvLayer(emb_dim=self.img_emb_dim, conv_type=self.conv_type)
        self.img_norm = nn.LayerNorm(self.img_emb_dim)

        self.input_proj = nn.Linear(self.img_emb_dim, self.hidden_dim)

        # --- Conditioning Module ---
        self.conditioner = None
        if self.onco_onehot_dim > 0:
            if self.conditioning_strategy == "cross_attention":
                # For cross-attention, we need to embed the onco_onehot to match img_emb_dim
                self.onco_embedder = nn.Linear(self.onco_onehot_dim, self.img_emb_dim)
                self.conditioner = CrossAttentionLayer(
                    query_dim=self.img_emb_dim, # The image embedding is the query
                    context_dim=self.img_emb_dim, # The onco embedding is the context
                    dropout_rate=self.dropout_rate,
                    n_heads=self.cross_attn_heads
                )
            else:
                raise ValueError(f"[WARN_MODEL]NO cross attention conditioning!")
        
        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=self.n_heads, batch_first=True, dropout=self.dropout_rate
            )  # [batch, seq, hidden_dim]
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # --- Single Regression Head for Gene Prediction ---
        self.gene_regression_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.gene_input_dim),
        )

    def _calculate_gene_loss(self, predicted_gene, target_gene):
        """ The only loss function needed for this model's primary task. """
        mask = ~torch.isnan(target_gene)
        if not mask.any():
            return predicted_gene.sum() * 0.0
        return F.mse_loss(predicted_gene[mask], target_gene[mask])

    def _get_predictions(self, emb, edges, onco_onehot, batch):
        """
        A simplified prediction function. It no longer takes x_t or t.
        It processes the image features to produce a final latent representation.
        """
        # --- Graph Convolution on Image Features ---
        emb_spatial = emb
        for _ in range(self.n_conv_layers):
            emb_res = emb_spatial
            h_conv = self.img_conv(emb_spatial, edges)
            h_activated = F.relu(h_conv)
            h_dropped = self.dropout(h_activated)
            emb_spatial = self.img_norm(h_dropped + emb_res)
        
        # Conditioning for global information
        if self.conditioner is not None:
            if self.conditioning_strategy == "cross_attention":
                # The logic here is the same, but the input `query` is now `emb_spatial`.
                onco_embedding = self.onco_embedder(onco_onehot)  # [1, D] match to image embedding dim
                onco_embedding = onco_embedding.unsqueeze(0)  # [1, 1, D]
                query = emb_spatial.unsqueeze(0) #[1,n_patch, D]
                conditioned_emb = self.conditioner(query=query, context=onco_embedding) # key and value are both context
                conditioned_emb=conditioned_emb.squeeze(0) # remove the batch dim again
        else:
            # If no conditioning, just use the spatial embeddings
            conditioned_emb = emb_spatial

        # --- Prepare for Transformer ---
        # The input is just the conditioned image features.
        h = self.input_proj(conditioned_emb)
        
        h_bottleneck = self.transformer(h.unsqueeze(0)).squeeze(0)

        # --- Return latent features and the final gene prediction ---
        predicted_genes = self.gene_regression_head(h_bottleneck)
        
        output_dict = {
            "gene_pred": predicted_genes,
            "latent_features": h_bottleneck # For RND loss
        }
        return output_dict

    def forward(self, x_1, emb, edges, onco_onehot=None, batch=None):
        """
        The simplified training forward pass.
        `x_1` is the ground truth target.
        """
        # --- Get Model Predictions ---
        model_predictions = self._get_predictions(
            emb=emb, edges=edges, onco_onehot=onco_onehot, batch=batch
        )

        # --- Calculate Primary Loss (Gene Loss Only) ---
        gene_loss = self._calculate_gene_loss(model_predictions["gene_pred"], x_1)
        total_loss = self.gene_loss_weight * gene_loss # Assumes gene_loss_weight is in config
        loss_dict_for_logging = {"gene_loss": gene_loss.item()}
        
        return total_loss, loss_dict_for_logging
    
