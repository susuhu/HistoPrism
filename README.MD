
## HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction

HistoPrism is an efficient transformer architecture that predicts spatial gene expression profiles directly from H&E histology slides. It addresses key limitations in prior work regarding generalizability and biological evaluation.

- **Unified Pan-Cancer Architecture**: Unlike prior per-cancer approaches or computationally heavy generative baselines, we introduce a unified, *lightweight model* that generalizes across diverse tissue types. This efficiency democratizes access, enabling resource-constrained institutes to fine-tune on small datasets and scale inference without massive infrastructure.

- **Gene Pathway Coherence (GPC) Benchmark**: We move beyond standard High-Variance Gene (HVG) metrics, which neglect biologically critical low-variance genes due to skewed distributions. We propose GPC, a novel framework that evaluates performance on functional biological pathways, demonstrating that our model recovers coherent, clinically relevant transcriptomic patterns across the full gene spectrum.

![architecture](figures/HistoPrism_no_PE.png)


## Performance

HistoPrism achieves *competitive performance on standard High-Variance Gene (HVG) metrics* while demonstrating superior capability in recovering biologically meaningful signals. As shown below, our model *outperforms baselines* on the **Gene Pathway Coherence (GPC)** benchmark, successfully recovering coherent transcriptomic patterns across both GO terms and Hallmark pathways.

<p align="center">
  <img src="figures/GO_pcc_comparison_scatter_two_split.png" width="45%" />
  <img src="figures/hallmark_pcc_comparison_scatter_two_split.png" width="45%" /> 
</p>

In addition to biological accuracy, HistoPrism is designed for **computational efficiency**. It delivers state-of-the-art results with significantly lower resource requirements than generative baselines, enabling resource-constrained private institutes to fine-tune on a small ST dataset and scale inference to larger cohorts.

<p align="center">
  <img src="figures/HistoPrism_model_efficiency_2-8.png" width="90%" />
</p>



## 💡 Optimization Tip
While we trained HistoPrism on a comprehensive 38k gene panel to ensure a fair benchmark in the paper, we recommend training on a focused target gene panel for practical applications. Narrowing the gene set concentrates the model's capacity on your specific biological targets, often yielding sharper predictions.

## Quick Start
### Data Preprocessing

- Download HEST1k from huggingface https://huggingface.co/datasets/MahmoodLab/hest. to `data/HEST1K` or [PATH/TO/YOUR/HEST1K]
- Please follow respective pathology foundation model (PFM) repositories for preprocessing. For example, preprocessed data with PFM [uni](https://github.com/mahmoodlab/UNI) is in `data/sample_processed_HEST1K/uni` with names `[sample_id]_embeddings.pt` which contains barcodes and correspongding uni features. This path is referred in `utils/custom_dataset.py` for dataloading.
- Also download hest_bench https://huggingface.co/datasets/MahmoodLab/hest-bench and put it in `data/hest_bench` [PATH/TO/YOUR/hest_bench] for fixed held-out test set and HVG-50 genes list
- Data split is merged from hest-bench and provided in `data/split_0` and `data/split_1`.


Following code could be used to preprocecss HEST1k dataset.

```
python3 utils/pfm_preprocesing.py
```

Folder structure:
```text
.
├── configs
│   └── training_config.yml
├── checkpoints_paper # pretrained weights from the paper
│   ├── split_0_paper
│   │   ├── ckpt # pretrained_model_weights
│   │   └── config.yml
│   └── split_1_paper
│       ├── ckpt
│       └── config.yml
├── data
│   ├── Gene_Ontology # GO and Hallmark pathways
│   ├── gene_panel
│   ├── hest_bench # downloaded hest_bench
│   ├── HEST1K # downloaded HEST1K raw data
│   ├── sample_processed_HEST1K # preprocessed WSIs with PFMs
│   ├── split_0
│   └── split_1
├── models
│   └── HistoPrIsm.py
├── utils
│   ├── custom_dataset.py
│   ├── hest_utils.py
│   ├── other_utils.py
│   ├── pfm_preprocesing.py
│   └── training_utils.py
├── inference.py
└── train.py
```

### Training

```
python3 train.py -f configs/training_config.yml
```

### Inference
All gene inference, HVG-50, hallmark pathways and GO pathways Pearson Corrrelation Coefficients (PCC)

```
python3 inference.py -l training_log/[PATH/TO/YOUR/TRAINING/LOG/DIR]
```

### Checkpoints from the paper
Pretrained model weights in the paper can be downloaded from [hugginface](https://huggingface.co/HuSusu/HistoPrism). You might need to rename it to best_model.pt if you run the inference code directly.

Direct inference can be done via:

```
python3 inference.py -l checkpoints_paper/split_0_paper
```


## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. For full details, see the [LICENSE](LICENSE) file.

## Citation

If you find **HistoPrism** or our **Gene Pathway Coherence (GPC)** benchmark useful in your research, please consider citing our paper:

```bibtex
@inproceedings{Hu2026HistoPrism,
  title={{HistoPrism}: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction},
  author={Hu, Susu and Zeng, Qinghe and Bhasker, Nithya and Kather, Jakob Nikolas and Speidel, Stefanie},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2601.21560}
}
