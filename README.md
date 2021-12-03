# Debiasing Methods in Natural Language Understanding Make Bias More Accessible

_Accepted as a conference paper for EMNLP 2021_

[Paper](https://aclanthology.org/2021.emnlp-main.116/) | [Arxiv](https://arxiv.org/abs/2109.04095)

> **Abstract**: Model robustness to bias is often determined by the generalization 
> on carefully designed out-of-distribution datasets. 
> Recent debiasing methods in natural language understanding (NLU) improve performance 
> on such datasets by pressuring models into making unbiased predictions. 
> An underlying assumption behind such methods is that this also leads to the discovery 
> of more robust features in the model’s inner representations. 
> We propose a general probing-based framework that allows for post-hoc 
> interpretation of biases in language models, and use an information-theoretic 
> approach to measure the extractability of certain biases from the model's representations. 
> We experiment with several NLU datasets and known biases, and show that, counter-intuitively, 
> the more a language model is pushed towards a debiased regime, the more bias is actually encoded 
> in its inner representations. 

## Environment setup

First, clone this repository
```shell script
git clone https://github.com/technion-cs-nlp/bias-probing
```

### Anaconda/Miniconda

From the root folder of the project run
```
conda env create -f environment.yml
conda activate probing
```

### Other environments

Dependencies are listed in `environment.yml` for installing with `pip`,`venv`, etc,

### Tested Environment
All package versions and environment settings are documented in `environment.yml`.

Tested on `Python 3.7.4` with:
* **transformers** 4.5.1
* **datasets** 1.3.0
* **pytorch** 1.7.0

Compatibility tested with `Cuda v10.2`. Models were trained on NVIDIA GeForce RTX 2080 Ti.

## Code structure

### Framework

* `bias_probing/` contains models, data files, and implementation of the probing framework.
* `scripts/` is the entry point for running the experiments and training models.

## Reproducing the Results

_TBD_

## Citation

If you find this repository useful in your work, please cite our paper:

```
@inproceedings{mendelson-belinkov-2021-debiasing,
    title = "Debiasing Methods in Natural Language Understanding Make Bias More Accessible",
    author = "Mendelson, Michael  and
      Belinkov, Yonatan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.116",
    pages = "1545--1557",
    abstract = "Model robustness to bias is often determined by the generalization on carefully designed out-of-distribution datasets. Recent debiasing methods in natural language understanding (NLU) improve performance on such datasets by pressuring models into making unbiased predictions. An underlying assumption behind such methods is that this also leads to the discovery of more robust features in the model{'}s inner representations. We propose a general probing-based framework that allows for post-hoc interpretation of biases in language models, and use an information-theoretic approach to measure the extractability of certain biases from the model{'}s representations. We experiment with several NLU datasets and known biases, and show that, counter-intuitively, the more a language model is pushed towards a debiased regime, the more bias is actually encoded in its inner representations.",
}
```

## License

This work is open-sourced under the MIT license. For more information check out [LICENSE](LICENSE.md)


