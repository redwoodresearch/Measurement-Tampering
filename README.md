# Benchmarks for Detecting Measurement Tampering

This repository contains the code for the [Benchmarks for Detecting Measurement Tampering](https://arxiv.org/abs/2308.15605).

## Abstract

> When training powerful AI systems to perform complex tasks, it may be challenging to provide training signals which are robust to optimization. One concern is \textit{measurement tampering}, where the AI system manipulates multiple measurements to create the illusion of good results instead of achieving the desired outcome. In this work, we build four new text-based datasets to evaluate measurement tampering detection techniques on large language models. Concretely, given sets of text inputs and measurements aimed at determining if some outcome occurred, as well as a base model able to accurately predict measurements, the goal is to determine if examples where all measurements indicate the outcome occurred actually had the outcome occur, or if this was caused by measurement tampering. We demonstrate techniques that outperform simple baselines on most datasets, but don't achieve maximum performance. We believe there is significant room for improvement for both techniques and datasets, and we are excited for future work tackling measurement tampering.

## How to use this repository

We recommend starting with `demo.ipynb` file which is a simple Jupyter notebook that demonstrates how a simple and effective technique which we call *probing for evidence of tampering* works.

### Reproducing the main results of the paper

- Run `pip install -r requirements.txt` to install the required packages.
- If you want to reproduce the results of the `<name>` dataset, run `python <name>/from_hf.py` to download the dataset from HuggingFace Datasets and convert it to a format that is compatible with the code in this repository. The different folders are the ones corresponding to the different datasets (except `measurement_tampering` which contains the code common to each method).
- Then, to run the full sweep of methods, run `python measurement_tampering/train_ray.py <model_name> --dataset_kind <name>`.
- This will save the results of the run, which can then be analyzed by opening `<name>/plotting_evals.py` as a notebook and running it.

### Generating the data yourself

You might want to regenerate the data, for example to examine the impact of variations of the dataset, as we've done in the appendix.

- **Diamond in a vault**: data is automatically generated when executing `python measurement_tampering/train_ray.py --dataset_kind diamonds`. Some global variables such as `OBFUSCATE` and `TAMPER_PROP` control generation parameters.
- **Generated stories**: execute `python money_making/prompt_gen.py` and `python money_making/get_training_data.py` (this will require setting an OPENAI_API_KEY environment variable first).
- **Text properties**: look at `text_properties/simplified_get_training_data.py` and `text_properties/data_initial.py`. Contact us for assistance about how to run these scripts.
- **Function correctness**: look at the `func_correct/pipeline.sh` file. You will need to merge the results of two runs of the pipeline if you want to use both human-generated and AI generated data.

## Glossary

We've changed the terminology during the process of writing the paper. Here is a table to translate between the different terms.

| Paper | Codebase |
--- | ---
| Fake positive | Tamper |
| Real positive | Nano |
| Real vs Fake AUROC | auroc_tn |
| Trusted | Clean |
| Untrusted | Dirty |
| Measurements | Sensors / Passes |
| Aggregated measurement | Junction sensor |
| Generated stories | Money making new |
| Generated stories easy | Money making |
| Ground truth | Correctness |
| PACE | Amnesic probing |
| Dirtiness probing | OOD probing |

## Known issues

- `python <name>/from_hf.py` only works for `diamonds` and `func_correct`.
- The folder names and seq_len defined in `run_config.json` files are not consitently used in the data generation pipelines.
