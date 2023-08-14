# Measurement Tampering Detection

This repository contains the code for the Measurement Tampering Detection Benchmark.

We introduce four datasets to evaluate measurement tampering detection techniques, and we introduce techniques for this new task.

## How to use this repository

We recommend starting with `demo.ipynb` file which is a simple Jupyter notebook that demonstrates how a simple and effective technique which we call *probing for evidence of tampering* works.

### Reproducing the main results of the paper

- Run `pip install -r requirements.txt` to install the required packages.
- If you want to reproduce the results of the `<name>` dataset, run `python <name>/from_hf.py` to download the dataset from HuggingFace Datasets and convert it to a format that is compatible with the code in this repository. The different folders are the ones corresponding to the different datasets (except `measurement_tampering` which contains the code common to each method).
- Then, to run the full sweep of methods, run `python measurement_tampering/train_ray.py <model_name> --dataset_kind <name>`.
- This will save the results of the run, which can then be analyzed by opening `<name>/plotting_evals.py` as a notebook and running it.

### Generating the data yourself

You might want to regenerate the data, for example to examine the impact of variations of the dataset, as we've done in the appendix.

- **Diamond in a vault**: data is automatically generated when executing `python measurement_tampering/train_ray.py --dataset_kind <name>`. Some global variables such as `OBFUSCATE` and `TAMPER_PROP` control generation parameters.
- **Generated stories**: execute `python money_making/prompt_gen.py` and `python money_making/get_training_data.py` (this will require setting an OPENAI_API_KEY environment variable first).
- **Text properties**: look at `text_properties/simplified_get_training_data.py` and `text_properties/data_initial.py`. Generating this data requires running scripts multiple times with different parameters, contact the authors for assistance in regenerating this data.
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
| Generated stories | Money making |
| Ground truth | Correctness |
| PACE | Amnesic probing |
| Dirtiness probing | OOD probing |

## Known issues

- `python <name>/from_hf.py` only works for `diamonds` and `func_correct`.
- The folder names are hidden in the `run_config` files and are quite obscure, and those are not used in plotting evals.
- A figure and a link to the paper would be nice.