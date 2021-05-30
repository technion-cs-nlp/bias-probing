import wandb
import pandas as pd
import logging
logger = logging.getLogger('export')
logging.basicConfig()
api = wandb.Api()

"""
These can be replaced, but make sure you also correct the names in `probing.py` and `training.py` scripts.
"""
WANDB_USERNAME = '<ANONYMIZED>'
MODEL_TRAINING_PROJECT_NAME = 'bias-probing'
ONLINE_CODE_PROJECT_NAME = 'online-code'


def _dump_runs_from(path: str, output_file: str):
    logger.info(f'Export from wandb.ai/{path} to {output_file}')
    runs = api.runs(path)
    summary_list = []
    config_list = []
    name_list = []
    for run in runs:
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)
    all_df.to_csv(output_file)
    logger.info('Success')


_dump_runs_from(f'{WANDB_USERNAME}/{MODEL_TRAINING_PROJECT_NAME}', output_file='results/results_debiasing.csv')
_dump_runs_from(f'{WANDB_USERNAME}/{ONLINE_CODE_PROJECT_NAME}', output_file='results/results_online_code.csv')
