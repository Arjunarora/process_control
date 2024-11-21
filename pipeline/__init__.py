import mlflow
from mlflow.utils import mlflow_tags
import git
import pathlib
import logging

from config import FILE_DEPENDENCIES, RELEVANT_SECTIONS


# Adapted from https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/main.py
def already_ran(entrypoint: str, config: dict, mlflow_experiment_id=None, ignore_keys=None, ignore_git=False) -> [mlflow.tracking.fluent.Run, None]:
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    if ignore_keys is None:
        ignore_keys = [
            'normalization_ranges'  # Since this parameter is too long for MLflow to handle, it is not logged.
        ]

    # Get list of all runs from all experiments, if no experiment id is specified.
    list_runs = mlflow.MlflowClient().search_runs(experiment_ids=[_ for _ in range(10)] if not mlflow_experiment_id else [mlflow_experiment_id])

    # Get current local git commit
    repo = git.Repo('.')
    git_branch = repo.active_branch.name
    if not git_branch:
        git_branch = 'dev'
    current_commit = repo.commit(git_branch)

    for run_other in list_runs:
        run_id_other = run_other.info.run_id
        run_tags_other = run_other.data.tags

        # Ignore runs with different entrypoints
        entrypoint_other = run_tags_other.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None)
        if not entrypoint_other:
            entrypoint_other = run_tags_other.get('entrypoint')
        if entrypoint_other != entrypoint:
            # logging.debug(f'    Run {run_id_other}: Match failed because of different entrypoint: {entrypoint_other} != {entrypoint}.')
            continue

        # Ignore runs that have not finished
        if run_other.info.status != 'FINISHED':
            # logging.debug(f'    Run {run_id_other}: Match failed because run has status {run_other.info.status} and not FINISHED.')
            continue

        # Ignore runs with different parameters
        match_failed = False
        for section in config.keys():
            if section not in RELEVANT_SECTIONS.get(entrypoint):
                continue
            for key, val in config[section].items():

                # Skip ignored keys
                if key in ignore_keys:
                    continue
                if entrypoint == 'generate_artificial_data' and key in ['min_nr_experiments', 'max_nr_experiments']:
                    continue

                val_other = str(run_other.data.params.get(key))
                val = str(val)
                if val_other != val:
                    match_failed = True
                    # logging.debug(f'    Run {run_id_other}: Match failed because of different parameters {key}: {val} != {val_other}.')
                    break
            if match_failed:
                break
        if match_failed:
            continue

        # Ignore runs that have different git commits of relevant files
        if not ignore_git:
            previous_commit_hexsha = run_tags_other.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
            if current_commit.hexsha != previous_commit_hexsha:
                list_changed_files = []
                previous_commit = repo.commit(previous_commit_hexsha)
                diff = current_commit.diff(previous_commit)
                for d in diff:
                    b_path = pathlib.Path(d.b_path).absolute()
                    list_changed_files.append(b_path.name)

                # Account for changes in pipeline files and dependencies
                for file, list_dependant_stages in FILE_DEPENDENCIES.items():
                    if file in list_changed_files and entrypoint in list_dependant_stages:
                        match_failed = True
                        # logging.debug(f'    Run {run_id_other}: Match failed because of relevant code changes in file {file}.')
                        break
                if match_failed:
                    continue

        # Run could be matched -> No rerun necessary
        logging.info(f'Found existing run for {entrypoint} with specified parameters: {run_other.info.run_id}')
        return run_other

    logging.info(f'Couldn\'t find existing run for {entrypoint} with specified parameters.')
    return None
