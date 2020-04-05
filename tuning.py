import yaml
from task import Task
import optuna
import torch
import logging
from logging import getLogger
logger = getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_objective(hparams):

    def objective(trial: optuna.trial.Trial):
        encoder_num_layers = trial.suggest_int("encoder.num_layer", 1, 4)
        decoder_num_layers = trial.suggest_int("decoder.num_layer", 1, 4)
        transformer_d_model = trial.suggest_categorical('transformer.d_model', [64, 128, 256])
        hparams["basic"]["transformer"]["d_model"] = transformer_d_model
        hparams["decoder"]["num_layer"] = decoder_num_layers
        hparams["encoder"]["num_layer"] = encoder_num_layers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        task = Task(hparams, device)
        return task.train(trial)
    return objective

def main():
    with open("./optuna_args.yaml", "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)

    objection = generate_objective(hparams)
    study = optuna.create_study()
    optuna.logging.enable_default_handler()
    study.optimize(objection, n_trials=10)
    print(study.best_params)
    trial_df = study.trials_dataframe()
    trial_df.to_csv("transformer.csv")

if __name__ == '__main__':
    main()
