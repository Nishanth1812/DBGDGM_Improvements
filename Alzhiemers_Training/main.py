import argparse
import logging
import sys
from pathlib import Path

import torch

from src.dataset import load_dataset
from src.inference import inference
from src.model import Model
from src.train import train


def main(args):
    log_format = '%(asctime)s [%(levelname)s] %(message)s'

    # StreamHandler → sys.stdout so logs appear live in the terminal.
    # force=True resets any handlers a previously imported module may have set.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))

    file_handler = logging.FileHandler(f'fMRI_{args.dataset}_{args.trial}.log', mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[stream_handler, file_handler],
        force=True,
    )


    # Setup
    data_dir = Path(__file__).parent / "data"
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        if args.gpu:
            logging.warning(f'Torch cannot find GPU. Using device: {device} instead.')

    logging.info(f'Using device: {device}')

    # Hyperparameters
    dataset_args = dict(
        dataset=args.dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        measure="correlation",
        top_percent=10, # Adjusted for Oasis
        grid_size=args.grid_size
    )

    model_args = dict(
        sigma=1.0,
        gamma=0.1,
        categorical_dim=args.categorical_dim,
        embedding_dim=128
    )

    train_args = dict(
        num_epochs=1001,
        save_path=Path(args.save_path) if args.save_path else Path.cwd() / f"models_{args.dataset}_{args.trial}",
        batch_size=4,
        learning_rate=1e-4,
        device=device,
        temp_min=0.05,
        anneal_rate=3e-4,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        temp=1.0,
        eval_every=20,
        early_stopping_patience=15,
        resume_from=args.resume_from,
    )

    inference_args = dict(
        load_path=Path.cwd() / f"models_{args.dataset}_{args.trial}",
        save_path=Path.cwd() / f"models_{args.dataset}_{args.trial}",
        device=device,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        num_samples=1
    )

    # Log all dataset parameters.
    logging.debug('Dataset args: %s', dataset_args)
    # Log all model parameters.
    logging.debug('Model args: %s', model_args)
    # Log all training setup parameters.
    logging.debug('Train args: %s', train_args)
    # Log all inference setup parameters.
    logging.debug('Inference args: %s', inference_args)

    # Dataset
    logging.info('Loading data.')
    dataset = load_dataset(**dataset_args, data_dir=data_dir)
    num_subjects, num_nodes = len(dataset), dataset[0][1][0].number_of_nodes()
    logging.info(f'{num_subjects} subjects with {num_nodes} nodes each.')

    # model
    model = Model(num_subjects, num_nodes, **model_args, device=device)

    if args.command != 'inference':
        # Train
        logging.info('Starting training.')
        train(model, dataset, **train_args)
        logging.info('Finished training.')

    else:
        logging.info('Starting inference.')
        inference(model, dataset, **inference_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DBGDGM model.')
    parser.add_argument('--dataset', required=True, type=str, choices=['ukb', 'hcp', 'oasis'])
    parser.add_argument('--categorical-dim', required=True, type=int)
    parser.add_argument('--valid-prop', default=0.1, type=float)
    parser.add_argument('--test-prop', default=0.1, type=float)
    parser.add_argument('--trial', required=True, type=int)
    parser.add_argument('--gpu', default=None, type=int, choices=[0, 1])
    parser.add_argument('--window-size', default=15, type=int)
    parser.add_argument('--window-stride', default=5, type=int)
    parser.add_argument('--grid-size', default=15, type=int)

    parser.add_argument('--resume-from', default=None, type=str)
    parser.add_argument('--save-path', default=None, type=str,
                        help='Directory to save checkpoints (e.g. /mnt/Training_results/models_oasis_1).')

    subparsers = parser.add_subparsers(dest='command')
    parser_foo = subparsers.add_parser('inference')

    args = parser.parse_args()
    main(args)
