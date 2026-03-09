import random
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from src.dataset import load_dataset
from src.inference import inference
from src.model import Model
from src.smri_model import SmriModel
from src.train import train

def main(args):
    log_format = '%(asctime)s [%(levelname)s] %(message)s'

    # StreamHandler → sys.stdout so logs appear live in the terminal.
    # force=True resets any handlers a previously imported module may have set.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))

    resolved_data_mode = args.data_mode if args.data_mode != 'auto' else ('smri' if args.dataset == 'oasis' else 'graph')
    log_prefix = 'smri' if resolved_data_mode == 'smri' else 'fmri'

    file_handler = logging.FileHandler(f'{log_prefix}_{args.dataset}_{args.trial}.log', mode='a')
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
        gpu_name = torch.cuda.get_device_name(device)
        is_h100 = 'H100' in gpu_name.upper()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        gpu_name = 'cpu'
        is_h100 = False
        if args.gpu:
            logging.warning(f'Torch cannot find GPU. Using device: {device} instead.')

    logging.info(f'Using device: {device}')
    logging.info(f'Accelerator: {gpu_name}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    batch_size = args.batch_size if args.batch_size is not None else (32 if is_h100 else 8)
    amp_dtype = args.amp_dtype if args.amp_dtype != 'auto' else ('bf16' if is_h100 else 'fp16')

    # Hyperparameters
    dataset_args = dict(
        dataset=args.dataset,
        data_mode=resolved_data_mode,
        window_size=args.window_size,
        window_stride=args.window_stride,
        measure="correlation",
        top_percent=10, # Adjusted for Oasis
        grid_size=args.grid_size,
        image_size=args.image_size,
        num_slices=args.num_slices,
    )

    model_args = dict(
        sigma=1.0,
        gamma=0.1,
        categorical_dim=args.categorical_dim,
        embedding_dim=args.embedding_dim
    )

    resolved_classification_weight = args.classification_weight
    if resolved_classification_weight is None:
        resolved_classification_weight = 1.0 if resolved_data_mode == 'smri' else 0.0

    train_args = dict(
        num_epochs=args.num_epochs,
        save_path=Path(args.save_path) if args.save_path else Path.cwd() / f"models_{args.dataset}_{args.trial}",
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        temp_min=args.temp_min,
        anneal_rate=args.anneal_rate,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        temp=1.0,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping_patience,
        eval_split=args.subject_eval_prop,
        classification_weight=resolved_classification_weight,
        amp_dtype=amp_dtype,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        resume_from=args.resume_from,
    )

    inference_args = dict(
        load_path=Path.cwd() / f"models_{args.dataset}_{args.trial}",
        save_path=Path.cwd() / f"models_{args.dataset}_{args.trial}",
        device=device,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
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
    if not dataset:
        raise ValueError('Dataset is empty. Check the OASIS data directory and preprocessing configuration.')

    num_subjects = len(dataset)
    num_classes = len({sample[2] for sample in dataset})

    if resolved_data_mode == 'smri':
        input_shape = dataset[0][1]['volume'].shape
        logging.info(f'{num_subjects} subjects with sMRI volume shape {input_shape}.')
    else:
        num_nodes = dataset[0][1][0]['num_nodes']
        logging.info(f'{num_subjects} subjects with {num_nodes} nodes each.')

    logging.info(
        'Resolved training config | data_mode=%s batch_size=%s amp_dtype=%s learning_rate=%s weight_decay=%s classification_weight=%s',
        resolved_data_mode,
        batch_size,
        amp_dtype,
        args.learning_rate,
        args.weight_decay,
        resolved_classification_weight,
    )

    # model
    if resolved_data_mode == 'smri':
        model = SmriModel(input_shape=input_shape, embedding_dim=args.embedding_dim, num_classes=num_classes, device=device)
    else:
        model = Model(num_subjects, num_nodes, num_classes=num_classes, **model_args, device=device)

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
    parser.add_argument('--data-mode', default='auto', choices=['auto', 'graph', 'smri'])
    parser.add_argument('--categorical-dim', required=True, type=int)
    parser.add_argument('--valid-prop', default=0.1, type=float)
    parser.add_argument('--test-prop', default=0.1, type=float)
    parser.add_argument('--subject-eval-prop', default=0.2, type=float)
    parser.add_argument('--trial', required=True, type=int)
    parser.add_argument('--gpu', default=None, type=int, choices=[0, 1])
    parser.add_argument('--window-size', default=15, type=int)
    parser.add_argument('--window-stride', default=5, type=int)
    parser.add_argument('--grid-size', default=15, type=int)
    parser.add_argument('--image-size', default=128, type=int)
    parser.add_argument('--num-slices', default=64, type=int)
    parser.add_argument('--embedding-dim', default=64, type=int)
    parser.add_argument('--batch-size', default=None, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--classification-weight', default=None, type=float,
                        help='Classification loss weight. Defaults to 1.0 for sMRI mode and 0.0 for graph mode.')
    parser.add_argument('--num-epochs', default=301, type=int)
    parser.add_argument('--anneal-rate', default=3e-4, type=float)
    parser.add_argument('--temp-min', default=0.05, type=float)
    parser.add_argument('--eval-every', default=5, type=int)
    parser.add_argument('--early-stopping-patience', default=5, type=int)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--amp-dtype', default='auto', choices=['auto', 'bf16', 'fp16'])

    parser.add_argument('--resume-from', default=None, type=str)
    parser.add_argument('--save-path', default=None, type=str,
                        help='Directory to save checkpoints (e.g. /mnt/Training_results/models_oasis_1).')

    subparsers = parser.add_subparsers(dest='command')
    parser_foo = subparsers.add_parser('inference')

    args = parser.parse_args()
    main(args)
