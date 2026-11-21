import argparse
import torch
from train import TrainConfig, train 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="d4", help="e.g,, d4, d6, d8")
    parser.add_argument("--seqs", type=int, default=20_000, help="num_train_sequences")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--micro-batches", type=int, default=4, help="Increase if T4 OOMs")
    parser.add_argument("--suffix", type=str, default="t4-baseline")
    parser.add_argument("--cpu-test", action="store_true", help="Allow CPU smoke test")
    args = parser.parse_args()

    if not args.cpu_test and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Pass --cpu-test to run on CPU.")

    config = TrainConfig(
            model_name=args.model,
            num_train_sequences=args.seqs,
            learning_rate=args.lr,
            weight_decay=args.wd,
            batch_size=64,
            num_micro_batches=args.micro_batches,
            num_evals=20,
            data_seed=None,
            wandb_online=False,
            save_model=False,
            latest_checkpoint_frequency=None,
            run_name_suffix=args.suffix 
        )
    train(config)


if __name__ == "__main__":
    main()
