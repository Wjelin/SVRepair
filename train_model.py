import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

from dataset import SVRepairDataset
from model.svrepair_model import SVRepairModel
from model.config import ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    logger.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.info("CUDA random seeds set")
    logger.info("All random seeds initialized")


def train(model, train_dataset, eval_dataset, args):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = total_steps // 10

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.info("***** Running Training *****")
    logger.info(f"Optimizer: AdamW with lr={args.lr}")
    logger.info(f"Scheduler: Linear warmup for {warmup_steps} steps, then linear decay")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num epochs = %d", args.epochs)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Total steps = %d", total_steps)
    logger.info("  Lambda Dep = %f", args.lambda_dep)

    model.train()
    step = 0
    early_stop = 0
    best_eval_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}"):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["rep_input_ids"],
                dep_distances=batch["dep_distances"],
                dep_scores=batch["dep_scores"],
            )

            loss = outputs["loss"]
            lambda_dep = args.lambda_dep
            dep_loss = torch.tensor(0.0, device=args.device)
            if outputs["dep_loss"] is not None:
                dep_loss = outputs["dep_loss"] 
            total_loss = loss + lambda_dep * dep_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

            if step % 100 == 0:
                logger.info(
                    f"[Step {step}/{total_steps}] "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"Repair Loss: {loss.item():.4f} | "
                    f"Dep Loss: {dep_loss.item():.4f} | "
                    f"Lambda Dep: {lambda_dep:.4f}"
                )

        avg_loss = epoch_loss / len(train_dataloader)
        eval_loss = evaluate(model, eval_dataset, args)

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Average Training Loss (Repair Loss): {avg_loss:.4f}")
        logger.info(f"  Evaluation Loss (Repair Loss): {eval_loss:.4f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_path = os.path.join(args.ckpt_dir, f"best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🏆 New best model saved with eval loss {eval_loss:.4f}")
        else:
            if early_stop >= args.early_stop_patience:
                print("Early stopping.")
                break

        model.train()


def evaluate(model, eval_dataset, args):
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)

    logger.info("***** Running Evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    eval_loss, count = 0, 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["rep_input_ids"],
                dep_distances=batch["dep_distances"],
                dep_scores=batch["dep_scores"],
            )
            eval_loss = outputs["loss"].item()
            count += 1

    eval_loss = eval_loss / count
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Repair Loss: {str(eval_loss)}")
    return eval_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="Salesforce/codet5-base",
                        help="Path to pretrained model.")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "only_aux_task", "only_struct_attn_bias", "baseline"],
                        help="Ablation study configuration.")   
    parser.add_argument("--struct_bias_init", type=str, default="default",
                        choices=["default", "fixed", "zero"],
                        help=(
                            "Initialization strategy for the structural bias table: "
                            "'default' uses the model's standard initialization, "
                            "'fixed' assigns predefined bias values based on dependency levels, "
                            "'zero' initializes all structural bias entries to zero."
                        ))
    parser.add_argument('--ckpt_dir', type=str,
                        help="Directory to save model checkpoints.")
    parser.add_argument("--load_pretrained_model", default=False, action='store_true',
                        help="Whether to load pretrained model")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to model pretrained on Bug-Fix data.")

    parser.add_argument('--train_data_path', type=str, default="data/cve_fixes_and_big_vul/train.jsonl",
                        help="Path to train data file.")
    parser.add_argument('--eval_data_path', type=str, default="data/cve_fixes_and_big_vul/eval.jsonl",
                        help="Path to evaluation data file.")

    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum sequence length for target.")
    parser.add_argument('--target_max_length', type=int, default=256,
                        help="Maximum sequence length for target.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training and evaluation.")
    parser.add_argument('--epochs', type=int, default=75,
                        help="Total number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Initial learning rate.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay (L2 penalty) to apply to the optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon hyperparameter for Adam optimizer to prevent division by zero.")
    parser.add_argument("--lambda_dep", default=1.0, type=float,
                        help="A hyperparameter to weight the auxiliary task loss.")
    parser.add_argument("--early_stop_patience", type=int, default=20,
                    help="Early stopping patience (number of epochs to wait before stopping after no improvement).")

    parser.add_argument('--seed', type=int, default=12345,
                        help="Random seed for initialization.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use for training (cuda or cpu).")
    parser.add_argument("--gpu_id", type=str, default="1",
                        help="GPU device ID to use.")
    args = parser.parse_args()

    args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_path)
    tokenizer.add_tokens([
        "<S2SV_StartBug>", "<S2SV_EndBug>",
        "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"
    ])
    config = ModelConfig.from_preset(args.ablation)
    config.struct_bias_init = args.struct_bias_init

    model = SVRepairModel(args.model_path, config).to(args.device)
    model.t5.resize_token_embeddings(len(tokenizer))
    if args.load_pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model_path))

    train_dataset = SVRepairDataset(tokenizer, args.train_data_path,
                                    max_length=args.max_length, target_max_length=args.target_max_length)
    eval_dataset = SVRepairDataset(tokenizer, args.eval_data_path,
                                   max_length=args.max_length, target_max_length=args.target_max_length)
    train(model, train_dataset, eval_dataset, args)


if __name__ == "__main__":
    main()
