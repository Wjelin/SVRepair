import argparse
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from dataset import SVRepairDataset
from model.svrepair_model import SVRepairModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def test(model, tokenizer, test_dataset, args):
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Num beams = %d", args.num_beams)

    model.eval()
    accuracy = []
    raw_predictions = []
    all_predictions = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            beam_outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                dep_distances=batch["dep_distances"],
                dep_scores=batch["dep_scores"],
                do_generate=True,
                num_beams=args.num_beams,
            )
        beam_outputs = beam_outputs.detach().cpu().tolist()
        labels = batch["rep_input_ids"].detach().cpu().tolist()
        beam_predictions = []
        correct_pred = False
        for single_output in beam_outputs:
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            beam_predictions.append(prediction)

            ground_truth = tokenizer.decode(labels[0], skip_special_tokens=False)
            ground_truth = clean_tokens(ground_truth)
            if prediction == ground_truth:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
        all_predictions.append(beam_predictions)
        t = str(round(sum(accuracy) / len(accuracy), 4))
        logger.info(f"test acc: {t}")

    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")
    return all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Salesforce/codet5-base",
                        help="Path to pretrained model.")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "only_aux_task", "only_struct_attn_bias", "baseline"],
                        help="Ablation study configuration.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to model checkpoint for weights initialization.")

    parser.add_argument('--test_data_path', type=str, default="data/cve_fixes_and_big_vul/test.jsonl",
                        help="Path to test data file.")

    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for input/target.")
    parser.add_argument('--target_max_length', type=int, default=256,
                        help="Maximum sequence length for target.")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam size for beam search (number of sequences to generate).")

    parser.add_argument("--gpu_id", type=str, help="GPU device ID to use.")
    args = parser.parse_args()

    args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_path)
    tokenizer.add_tokens([
        "<S2SV_StartBug>", "<S2SV_EndBug>",
        "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"
    ])

    model = SVRepairModel(args.model_path, args.ablation)
    model.t5.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model = model.to(args.device)

    test_dataset = SVRepairDataset(tokenizer, args.test_data_path, max_length=args.max_length, target_max_length=args.target_max_length)

    test(model, tokenizer, test_dataset, args)


if __name__ == "__main__":
    main()
