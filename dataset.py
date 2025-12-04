import json
import logging
import math

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SVRepairDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=512, target_max_length=256, alpha=0.1, max_depth=3):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.alpha = alpha
        self.max_depth = max_depth
        self._load_data(data_path)

    def _load_data(self, data_path):
        assert data_path.endswith('.pt') or data_path.endswith('.jsonl')

        if data_path.endswith('.pt'):
            logger.info(f"Loading dataset from {data_path}")
            self.samples = torch.load(data_path)
        else:
            logger.info(f"Processing raw dataset from {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = [json.loads(line.strip()) for line in f]

            for item in tqdm(raw_data):
                self.samples.append(self._process_sample(item))

            cache_path = data_path[:-6] + f"_cached_{str(self.alpha).replace('.','')}.pt"
            torch.save(self.samples, cache_path)
            logger.info(f"Saved processed dataset to {cache_path}")

        for sample in self.samples[:1]:
            logger.info("*** Example ***")
            logger.info(f"Input IDs: {sample['input_ids']}")
            logger.info(f"Attention Mask: {sample['attention_mask']}")
            logger.info(f"Dependency Distances: {sample['dep_distances']}")
            logger.info(f"Dependency Scores: {sample['dep_scores']}")
            logger.info(f"Target IDs: {sample['rep_input_ids']}")
            logger.info(f"Input Shape: {sample['input_ids'].shape}")
            logger.info(f"Output Shape: {sample['rep_input_ids'].shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _process_sample(self, sample):
        source = sample["source"]
        target = sample["target"]
        slices = sample["slices"]

        encoding = self.tokenizer(
            source,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding["input_ids"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        dep_distances = torch.zeros(input_ids.size(0), dtype=torch.long)
        dep_scores = torch.zeros(input_ids.size(0), dtype=torch.float)
        for idx, (start, end) in enumerate(offsets):
            min_distance = self.max_depth + 1
            max_score = 0.0
            for slice_object in slices:
                distance = slice_object["depth"]
                if distance <= self.max_depth and slice_object["start_index"] < end and start < slice_object["end_index"]:
                    min_distance = min(min_distance, distance)
                    max_score = max(max_score, math.exp(-self.alpha * distance ** 2))
            dep_distances[idx] = min_distance
            dep_scores[idx] = max_score

        rep_input_ids = self.tokenizer.encode(
            target,
            truncation=True,
            max_length=self.target_max_length,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dep_distances": dep_distances,
            "dep_scores": dep_scores,
            "rep_input_ids": rep_input_ids
        }


if __name__ == "__main__":
    tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base")
    tokenizer.add_tokens([
        "<S2SV_StartBug>", "<S2SV_EndBug>",
        "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"
    ])
    SVRepairDataset(tokenizer, "data/cve_fixes_and_big_vul/train.jsonl")
    SVRepairDataset(tokenizer, "data/cve_fixes_and_big_vul/eval.jsonl")
    SVRepairDataset(tokenizer, "data/cve_fixes_and_big_vul/test.jsonl")
    SVRepairDataset(tokenizer, "data/vrepair_bug_data/train.jsonl")
    SVRepairDataset(tokenizer, "data/vrepair_bug_data/eval.jsonl")
