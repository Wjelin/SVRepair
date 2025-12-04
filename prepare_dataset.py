import json
from pathlib import Path

from tqdm import tqdm

DATA_TYPE="test"
DATA_DIR=Path(f"~/data/cve_fixes_and_big_vul/{DATA_TYPE}/").expanduser()
OUTPUT_DIR=Path("./data/cve_fixes_and_big_vul/")
TOKENIZER_PATH = "Salesforce/codet5-base"
START_MARKER = "<S2SV_StartBug> "
END_MARKER = " <S2SV_EndBug>"
SPECIAL_TOKENS = [
    "<S2SV_StartBug>", "<S2SV_EndBug>",
    "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"
]
ADD_CWE_PREFIX=True


def insert_markers(source_code, vul_ranges, cwe_id):
    prefix_len = 0
    marked_code = source_code
    if ADD_CWE_PREFIX and cwe_id:
        marked_code = f"{cwe_id} {marked_code}"
        prefix_len = len(cwe_id) + 1

    insertions = []
    for start_idx, end_idx in vul_ranges:
        insertions.append((start_idx + prefix_len, START_MARKER))
        insertions.append((end_idx + prefix_len, END_MARKER))

    insertions.sort(key=lambda x: x[0], reverse=True)

    for idx, marker in insertions:
        marked_code = marked_code[:idx] + marker + marked_code[idx:]

    insertions.reverse()
    return marked_code, insertions


def adjust_slice_indices(slices, insertions, prefix_len):
    for code_slice in slices:
        shift_start = 0
        shift_end = 0
        code_slice["start_index"] += prefix_len
        code_slice["end_index"] += prefix_len
        for idx, marker in insertions:
            length = len(marker)
            if code_slice["start_index"] >= idx:
                shift_start += length
            if code_slice["end_index"] > idx:
                shift_end += length
        code_slice["start_index"] += shift_start
        code_slice["end_index"] += shift_end
    return slices


def validate_slices(marked_code, slices):
    flag = True
    for code_slice in slices:
        extracted = marked_code[code_slice["start_index"]:code_slice["end_index"]]
        if extracted != code_slice["code"]:
            flag = False
    return flag


def main():
    dataset = []

    code_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )
    for index_dir in tqdm(code_dirs, total=len(code_dirs)):
        data = {}

        with open(index_dir / f"{index_dir.name}.c", 'r') as file:
            source = file.read()
        with open(index_dir / "vulnerabilities.json", 'r') as file:
            vulnerabilities = json.load(file)
        with open(index_dir / "slices.json", 'r') as file:
            slices = json.load(file)
        with open(index_dir / "target.txt", 'r') as file:
            target = file.read()
        with open(index_dir / "info.json", 'r') as file:
            info = json.load(file)

        vul_ranges = [(vul["start_index"], vul["end_index"]) for vul in vulnerabilities]
        cwe_id = info["cwe_id"]
        prefix_len = len(cwe_id) + 1 if ADD_CWE_PREFIX and cwe_id else 0
        marked_code, insertions = insert_markers(source, vul_ranges, cwe_id)
        adjusted_slices = adjust_slice_indices(slices, insertions, prefix_len)

        assert validate_slices(marked_code, adjusted_slices)

        data.update({
            "source": marked_code,
            "vulnerabilities": vulnerabilities,
            "slices": adjusted_slices,
            "target": target
        })
        dataset.append(data)

    with open(OUTPUT_DIR / f"{DATA_TYPE}{'_with_prefix' if ADD_CWE_PREFIX else ''}.jsonl", 'w', encoding="utf-8") as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


if __name__ == "__main__":
    main()
