python joern_parse.py \
    --joern_path=~/bin/joern/joern-cli/ \
    --data_dir=~/data/cve_fixes_and_big_vul/train/  \
    --batch_size=100 2>&1 | tee log/data/vul_train.log

python joern_parse.py \
    --joern_path=~/bin/joern/joern-cli/ \
    --data_dir=~/data/cve_fixes_and_big_vul/eval/  \
    --batch_size=100 2>&1 | tee log/data/vul_eval.log

python joern_parse.py \
    --joern_path=~/bin/joern/joern-cli/ \
    --data_dir=~/data/cve_fixes_and_big_vul/test/  \
    --batch_size=100 2>&1 | tee log/data/vul_test.log

python joern_parse.py \
    --joern_path=~/bin/joern/joern-cli/ \
    --data_dir=~/data/vrepair_non_domain_data/train/  \
    --batch_size=100 2>&1 | tee log/data/bug_train.log

python joern_parse.py \
    --joern_path=~/bin/joern/joern-cli/ \
    --data_dir=~/data/vrepair_non_domain_data/eval/  \
    --batch_size=100 2>&1 | tee log/data/bug_eval.log