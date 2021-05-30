
## Baseline

MNLI → Subsequence
```shell
# TinyBERT
python scripts/probing.py 
  --seed 42 
  --name="baseline" 
  --task_config_file="mnli_lex_class_sub.json" 
  --model_name_or_path="seed:42/baseline/bert_multi_nli_reprod"
  --overwrite_cache 
```

## TinyBERT

MNLI → Overlap
```shell
# TinyBERT
python scripts/probing.py 
  --seed 42 
  --name="tiny" 
  --task_config_file="mnli_lex_class.json" 
  --model_name_or_path="implicit/bert_tiny_multi_nli" 
  --embedding_size=128 
  --overwrite_cache 
```