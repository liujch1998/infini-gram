subsets=("wiki" "books" "pes2o" "reddit" "c4" "stack")

for subset in "${subsets[@]}"; do
    echo ""
    echo ${subset}
    python main.py --data_dir /data_c/dolma-v1_6/${subset} --temp_dir /data_t/v4_dolma-v1_6-${subset}_llama --save_dir /data_i/v4_dolma-v1_6-${subset}_llama --tokenizer llama --cpus 128 --mem 850
done
