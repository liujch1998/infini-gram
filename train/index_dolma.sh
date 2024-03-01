subsets=("wiki" "books" "pes2o" "reddit" "c4" "stack")

for subset in "${subsets[@]}"; do
    echo ""
    echo ${subset}
    python main.py --data_dir /data_c/dolma-v1_6/${subset} --temp_dir /data_t/v4_dolma-v1_6-${subset}_llama --save_dir /data_i/v4_dolma-v1_6-${subset}_llama --tokenizer llama --cpus 128 --mem 850
done


subsets=("head" "middle" "tail")

for subset in "${subsets[@]}"; do
    if [ "$subset" = "tail" ]; then
        shards=4
    else
        shards=2
    fi
    echo ""
    echo "cc_${subset}"
    echo "${shards}"
    python main.py --data_dir /data_c/dolma-v1_6/cc_en_${subset} --temp_dir /data_t/v4_dolma-v1_6-cc_en_${subset}_llama --save_dir /data_i_${subset}/v4_dolma-v1_6-cc_en_${subset}_llama --tokenizer llama --shards ${shards} --cpus 128 --mem 850
done
