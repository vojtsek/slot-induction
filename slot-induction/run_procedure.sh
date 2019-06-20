set -x
stage=$1
shift

root=$1
topic_count=10
pickled_sentences=sentences-$3.pkl
if [ "$stage" = "topic" ]; then
    topic_script=/home/hudecek/hudecek/dialogues-semi-sup/slot-induction/BTM/script/runExample.sh
    topic_output=/home/hudecek/hudecek/dialogues-semi-sup/slot-induction/BTM/output/model
    btm_utterances=/home/hudecek/hudecek/dialogues-semi-sup/slot-induction/BTM/data/utterances.txt
    btm_ids=/home/hudecek/hudecek/dialogues-semi-sup/slot-induction/BTM/data/utterances_ids.txt
    rm $btm_utterances
    rm $btm_ids
    for f in $root/*; do cat $f/raw.txt >> $btm_utterances; done
    for f in $root/*; do cat $f/raw.txt | while read line; do echo $f >> $btm_ids; done; done
    for f in $root/*; do rm $f/topics.txt; done
    cwd=$PWD
    sh $topic_script $topic_count $btm_utterances > /dev/null
    paste -d'&' $btm_ids $topic_output/k$topic_count.pz_d | while read line; do f=`echo $line | cut -d"&" -f1`; v=`echo $line | cut -d"&" -f2`; echo $v >> $f/topics.txt; done
elif [ "$stage" = "elmo" ]; then
    embedding_file=$2
    python compute_elmo.py $root elmo/model $embedding_file
elif [ "$stage" = "cluster" ]; then
    embedding_file=$2
    python cluster_intents.py --root $root --embeddings $embedding_file --features --features_file $pickled_sentences
    qsub -cwd -N autoencoding -j y -q cpu-troja.q submit-ae.sh $topic_count $pickled_sentences
    echo "Features computed, submitted autoencoding. Wait for it to finish and then run this with cluster2"
elif [ "$stage" = "cluster2" ]; then
    embedding_file=$2
    python cluster_intents.py --root $root --embeddings $embedding_file --features_file $pickled_sentences > clustered_intents.out
    cat clustered_intents.out | python copy_clusterized.py "semafor-frames.json" "frames_"
    cat clustered_intents.out | python copy_clusterized.py "raw.txt" "raw_"
    cat clustered_intents.out | python copy_clusterized.py "state.json" "state_"
    bash create_clustered_dirs.sh $root clust0
    bash create_clustered_dirs.sh $root clust1
    bash create_clustered_dirs.sh $root clust2
elif [ "$stage" = "induct" ]; then
    embedding_file=$2
    qsub -cwd -j y -q cpu-troja.q submit.sh $root clust0 $embedding_file $3
    qsub -cwd -j y -q cpu-troja.q submit.sh $root clust1 $embedding_file $3
    qsub -cwd -j y -q cpu-troja.q submit.sh $root clust2 $embedding_file $3
    qsub -cwd -j y -q cpu-troja.q submit.sh $root all $embedding_file $3
elif [ "$stage" = "analyze" ]; then
    data=$2
    cluster=$3
    if [ ! -z $4 ]; then
        bash augment_with_topics.sh $data-$cluster-induction_results_semafor_parser_embs.txt > $data-$cluster-induction_results_semafor_parser_embs_topics.txt
    fi
    python analyze.py --frame_file $data-$cluster-induction_results_semafor_parser_embs_topics.txt --similarities similarities-set.json --alpha 0.7 --output_mapping inducted-$data-$cluster.pkl --root $root --chosen_slots slots-chosen-$data-$cluster.pkl
else
    echo "UNKNOWN stage!"
fi
