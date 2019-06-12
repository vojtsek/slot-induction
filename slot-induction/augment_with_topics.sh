frame_file=$1
topic_script=/home/hudecek/hudecek/dialogues-semi-sup/slot-induction/BTM/script/runExample.sh
topic_output=/home/hudecek/hudecek/dialogues-semi-sup/slot-induction/BTM/output/model
utterances=/home/hudecek/hudecek/dialogues-semi-sup/slot-induction/BTM/data/camrest.txt
topic_count=10
cwd=$PWD

python frames2plain.py $frame_file > $frame_file-plain
count_plain=`wc -l < $frame_file-plain`
cat $utterances $frame_file-plain > ${utterances%/*}/cated.txt
sh $topic_script $topic_count ${utterances%/*}/cated.txt > /dev/null
paste -d' ' <(tail -n $count_plain $topic_output/k$topic_count.pz_d) $frame_file
