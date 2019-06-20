source ../venv3.6/bin/activate
#python induct.py ~/hudecek/data/kvret/parsing/$1 0.3 $1-induction_results_kvret_semafor_parser_embs.txt

python induct.py $1/$2 $3 $4-$2-induction_results_semafor_parser_embs.txt
