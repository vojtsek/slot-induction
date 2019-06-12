source ../venv3.6/bin/activate
feat_size=$1
file=$2
python autoencode.py --feature boe --output $file --input $file --encoded_dim $feat_size
python autoencode.py --feature bow --output $file --input $file --encoded_dim $feat_size
python autoencode.py --feature bop --output $file --input $file --encoded_dim $feat_size
