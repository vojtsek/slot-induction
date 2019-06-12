root=$1
D=$2
for f in $root/all/*; do
    mkdir -p $root/$D/${f##*/}
    cp $f/raw_$D $root/$D/${f##*/}/raw.txt
    cp $f/frames_$D $root/$D/${f##*/}/frames.json
    cp $f/state_$D $root/$D/${f##*/}/state.json
done
