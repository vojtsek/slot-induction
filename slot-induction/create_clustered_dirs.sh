root=$1
newroot=${root%/*}
D=$2
for f in $root/*; do
    mkdir -p $newroot/$D/${f##*/}
    cp $f/raw_$D $newroot/$D/${f##*/}/raw.txt
    cp $f/frames_$D $newroot/$D/${f##*/}/frames.json
    cp $f/state_$D $newroot/$D/${f##*/}/state.json
done
