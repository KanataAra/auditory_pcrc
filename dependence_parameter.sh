for j in  `seq 0.0 0.05 1.0 `
do
    for i in `seq 0 10 `
    do
	./hpcrc.py $j
    done
done
