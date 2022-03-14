
for j in {12..504..6}
do
    echo -n $1 ","  ${j} "," 64
    ${1} 64 64 5 ${j}
done
