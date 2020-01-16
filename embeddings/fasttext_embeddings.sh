LANG=$1

mkdir fastText157

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$LANG.300.bin.gz --directory-prefix=./fastText157

gzip -dc <./fastText157/cc.$LANG.300.bin.gz > ./fastText157/cc.$LANG.300.vec.gz.top1.bin

rm ./fastText157/cc.$LANG.300.bin.gz

touch ./fastText157/cc.$LANG.300.vec.gz.top1
