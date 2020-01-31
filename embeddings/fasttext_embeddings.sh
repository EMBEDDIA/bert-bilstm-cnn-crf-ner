#!/bin/sh

LANG=$1

DIR="${0%/*}"

DIR=$DIR/fastText157

if [[ ! -d "$DIR" ]]
then
    mkdir $DIR
fi

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$LANG.300.bin.gz --directory-prefix=$DIR

echo "Unzipping file"
gzip -dc <$DIR/cc.$LANG.300.bin.gz > $DIR/cc.$LANG.300.vec.gz.top1.bin

echo "Cleaning $DIR"
rm $DIR/cc.$LANG.300.bin.gz
