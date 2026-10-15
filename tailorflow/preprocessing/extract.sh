#!/bin/bash
# CLOTH3D {train,val_t1,test_t1}.tar.gz -> data/C3D_RAW/{train,val,test}/<id>/{info.mat,<Garment>.obj}
set -e
RAW=${1:-$(cd "$(dirname "$0")/../.." && pwd)/data/C3D_RAW}
UNZIP=$(command -v pigz || echo gzip)
for s in train val_t1 test_t1; do
    out=$RAW/${s%_t1}
    [ -f "$out/.done" ] && { echo "$out done"; continue; }
    mkdir -p "$out"
    echo "extracting $s.tar.gz -> $out"
    tar -I "$UNZIP" -xf "$RAW/$s.tar.gz" -C "$out" --strip-components 1 --wildcards '*.obj' '*/info.mat'
    touch "$out/.done"
    echo "$out: $(ls "$out" | wc -l) samples"
done
