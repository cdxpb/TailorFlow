#!/usr/bin/env python3
"""Check that the inputs we cannot download for you are in place.

Two upstream assets are licence-gated: you have to register, accept the terms and download them
yourself. This script verifies they are present, prints what is missing and where to get it, and exits
non-zero if anything is absent, so `build.py` never fails with a confusing traceback.

    python tailorflow/preprocessing/fetch.py            # check
    python tailorflow/preprocessing/fetch.py --help
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', '..', 'data'))
RAW = os.path.join(DATA, 'C3D_RAW')
SMPL = os.path.join(DATA, 'smpl')

CLOTH3D_URL = 'https://cloth3d.is.tue.mpg.de'
SMPL_URL = 'https://smpl.is.tue.mpg.de'

# name -> (expected absolute path, minimum plausible size, human hint)
CLOTH3D = {
    'train.tar.gz': 'train split, the bulk of the dataset',
    'val_t1.tar.gz': 'validation split',
    'test_t1.tar.gz': 'test split',
}
SMPL_FILES = {
    'SMPL_FEMALE.pkl': 'female body model (~248 MB)',
    'SMPL_MALE.pkl': 'male body model (~39 MB)',
}


def check_group(title, url, base, files, min_mb):
    print(f'\n{title}\n{"=" * len(title)}')
    print(f'  source    {url}   (registration required; downloads are manual)')
    print(f'  expected  {base}/')
    missing, present = [], []
    for name, hint in files.items():
        p = os.path.join(base, name)
        if os.path.isfile(p):
            mb = os.path.getsize(p) / 1e6
            flag = '' if mb >= min_mb else '  <-- SUSPICIOUSLY SMALL, incomplete download?'
            present.append(name)
            print(f'    [ok]     {name:<20} {mb:8.1f} MB  ({hint}){flag}')
        else:
            missing.append(name)
            print(f'    [MISSING] {name:<19}  ({hint})')
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.parse_args()

    miss = []
    miss += check_group('CLOTH3D  (simulated garment meshes + fitted SMPL bodies)',
                        CLOTH3D_URL, RAW, CLOTH3D, min_mb=100)
    miss += check_group('SMPL     (body model, v1.0 or v1.1; first 10 betas are identical)',
                        SMPL_URL, SMPL, SMPL_FILES, min_mb=20)

    print()
    if miss:
        print(f'INCOMPLETE: {len(miss)} file(s) missing -> {", ".join(miss)}')
        print()
        print('Place them at the paths above, then run:')
        print('  bash  tailorflow/preprocessing/extract.sh     # CLOTH3D tarballs -> data/C3D_RAW/{train,val,test}/')
        print('  python tailorflow/preprocessing/build.py      # -> data/C3D_TAPOSE/')
        print('  python tailorflow/preprocessing/check.py      # sanity renders + stats')
        return 1

    print('OK: CLOTH3D archives and SMPL pickles are present.')
    print('Next:')
    print('  bash  tailorflow/preprocessing/extract.sh')
    print('  python tailorflow/preprocessing/build.py')
    print('  python tailorflow/preprocessing/check.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
