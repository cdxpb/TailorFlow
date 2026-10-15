# Data preprocessing

Converts the CLOTH3D release into the training data. Run from the repository root.

## Inputs

| File | Source |
|---|---|
| `data/C3D_RAW/{train,val_t1,test_t1}.tar.gz` | CLOTH3D, <https://cloth3d.is.tue.mpg.de> |
| `data/smpl/SMPL_{FEMALE,MALE}.pkl` | SMPL v1.0 or v1.1, <https://smpl.is.tue.mpg.de> |

## Steps

```bash
python tailorflow/preprocessing/fetch.py                 # check that the inputs are in place
bash   tailorflow/preprocessing/extract.sh               # unpack garment meshes and metadata
python tailorflow/preprocessing/smpl_assets.py           # SMPL faces and body-part labels
python tailorflow/preprocessing/build.py --workers 16    # build the dataset
python tailorflow/preprocessing/check.py                 # optional: statistics and preview renders
python tailorflow/preprocessing/body_stats.py            # optional: rebuild viz/body_stats.npz for the demo
```

`build.py` can be stopped and restarted; finished items are skipped.

## Processing

For each garment, `build.py`:

1. builds the SMPL body from the garment's shape parameters and gender,
2. keeps the garment and body in the CLOTH3D rest pose and centres them on the pelvis,
3. samples 4096 surface points with farthest point sampling,
4. computes which body vertices the garment covers by casting rays along the body normals.

Coverage edits used for training (shorter hems, sleeves, straps) are cut from the garment mesh on the fly
(`tailorflow/occ_aug.py`), so they are not stored.

## Output

`data/C3D_TAPOSE/{train,val,test}/{Dress,Jumpsuit,Skirt,Trousers,Upper}/<id>.npz`

| Key | Shape | Content |
|---|---|---|
| `points` | 4096 x 3 | garment surface points |
| `body`, `body_normals` | 6890 x 3 | SMPL body vertices and normals |
| `occ` | 6890 | body vertices covered by the garment |
| `betas`, `gender` | 10, 1 | SMPL shape and gender |
| `verts`, `faces`, `owner` | | garment mesh and the body vertex each garment vertex belongs to |

Set `C3D_ROOT` to read the dataset from another location.
