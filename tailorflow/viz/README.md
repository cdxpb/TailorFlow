# Interactive demo

A browser interface for sampling, editing and refitting garments. The model stays loaded on one GPU, so each step
takes a few seconds.

```bash
python tailorflow/viz/server.py --port 8000 --gpu 0
```

The demo needs the checkpoints and the SMPL files (after `tailorflow/preprocessing/smpl_assets.py`), not the dataset.

Open <http://localhost:8000>. On a remote machine, forward the port first: `ssh -L 8000:localhost:8000 <host>`.

## Usage

- **Sample**: choose a garment class, the number of samples and the point count. Optionally paint the area the
  garment should cover on the body.
- **Edit**: paint a new coverage on the selected garment and apply it. Sleeves, hems, straps and necklines grow or
  shrink, and the rest of the garment stays unchanged. Mirror mode paints both sides at once.
- **Transfer**: show the selected garment on several random bodies.
- **Body**: the panel below the viewport sets gender and the 10 SMPL shape parameters. Changing the body refits the
  selected garment.

Garments are shown as point clouds and can be meshed with **Mesh this garment** or **Mesh all garments**.

| Key | Action |
|---|---|
| `S` | sample |
| `T` | transfer |
| `M` | mesh the selected garment |
| `B` | random body |
| `F` | reset the view |
| `←` / `→` | previous / next garment |
| `Space` (hold) | rotate the view while painting |
| `Esc` | stop painting |

## Files

| File | Content |
|---|---|
| `server.py` | Flask server holding the model |
| `bodymodel.py` | SMPL bodies from shape parameters and gender |
| `body_stats.npz` | train-set shape distribution per gender and the train coverage masks, used for random bodies and coverages |
| `static/` | web interface (three.js, included) |
