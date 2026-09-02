# data/labels/

Label files passed to the demo scripts via `--label` (one class name per line,
ordered by class index):

- `my_label.txt`    — 18-class base model (matches `data/myvideo/classInd.txt`)
- `my_label_7.txt`  — legacy 7-class model
- `my_label_dt.txt` — digital-twin label variant

Canonical class indices live next to their datasets and are referenced by the
configs and evaluation tools — they intentionally stay there:

- 18-class: `data/myvideo/classInd.txt`
- 4-class digital twin: `data/digitaltwin_action/label_4cls.txt` and `classInd.txt`
