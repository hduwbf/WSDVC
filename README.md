## Proposal Generation via Distillation Learning

### Teacher Networks

Run the bmn.sh files in the action/activity/highlight_teacher folders separately
```bash
sh bmn.sh
```
and move the checkpoints into event_student/checkpoint.
### Student Networks
Run the bmn.sh file
```bash
sh bmn.sh
```

## Cross-modal Matching

Training:
```bash
python train.py
```
Testing:
```bash
python eval.py config/anet_coot.yaml provided_models/anet_coot_AB.pth
```
Make pseudo dataset:
```bash
python make_ws_data.py config/anet_coot.yaml provided_models/anet_coot_AB.pth
```

## Caption Generation

Training:
```bash
python mm_train.py
```
Testing:
```
sh mm_eval.sh
```