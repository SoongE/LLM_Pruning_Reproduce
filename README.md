# LLM Pruning Reproduce

## Implemented Methods
- [x] [Streamline](https://arxiv.org/pdf/2403.19135), ICLR'25
- [ ] LaCo, Findings of EMNLP'24
- [ ] ShortGPT, Findings of ACL'25
- [ ] [RestoringLCC](https://arxiv.org/pdf/2510.21834), NeurIPS'25

## Dataset Details
- Datasets: [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  - subset / split: sample-10BT / train
- Number of samples
  - Train: 120000
  - Test: 4000
  - Calibration: first 50 in shuffled train samples
- Seed: 42
- Max Length: 1024

## Recovery Implementation Details
### Streamline
- type: layer-wise distillation
- epoch: 20
- batch: 8
- grad_accumulation: 1
- lr: 1e-4
- min_lr: 1e-5
