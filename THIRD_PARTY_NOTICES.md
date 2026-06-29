# Third-Party Notices

This repository separates public code and sanitized loss matrices from private
training infrastructure, model implementations, weights, and data composition.

## Public Smoke Assets

- `loss_acquisition/llm_pretraining_sglang/data/hf_smollm2_smoke/validation.jsonl`
  contains four public validation examples generated from the TinyStories
  validation split.
- `loss_acquisition/llm_pretraining_sglang/configs/hf_smollm2_smoke.yaml`
  uses `HuggingFaceTB/SmolLM2-135M` as a public SGLang smoke-test model.

Please review the upstream HuggingFace model and dataset cards before
redistributing modified versions of those assets.

## Synthetic Mano

The synthetic Mano data generation, training recipe, and evaluation path are
adapted from the PhysicsLM4 release:

```bibtex
@inproceedings{Allen2025-canon,
  author = {{Allen-Zhu}, Zeyuan},
  title = {{Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers}},
  year = {2025},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems},
  series = {NeurIPS~'25},
  note = {Full version available at \url{https://ssrn.com/abstract=5240330}}
}

@misc{Allen2025-resonate,
  title = {{Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality}},
  author = {{Allen-Zhu}, Zeyuan},
  year = {2025},
  url = {https://physics.allen-zhu.com/part-4-architecture-design/part-4-2},
  note = {Code released at \url{https://github.com/facebookresearch/PhysicsLM4}},
}
```

## Runtime Dependencies

Optional loss acquisition paths can use SGLang, HuggingFace Transformers,
HuggingFace Datasets, and the HuggingFace Hub client. They are optional and are
not required for the core `tokenlearning` analysis package.
