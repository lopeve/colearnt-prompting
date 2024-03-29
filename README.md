# Enhanced Co-training for Large Language Models

This repository contains the implementation for our ICML 2022 paper [Co-training Improves Prompt-based Learning for Large Language Models](https://arxiv.org/abs/2202.00828) and subsequent advancements, including tuning methodologies based on  [T-Few](https://github.com/r-three/t-few).

The code is instrumental for:
  - Enhancing the zero-shot and few-shot performance of large language models
  - Distilling large models like GPT-3 and T0 into compact task-specific models.

We sucesfully built many parts of this repository on top of the outstanding [T-Few](https://github.com/r-three/t-few) repository.

If you find this code useful, please consider citing our paper:

```
@inproceedings{lang2022co,
  title={Co-training improves prompt-based learning for large language models},
  author={Lang, Hunter and Agrawal, Monica N and Kim, Yoon and Sontag, David},
  booktitle={International Conference on Machine Learning},
  pages={11985--12003},
  year={2022},
  organization={PMLR}
}
```

Setup, usage, model training, result reproduction, and method of application to your data set are outlined in detail in the original README content.

__Note__:

This is the new home for the project, under the new owner lopeve, for any references or author contacts please see the original paper and repository maintained by clinicalml.