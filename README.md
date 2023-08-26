<h2 align="left">detrex</h2>
<p align="left">
    <a href="https://github.com/IDEA-Research/detrex/releases">
        <img alt="release" src="https://img.shields.io/github/v/release/IDEA-Research/detrex">
    </a>
    <a href="https://detrex.readthedocs.io/en/latest/index.html">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href='https://detrex.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/detrex/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/IDEA-Research/detrex/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/IDEA-Research/detrex.svg?color=blue">
    </a>
    <a href="https://github.com/IDEA-Research/detrex/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/IDEA-Research/detrex/issues">
        <img alt="open issues" src="https://img.shields.io/github/issues/IDEA-Research/detrex">
    </a>
</p>

[📘Documentation](https://detrex.readthedocs.io/en/latest/index.html) |
[🛠️Installation](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) |
[👀Model Zoo](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) |
[🚀Awesome DETR](https://github.com/IDEA-Research/awesome-detection-transformer) |
[🆕News](#whats-new) |
[🤔Reporting Issues](https://github.com/IDEA-Research/detrex/issues/new/choose)


## Introduction

detrex is an open-source toolbox that provides state-of-the-art Transformer-based detection algorithms. It is built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and its module design is partially borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection) and [DETR](https://github.com/facebookresearch/detr). Many thanks for their nicely organized code. The main branch works with **Pytorch 1.10+** or higher (we recommend **Pytorch 1.12**).

<div align="center">
  <img src="./assets/detr_arch.png" width="100%"/>
</div>

<details open>
<summary> Major Features </summary>

- **Modular Design.** detrex decomposes the Transformer-based detection framework into various components which help users easily build their own customized models.

- **State-of-the-art Methods.** detrex provides a series of Transformer-based detection algorithms, including [DINO](https://arxiv.org/abs/2203.03605) which reached the SOTA of DETR-like models with **63.3AP**!

- **Easy to Use.** detrex is designed to be **light-weight** and easy for users to use:
  - [LazyConfig System](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more flexible syntax and cleaner config files.
  - Light-weight [training engine](./tools/train_net.py) modified from detectron2 [lazyconfig_train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py)

Apart from detrex, we also released a repo [Awesome Detection Transformer](https://github.com/IDEA-Research/awesome-detection-transformer) to present papers about Transformer for detection and segmentation.

</details>

## Installation

Please refer to [Installation Instructions](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) for the details of installation.

## License

This project is released under the [Apache 2.0 license](LICENSE).


