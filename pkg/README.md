# infini-gram

To use this package, please check out its **documentation** here: <https://infini-gram.io/pkg_doc>

To learn more about infini-gram:
* Paper: <https://arxiv.org/abs/2401.17377>
* Project Home: <https://infini-gram.io>
* Web Interface: <https://infini-gram.io/demo>
* API Endpoint: <https://infini-gram.io/api_doc>

## License

This package is licensed under the [UW Academic Software License](https://infini-gram.io/LICENSE).
Use by universities and non-profit institutions is allowed.
Commercial use is not allowed.
A copy of the license is enclosed with the package distribution.

The suffix array implementation is adapted from [Lee et al. (2021)](https://github.com/google-research/deduplicate-text-datasets), which is distributed under Apache-2.0.

## Citation

If you find infini-gram useful, please kindly cite our paper:
```bibtex
@article{Liu2024InfiniGram,
  title={Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens},
  author={Liu, Jiacheng and Min, Sewon and Zettlemoyer, Luke and Choi, Yejin and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2401.17377},
  year={2024}
}
```

## Building the package
`python -m pip wheel ./pkg --wheel-dir=./build/cp311-macosx_arm64/built_wheel --no-deps`