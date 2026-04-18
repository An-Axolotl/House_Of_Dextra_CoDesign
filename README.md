# House of Dextra: Cross-Embodied Co-Design for Dexterous Hands

![House of Dextra](header.gif)

House of Dextra is the unified repository for the dexterous hand co-design stack used in *House of Dextra: Cross-Embodied Co-Design for Dexterous Hands*.

At a high level, the project combines:
- morphology generation for candidate hand designs,
- simulation-based policy evaluation and search,
- and real-world control/deployment tooling.

For full method details, experiment setup, and results, see https://openreview.net/pdf?id=k8ovuXEQQu.

For hardware build instructions, see the [House of Dextra Build Guide](https://an-axolotl.github.io/HouseofDextra/build_guide).

## Repository Map

- [`Main/`](Main/): graph-heuristic search loop and experiment orchestration.
- [`IsaacLab/`](IsaacLab/): simulation environments, policies, and task integration.
- [`Generation/`](Generation/): hand asset generation and conversion pipeline.
- [`RealControl/`](RealControl/): real hardware control and deployment utilities.

## Read This Next

The top-level README is intentionally high-level. Use the setup guide and directory READMEs for workflow-specific instructions:
- [`SETUP.md`](SETUP.md)
- [`Main/README.md`](Main/README.md)
- [`IsaacLab/README.md`](IsaacLab/README.md)
- [`Generation/README.md`](Generation/README.md)
- [`RealControl/README.md`](RealControl/README.md)

Setup, prerequisites, and Isaac Sim installation steps are documented in [`SETUP.md`](SETUP.md).

## Citation

If you use this in your research, please cite:

```bibtex
@article{fay2025crossembodied,
  title={House of Dextra: Cross Embodied Co-Design for Dexterous Hands},
  author={Fay, Kehlani and Djapri, Darin and Zorin, Anya and Clinton, James
          and El Lahib, Ali and Su, Hao and Tolley, Michael T. and Yi, Sha
          and Wang, Xiaolong},
  journal={arXiv preprint},
  year={2025},
  month={December}
}
```
