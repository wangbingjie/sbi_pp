# SBI++

A complete methodology based on simulation-based (likelihood-free) inference that is customized for astronomical applications. Specifically, SBI++ retains the fast inference speed of ∼1 sec for objects in the observational training set distribution, and additionally permits parameter inference outside of the trained noise and data at ~1 min per object.

This repository contains the following scripts:

* `sbi_train.py`, which illustrates how to train an SBI model

* `sbi_pp.py`, which includes all the functions implementing SBI++

* `tutorial.ipynb`, which is a short tutorial showing the workings of SBI++

## Dependency

The [sbi](https://www.mackelab.org/sbi/) Python package, although the algorithms implemented in `sbi_pp.py` is not package-specific.

## Citation

If you find this code useful in your research, please cite [Wang et al., 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230405281W/abstract):

```
@ARTICLE{2023arXiv230405281W,
       author = {{Wang}, Bingjie and {Leja}, Joel and {Villar}, V. Ashley and {Speagle}, Joshua S.},
        title = "{SBI++: Flexible, Ultra-fast Likelihood-free Inference Customized for Astronomical Application}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = apr,
          eid = {arXiv:2304.05281},
        pages = {arXiv:2304.05281},
archivePrefix = {arXiv},
       eprint = {2304.05281},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230405281W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
and other dependencies that you may have.
