# SBI++

A complete methodology based on simulation-based (likelihood-free) inference that is customized for astronomical applications. Specifically, SBI++ retains the fast inference speed of ∼1 sec for objects in the observational training set distribution, and additionally permits parameter inference outside of the trained noise and data at ~1 min per object.

This repository contains the following scripts:

* `sbi_train.py`, which illustrates how to train an SBI model

* `sbi_pp.py`, which includes all the functions implementing SBI++

* `tutorial.ipynb`, which is a short tutorial showing the workings of SBI++

## Dependency

The [sbi](https://github.com/sbi-dev/sbi) Python package, although the algorithms implemented in `sbi_pp.py` is not package-specific.

## Citations

If you find this code useful in your research, please cite [Wang et al., 2023](https://ui.adsabs.harvard.edu/abs/2023ApJ...952L..10W/abstract):

```
@ARTICLE{2023ApJ...952L..10W,
       author = {{Wang}, Bingjie and {Leja}, Joel and {Villar}, V. Ashley and {Speagle}, Joshua S.},
        title = "{SBI$^{++}$: Flexible, Ultra-fast Likelihood-free Inference Customized for Astronomical Applications}",
      journal = {\apjl},
     keywords = {Algorithms, Astrostatistics, Computational astronomy, 1883, 1882, 293, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = jul,
       volume = {952},
       number = {1},
          eid = {L10},
        pages = {L10},
          doi = {10.3847/2041-8213/ace361},
archivePrefix = {arXiv},
       eprint = {2304.05281},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...952L..10W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If using the code as is, please also cite the following:

The sbi package
```
@article{tejero-cantero2020sbi,
  doi = {10.21105/joss.02505},
  url = {https://doi.org/10.21105/joss.02505},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {52},
  pages = {2505},
  author = {Alvaro Tejero-Cantero and Jan Boelts and Michael Deistler and Jan-Matthis Lueckmann and Conor Durkan and Pedro J. Gonçalves and David S. Greenberg and Jakob H. Macke},
  title = {sbi: A toolkit for simulation-based inference},
  journal = {Journal of Open Source Software}
}
```

and the SNPE_C algorithm
```
@ARTICLE{2019arXiv190507488G,
       author = {{Greenberg}, David S. and {Nonnenmacher}, Marcel and {Macke}, Jakob H.},
        title = "{Automatic Posterior Transformation for Likelihood-Free Inference}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
         year = 2019,
        month = may,
          eid = {arXiv:1905.07488},
        pages = {arXiv:1905.07488},
          doi = {10.48550/arXiv.1905.07488},
archivePrefix = {arXiv},
       eprint = {1905.07488},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190507488G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```