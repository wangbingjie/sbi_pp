# SBI++

A complete methodology based on simulation-based (likelihood-free) inference that is customized for astronomical applications. Specifically, SBI++ retains the fast inference speed of ∼1 sec for objects in the observational training set distribution, and additionally permits parameter inference outside of the trained noise and data at ~1 min per object.

This repository contains the following scripts:

* `sbi_train.py`, which illustrates how to train an SBI model

* `sbi_pp.py`, which includes all the functions implementing SBI++

* `tutorial.ipynb`, which is a short tutorial showing the workings of SBI++

## Dependency

The [sbi](https://www.mackelab.org/sbi/) Python package

## Reference

Wang et al., 2023b. ADS link to be included.
