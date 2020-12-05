# On correctness of assumption enforcement

This repository contains the code base for the paper "On correctness of assumption enforcement".

## Contents

* Frama-C-snapshot
* Example



## Installation

Since the tool is a non-standard version of Frama-C, it is recommended to use opam to install dependencies and then compile the source:

* Clone the code in this repository
* Install opam 4.05.0
* Install the customized Frama-C using the following commands where <dir> is the directory  Frama-C-snapshot.

```
    opam install depext
    opam depext frama-c
    opam install --deps-only frama-c
    opam pin add --kind=path frama-c <dir> 
```

More information can be found in the GitHub page of Frama-C: [Installation of Frama-C](https://github.com/Frama-C/Frama-C-snapshot/blob/20.0/INSTALL.md)






