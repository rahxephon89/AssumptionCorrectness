# On correctness of assumption enforcement

This repository contains the code base for the paper "On correctness of assumption enforcement".


## Contents

* Frama-C-snapshot
* Example

The directory Frama-C-snapshot contains the customized version of Frama-C to generate local proof implication using the method presented in the paper. The directory Example contains the code for the case study. 

## Installation

Since the tool is a non-standard version of Frama-C, it is recommended to use opam to install dependencies and then compile the source:

* Clone the code in this repository
* Install opam 4.05.0
* Install the customized Frama-C using the following commands where <dir> is the directory Frama-C-snapshot:

```
    opam install depext
    opam depext frama-c
    opam install --deps-only frama-c
    opam pin add --kind=path frama-c <dir> 
    opam install frama-c
```

Then go to the directory Frama-C-snapshot/src/plugins/qed and Frama-C-snapshot/src/plugins/wp and execute command respectively:

```
    make; make install
```

* Install [Coq] (https://coq.inria.fr/opam-using.html)

More information can be found in the GitHub page of Frama-C: [Installation of Frama-C](https://github.com/Frama-C/Frama-C-snapshot/blob/20.0/INSTALL.md)



## Execution of the example program

 After executing the following command, the proof obligation will be printed to the file output.txt.

 ```
   frama-c -wp-msg-key "VCGen" -wp find.c > output.txt
 ```



