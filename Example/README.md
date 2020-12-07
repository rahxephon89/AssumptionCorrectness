This directory contains the example code in the paper. *find_iter_original.c* is the original program while *find_iter_L1.c* and *find_iter_L2.c* are the modified programs with instrumented enforcement actions for the location *L1* and *L2*. Since the paper focuses on the enforcement actions, the code to call the enforcment monitors is omitted. 

Instead of extracting the proof obligation respectively from the original and the modified program, predicates are used in the program as the flag to mark the original code and the instrumented enforcement actions. For instance, in *find_iter.L1.c*, we have the following code snippet:

```
//@assert insertbegin;
	 if(iterator_hasNext(&iter2)){
		b = iterator_next(&iter2);
	 }else
		b = 0;
//@assert replacebegin;
        b = iterator_next(&iter2);
//@assert insertend;
//@assert dagbegin;
```

The snippet between the predicate *replacebegin* and *insertend* is in the original code while the snippet between the predicate *insertbegin* and *replacebegin* is the enforcement actions to replace the snippet in the original code. 

To generate the local verification condition, the following command is executed:

```
   frama-c -wp-msg-key "VCGen" -wp find_iter_L1.c > output.txt
```

The generated file lists the proof obligation of the original (VC) and modified program (VC') before and after the name unification. Then, the result of integrating VC into VC' is also printed out. Note that the output in the paper has been simplified by removing the verification condition at the else branch and rewritten into Gallina manually. 
