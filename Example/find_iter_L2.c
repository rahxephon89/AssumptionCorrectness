
#include "iterator.h"

/*@
   predicate insertbegin = \true;
*/

/*@
   predicate insertend = \true;
*/

/*@
   predicate dagbegin = \true;
*/

/*@
   predicate replacebegin = \true;
*/

/*@
        requires \separated(&iter, &iter2);
	requires iter.size == SIZE;
	requires iter.pointer == 0;
        requires n >= 0;
*/
int find(iterator iter, iterator iter2, int n, int v) 
{
  int i = 0;
  int idx = -1;
  int value;
  int b = 0;
  int point = 0;
  /*@	
    loop invariant idx != -1 ==> iter.c.array[idx] == v;
  */
  while(i < n){
   L1:
   b = iterator_next(&iter2);
   if(b){
      L2:
      //@assert insertbegin;
       if(iterator_hasNext(&iter)){
 	value = iterator_next(&iter);
	    if (value == v) {
	      idx = iter.pointer - 1;
         }
      }
    //@assert replacebegin;
       value = iterator_next(&iter);
       if (value == v) {
           idx = iter.pointer - 1;
       }
   //@assert insertend;
   //@assert dagbegin;
   }

   i++;
  }

  return idx;
}

