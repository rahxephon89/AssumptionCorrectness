
//#include "find.h"
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
	//requires \valid_read(c.array + (0..SIZE-1));
        requires \separated(&iter, &iter2);
	requires iter.c.size == SIZE;
	requires iter.pointer == 0;
        requires n >= 0;
	//ensures \result != -1 ==> iter.c.array[\result] == v;
	
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
    b = iterator_next(&iter2);
    /* 
   //@assert insertbegin;
       if(iterator_hasNext(&iter)){
 	value = iterator_next(&iter);
	    if (value == v) {
	      idx = iter.pointer - 1;
         }
        }
	    //}else{
	      //idx = idx;
            //}
    //@assert replacebegin;
  */
   if(b){
       value = iterator_next(&iter);
       if (value == v) {
	idx = iter.pointer - 1;
       }
   }

   ///@assert insertend;
   ///@assert dagbegin;
   i++;
  }

  return idx;
}

