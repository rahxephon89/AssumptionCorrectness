#include <stdlib.h>
#include <stdio.h>

#define SIZE 5

//@ghost int collection_size_1 = SIZE;

struct collection { int size; int array[SIZE];  };
typedef struct collection collection;

struct iterator {collection c; int pointer; collection *c1;};
typedef struct iterator iterator;

/*@
    predicate p_hasNext(iterator i, iterator j, integer v)  = 
               v == 1 ==> i.pointer < i.c.size &&
               v == 0 ==> j.pointer >= j.c.size;
*/

/*@
    assigns \nothing;
    ensures p_hasNext(*i, \old(*i), \result);
*/
int iterator_hasNext(iterator *i){
  if(i->pointer >= i->c.size)
    return 0; 
  return 1;
}


/*@
    predicate p_next(iterator i, iterator j, integer v) = v == (j.c).array[j.pointer] && 
                          (i).pointer == j.pointer + 1 &&
                          i.pointer <= SIZE;
 */

/*@
        requires \valid(i);
        requires \valid(i->c.array + (0..collection_size_1-1));
        requires 0 <= i->pointer < i->c.size;
        requires \separated(&(i->pointer), (i->c.array + (0..i->c.size-1)));
        assigns  i -> pointer;
        ensures p_next(*i, \old(*i), \result);
*/
int iterator_next(iterator *i) 
{
  L1:
  ;
  int q = 0;
  int ret = 0;
  L3:
  ret = ((*i).c).array[(*i).pointer];
  L2:
  (*i).pointer ++;
  return ret;
}
