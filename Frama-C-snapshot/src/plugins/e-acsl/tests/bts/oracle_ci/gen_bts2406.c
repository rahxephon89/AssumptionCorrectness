/* Generated by Frama-C */
#include "stdio.h"
#include "stdlib.h"
char const tab[];
char t[10];
void __e_acsl_globals_init(void)
{
  static char __e_acsl_already_run = 0;
  if (! __e_acsl_already_run) {
    __e_acsl_already_run = 1;
    __e_acsl_store_block((void *)(t),(size_t)10);
    __e_acsl_full_init((void *)(& t));
  }
  return;
}

int main(void)
{
  int __retres;
  __e_acsl_memory_init((int *)0,(char ***)0,(size_t)8);
  __e_acsl_globals_init();
  char *p = (char *)(tab);
  __e_acsl_store_block((void *)(& p),(size_t)8);
  __e_acsl_full_init((void *)(& p));
  /*@ assert ¬\valid(p + (0 .. 9)); */
  {
    int __gen_e_acsl_valid;
    __gen_e_acsl_valid = __e_acsl_valid((void *)(p + 1 * 0),(size_t)9,
                                        (void *)p,(void *)(& p));
    __e_acsl_assert(! __gen_e_acsl_valid,(char *)"Assertion",(char *)"main",
                    (char *)"!\\valid(p + (0 .. 9))",10);
  }
  /*@ assert \valid(&t[0 .. 9]); */
  {
    int __gen_e_acsl_valid_2;
    __gen_e_acsl_valid_2 = __e_acsl_valid((void *)(& t + 1 * 0),(size_t)9,
                                          (void *)(& t),(void *)0);
    __e_acsl_assert(__gen_e_acsl_valid_2,(char *)"Assertion",(char *)"main",
                    (char *)"\\valid(&t[0 .. 9])",11);
  }
  __retres = 0;
  __e_acsl_delete_block((void *)(t));
  __e_acsl_delete_block((void *)(& p));
  __e_acsl_memory_clean();
  return __retres;
}


