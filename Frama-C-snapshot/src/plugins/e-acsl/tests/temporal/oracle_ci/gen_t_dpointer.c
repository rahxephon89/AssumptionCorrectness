/* Generated by Frama-C */
#include "stdio.h"
#include "stdlib.h"
int main(void)
{
  int __retres;
  __e_acsl_memory_init((int *)0,(char ***)0,(size_t)8);
  __e_acsl_temporal_reset_parameters();
  __e_acsl_temporal_reset_return();
  int **p = malloc(sizeof(int *) * (unsigned long)3);
  __e_acsl_store_block((void *)(& p),(size_t)8);
  __e_acsl_full_init((void *)(& p));
  __e_acsl_temporal_store_nblock((void *)(& p),(void *)*(& p));
  int i = 0;
  while (i < 3) {
    /*@ assert \valid(p + i); */
    {
      int __gen_e_acsl_valid;
      __gen_e_acsl_valid = __e_acsl_valid((void *)(p + i),sizeof(int *),
                                          (void *)p,(void *)(& p));
      __e_acsl_assert(__gen_e_acsl_valid,(char *)"Assertion",(char *)"main",
                      (char *)"\\valid(p + i)",12);
    }
    __e_acsl_temporal_reset_parameters();
    __e_acsl_temporal_reset_return();
    __e_acsl_initialize((void *)(p + i),sizeof(int *));
    *(p + i) = (int *)malloc(sizeof(int));
    /*@ assert Eva: initialization: \initialized(p + i); */
    __e_acsl_temporal_store_nblock((void *)(p + i),(void *)*(p + i));
    /*@ assert \valid(*(p + i)); */
    {
      int __gen_e_acsl_initialized;
      int __gen_e_acsl_and;
      __gen_e_acsl_initialized = __e_acsl_initialized((void *)(p + i),
                                                      sizeof(int *));
      if (__gen_e_acsl_initialized) {
        int __gen_e_acsl_valid_read;
        int __gen_e_acsl_valid_2;
        __gen_e_acsl_valid_read = __e_acsl_valid_read((void *)(p + i),
                                                      sizeof(int *),
                                                      (void *)p,
                                                      (void *)(& p));
        __e_acsl_assert(__gen_e_acsl_valid_read,(char *)"RTE",(char *)"main",
                        (char *)"mem_access: \\valid_read(p + i)",14);
        /*@ assert Eva: initialization: \initialized(p + i); */
        __gen_e_acsl_valid_2 = __e_acsl_valid((void *)*(p + i),sizeof(int),
                                              (void *)*(p + i),
                                              (void *)(p + i));
        __gen_e_acsl_and = __gen_e_acsl_valid_2;
      }
      else __gen_e_acsl_and = 0;
      __e_acsl_assert(__gen_e_acsl_and,(char *)"Assertion",(char *)"main",
                      (char *)"\\valid(*(p + i))",14);
    }
    i ++;
  }
  __e_acsl_temporal_reset_parameters();
  __e_acsl_temporal_reset_return();
  __e_acsl_temporal_save_nreferent_parameter((void *)(p + 2),0U);
  /*@ assert Eva: initialization: \initialized(p + 2); */
  free((void *)*(p + 2));
  __e_acsl_temporal_reset_parameters();
  __e_acsl_temporal_reset_return();
  malloc(sizeof(int));
  /*@ assert ¬\valid(*(p + 2)); */
  {
    int __gen_e_acsl_initialized_2;
    int __gen_e_acsl_and_2;
    __gen_e_acsl_initialized_2 = __e_acsl_initialized((void *)(p + 2),
                                                      sizeof(int *));
    if (__gen_e_acsl_initialized_2) {
      int __gen_e_acsl_valid_read_2;
      int __gen_e_acsl_valid_3;
      __gen_e_acsl_valid_read_2 = __e_acsl_valid_read((void *)(p + 2),
                                                      sizeof(int *),
                                                      (void *)p,
                                                      (void *)(& p));
      __e_acsl_assert(__gen_e_acsl_valid_read_2,(char *)"RTE",(char *)"main",
                      (char *)"mem_access: \\valid_read(p + 2)",20);
      /*@ assert Eva: dangling_pointer: ¬\dangling(p + 2); */
      __gen_e_acsl_valid_3 = __e_acsl_valid((void *)*(p + 2),sizeof(int),
                                            (void *)*(p + 2),(void *)(
                                            p + 2));
      __gen_e_acsl_and_2 = __gen_e_acsl_valid_3;
    }
    else __gen_e_acsl_and_2 = 0;
    __e_acsl_assert(! __gen_e_acsl_and_2,(char *)"Assertion",(char *)"main",
                    (char *)"!\\valid(*(p + 2))",20);
  }
  __retres = 0;
  __e_acsl_delete_block((void *)(& p));
  __e_acsl_memory_clean();
  return __retres;
}


