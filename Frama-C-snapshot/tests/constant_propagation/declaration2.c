/* run.config
   OPT: -eva @EVA_OPTIONS@ -scf -journal-disable
*/

void f(int *x) { (*x)++; }

int main () {
  int Y = 42;
  f(&Y);
  return Y;
}
