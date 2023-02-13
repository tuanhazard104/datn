#include <stdio.h>
int f(int n) {
    if(n<=1) return 1;
    return 2*f(n-3)+4*f(n-2);
}
int main() {
    printf("%d", f(5));
}