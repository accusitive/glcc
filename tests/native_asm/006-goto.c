#include <stdio.h>
int main(void) {
    int hits = 0;
    asm goto ("cmpl $1, %0; je %l[yes]" :: "r"(1) :: yes);
    goto done;
yes:
    hits++;
done:;
    asm goto ("jmp %l[l2]" : : : : l1, l2);
    goto out;
l1:
    hits += 10;
    goto out;
l2:
    hits += 100;
out:;
    printf("%d\n", hits);
    return 0;
}
