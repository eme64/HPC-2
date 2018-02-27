#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "q1_trapz.h"
#include "q1_mc.h"

int main(int argc, char *argv[]){
    if(argc != 2){
        fprintf (stderr, "Wrong number of parameters. The syntax is: main [trapz, mc]\n");
        exit(-1);
    }

    if(strcmp(argv[1], "trapz") == 0)
        q1_trapz();
    else if(strcmp(argv[1], "mc") == 0)
        q1_mc();

    return 0;
}