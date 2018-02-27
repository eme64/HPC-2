#include "q1_integrands.h"
#include <math.h>

double integrand_1(const double x) {
    // TODO: Implement the first integrand:
    return -(x-1.0)*(x+1)*(x+1);
    // = -(x-1)(x²+2x+1)
    // = -x³-2x²-x+x²+2x+1
    // = -x³-x²+x+1
}

double integrand_2(const double x) {
    // TODO: Implement the second integrand:
    return -exp(-x);
}

double analytic_solution_1() {
    // TODO: Implement the analytic solution for the first integral:
    // -0.25*x^4-1/3*x^3+1/2*x^2+x
    return (-2.0/3.0+2.0);
}

double analytic_solution_2() {
    // TODO: Implement the analytic solution for the first integral:
    // exp(-x)
    return 0-1.0;
}
