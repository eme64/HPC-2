// Codes for HPCSE course (2012-2017)
#include <queue>
#include <array>
#include <cmath>
#include <string>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <omp.h>
typedef std::size_t size_type;
typedef double value_type;
typedef std::array<value_type,2> coordinate_type;

#ifndef M_PI
#define M_PI (3.14159265358979)
#endif

/*
some speedup, but seemingly only 50% ?
maybe overhead still too big?

------- Home machine:
serial:
SEGMENT_SAMPLES = 10, MAX_ERROR = 1e-05, NUM_THREADS = 1, Result = -3.466662539, Time(s) = 3.850345465
SEGMENT_SAMPLES = 10, MAX_ERROR = 1e-06, NUM_THREADS = 1, Result = -3.466663774, Time(s) = 17.20036047
SEGMENT_SAMPLES = 100, MAX_ERROR = 1e-06, NUM_THREADS = 1, Result = -3.466663749, Time(s) = 23.11093478
parallel:
SEGMENT_SAMPLES = 10, MAX_ERROR = 1e-05, NUM_THREADS = 8, Result = -3.466662539, Time(s) = 0.908704482
SEGMENT_SAMPLES = 10, MAX_ERROR = 1e-06, NUM_THREADS = 8, Result = -3.466663774, Time(s) = 7.332222433
SEGMENT_SAMPLES = 100, MAX_ERROR = 1e-06, NUM_THREADS = 8, Result = -3.466663749, Time(s) = 5.348561931
speedup: 3.850345465 / 0.908704482 = 23.11093478 / 5.348561931 = 4.3

------- Euler:

*/

// #define PRINT_SEGMENTS
std::ostream& operator<<(std::ostream& os, const coordinate_type& x)
{
    return os << "(" << x[0] << "," << x[1] << ")";
}


/// 2-dimensional trapezoidal rule for rectangular integration region [a,b]
template<class F>
value_type integrate(F f, const coordinate_type& a, const coordinate_type& b, size_type n)
{
    coordinate_type dx = {{(b[0]-a[0])/value_type(n), (b[1]-a[1])/value_type(n)}};

    // four corners *1
    value_type s = f(a) + f(b);
    coordinate_type x;
    x[0] = a[0]; x[1] = b[1];   s += f(x);
    x[0] = b[0]; x[1] = a[1];   s += f(x);

    // four boundaries *2
    std::array<coordinate_type,4> y = {{a,a,b,b}};
    for( size_type i = 1; i < n; ++i )
    {
        y[0][0] += dx[0]; y[1][1] += dx[1]; y[2][0] -= dx[0]; y[3][1] -= dx[1];
        s += 2*(f(y[0]) + f(y[1]) + f(y[2]) + f(y[3]));
    }

    // inner points *4
    x = a;
    for( size_type i = 1; i < n; ++i )
    {
        x[0] += dx[0];  x[1] = a[1];
        for( size_type j = 1; j < n; ++j )
        {
            x[1] += dx[1];
            s += 4*f(x);
        }
    }
    assert( std::abs(x[0]+dx[0]-b[0]) + std::abs(x[1]+dx[1]-b[1]) < 1e-12 );

    return 0.25*dx[0]*dx[1]*s;
}

/// Oscillating integrand with line singularity x^2+y^3 = 2.25
value_type integrand(const coordinate_type& x)
{
    return sin(M_PI/(x[0]*x[0]+x[1]*x[1]*x[1]-2.25));
}


value_type work(coordinate_type a, coordinate_type b, size_type n, value_type maxerrordensity)
{
    // could parallelize here, but not super worth it?
    value_type value;
    value_type value_star;
    coordinate_type dx = {{b[0]-a[0], b[1]-a[1]}};

    //#pragma omp task shared(value)
    {
      value = integrate(integrand,a,b,n);
    }
    //#pragma omp task shared(value_star)
    {
      value_star = integrate(integrand,a,b,n/2);
    }

    //#pragma omp taskwait

    value_type errordensity = std::abs((value - value_star)/dx[0]/dx[1]);

    // subdivide into quarters by halving along each dimension
    if( errordensity > maxerrordensity )
    {
        double r1, r2, r3, r4;
        coordinate_type center = {{0.5*(a[0]+b[0]), 0.5*(a[1]+b[1])}};
        coordinate_type a3 = {{a[0],center[1]}}, b3 = {{center[0],b[1]}};
        coordinate_type a4 = {{center[0],a[1]}}, b4 = {{b[0],center[1]}};

        // must parallelize these
        #pragma omp task shared(r1)
        {
          r1 = work(a, center, n, maxerrordensity);
        }
        #pragma omp task shared(r2)
        {
          r2 = work(center, b, n, maxerrordensity);
        }
        #pragma omp task shared(r3)
        {
          r3 = work(a3, b3, n, maxerrordensity);
        }

        //#pragma omp task shared(r4)
        //{
          r4 = work(a4, b4, n, maxerrordensity); // compute locally !
        //}

        #pragma omp taskwait

        value = r1+r2+r3+r4;
    }

#ifdef PRINT_SEGMENTS
    std::cout << "Segment [" << a << "," << b << "]\tvalue=" << value << "\terror density=" << errordensity << std::endl;
#endif

    return value;
}


int main(int argc, const char** argv)
{
    // read integration parameters from command line
    // pass nthreads by command line for a parallel version
    if( argc != 3 )
        throw std::runtime_error(std::string("usage: ")+argv[0]+" SEGMENT_SAMPLES MAX_ERROR");
    size_type segment_samples = std::stoul(argv[1]);
    value_type max_error      = std::stod (argv[2]);
    size_type nthreads        = omp_get_max_threads();
    std::cout.precision(10);

    value_type result;

    // full integration volume: [-1,1]^2
    coordinate_type a = {{-1,-1}};
    coordinate_type b = {{1,1}};
    coordinate_type dx = {{b[0]-a[0], b[1]-a[1]}};

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
      #pragma omp single nowait
      {
        #pragma omp task shared(result)
        {
          result = work(a, b, segment_samples, std::abs(max_error/dx[0]/dx[1]));
        }
      }
    }
    double t1 = omp_get_wtime();

    std::cout << "SEGMENT_SAMPLES = " << segment_samples << ", MAX_ERROR = " << max_error
              << ", NUM_THREADS = " << nthreads << ", Result = " << result << ", Time(s) = " << t1-t0  << std::endl;

    return 0;
}
