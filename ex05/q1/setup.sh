if [ "$1" == "Darwin" ]; then
    export INTEL_ROOT="/opt/intel/compilers_and_libraries_2018.1.126/mac"
    export INTEL_LIB="compiler/lib"
    export INTEL_MKLLIB="mkl/lib"

elif [ "$1" == "Euler" ]; then
    module load new parallel_studio_xe/2018.0
    export INTEL_ROOT="/cluster/apps/intel/parallel_studio_xe_2018_r0/compilers_and_libraries_2018.0.128/linux"
    export INTEL_LIB="compiler/lib/intel64"
    export INTEL_MKLLIB="mkl/lib/intel64"
fi