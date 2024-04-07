#include <omp.h>
#include <stdio.h>

int main() {
    int n = 10;
    int i;

    // Set the number of threads to 3
    omp_set_num_threads(3); // fixing the number of threads not yet created
    
    // Print the number of threads before parallel region
    printf("hilos antes = %d\n", omp_get_num_threads());

    // Parallel region with a parallel for loop
#pragma omp parallel for
    for (i = 0; i < n; i++) {
        // Inside the parallel region, each thread executes this block
        // Print the total number of threads
        printf("hilos = %d\n", omp_get_num_threads());
        // Print the thread number for each thread
        printf("yo soy %d\n", omp_get_thread_num());
    }

    return 0;
}
