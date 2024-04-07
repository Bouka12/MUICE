# Practica 3: OpenMP
* Tema numero 3 de OpenMP
* Asignatura: CAP
* Master: MUICE
* Estudiante: Mabrouka Salmi
* Fecha: 02/04/2024



## 1. CODE : Numero e Identificador de los Hilos

In this code  `practica3_Numero-e-Identificador-de-los-Hilos.cpp`

`omp_set_num_threads(3)` sets the number of threads to 3 for the parallel region.
`omp_get_num_threads()` retrieves the total number of threads in the parallel region.
`omp_get_thread_num()` retrieves the thread number for each thread.

## 2. CODE : Variables threadprivate

In this code `practica3_Variables_threadprivate.cpp`:

`#pragma omp threadprivate(a, x)` specifies that the variables a and x should be private to each thread.
`omp_set_dynamic(0)` turns off dynamic threads explicitly, meaning the number of threads used in the parallel region is determined by the program and not dynamically changed.
The first parallel region sets values for `a`, `b`, and `x` for each thread and prints them out.
The second parallel region prints out the values of `a`, `b`, and `x` from the first parallel region. Since a and x are thread-private, each thread will print the value it set during the first parallel region. However, `b` is private to each thread in the first parallel region, so its value will not be shared or consistent across threads in the second parallel region.

## 3. CODE :  Contruccion
In this code `practica3_Construccion.cpp`

* The `main()` function contains a loop that runs 10 times `(for (int i = 0; i < 10; i++))`.
* Inside the loop, a variable a is initialized to 0.
* The loop contains a parallel sections construct (`#pragma omp parallel sections`), which divides the following code into sections that can be executed concurrently.
* Inside the parallel sections, there are two sections defined (#pragma omp section). Each section increments the variable `a` by 1 and prints the thread ID and the value of a (`thid = omp_get_thread_num()`; `printf("Section X - Thread id=%d, a=%d\n", thid, a);`).
* Each iteration of the loop will execute the sections in parallel, with potentially different threads executing each section.

## 4. CODE  : Construccion_2
In this code `practica3_Construccion_2.cpp`

* `#pragma omp parallel sections private(a, thid)` declares a parallel sections region, where each section is executed by a separate thread. `private(a, thid)` ensures that each thread has its own private copy of the variables a and thid.

* Inside each section, `a` is incremented by 1 and the thread ID (thid) is obtained using `omp_get_thread_num()`. The values of a and thid are then printed out.

* The loop `for (int i = 0; i < 10; i++)` runs for 10 iterations, and the sections are executed in parallel for each iteration of the loop.

## 5. CODE : Paraleliza el ejercicio de la Practica 1

Here we used OpenMP to parallelize the operation `C = A + B + C*beta`. where `beta` and and the size of vectors `A`, `B`, and `C` are user-input defined
 * `#pragma omp parallel for` is used to parallelize the sum of vectors.


