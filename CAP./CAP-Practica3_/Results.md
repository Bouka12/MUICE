# Results of Practica_
**Output**
hilos antes = 1
hilos = 3
yo soy 1
hilos = 3
yo soy 1
hilos = 3
yo soy 1
hilos = 3
yo soy 2
hilos = 3
yo soy 2
hilos = 3
yo soy 2
hilos = 3
yo soy 0
hilos = 3
yo soy 0
hilos = 3
yo soy 0
hilos = 3
yo soy 0

# Results of Practica_

* **Output:**
Thread 1: a,b,x= 1 1 2.100000
Thread 0: a,b,x= 0 0 1.000000
Thread 1: a,b,x= 1 0 2.100000
Thread 0: a,b,x= 0 0 1.000000
* **Explanation:**
Each line represents the output produced by a thread when executing the program.
1. Thread 1: a,b,x= 1 1 2.100000: This line indicates that thread 1 is executing, and it shows the values of variables a, b, and x. In this case, a and b are both set to 1 (which corresponds to the thread number), and x is calculated as 1.1 * tid + 1.0, resulting in 2.1 for thread 1.

2. Thread 0: a,b,x= 0 0 1.000000: This line indicates that thread 0 is executing, and it shows the values of variables a, b, and x. In this case, a and b are both set to 0 (which corresponds to the thread number), and x is calculated as 1.1 * tid + 1.0, resulting in 1.0 for thread 0.

3. Thread 1: a,b,x= 1 0 2.100000: This line again shows the output of thread 1. The value of a remains 1, as it is thread-private. However, the value of b has been modified to 0, which is different from the previous line, indicating that b is not thread-private. The value of x remains the same (2.1), as it is also thread-private.

4. Thread 0: a,b,x= 0 0 1.000000: This line again shows the output of thread 0. Both a and b remain 0, and x remains 1.0.

From this output, you can see the effects of #pragma omp threadprivate(a, x) where a and x are thread-private, meaning each thread gets its own copy of these variables. However, b is not specified as thread-private, so it's shared among threads, leading to the observed behavior where its value changes between threads.

# Results of Practica3_Construccion
**OUtput**
Itr 0 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=1 a=2
Itr 1 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=1 a=2
Itr 2 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=1 a=2
Itr 3 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=1 a=2
Itr 4 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=1 a=2
Itr 5 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=0 a=2
Itr 6 ------------------>
Section 2 - Thread id=0 a=2
Section 1 - Thread id=1, a=1
Itr 7 ------------------>
Section 1 - Thread id=0, a=1
Section 2 - Thread id=0 a=2
Itr 8 ------------------>
Section 1 - Thread id=0, a=1
Section 2 - Thread id=0 a=2
Itr 9 ------------------>
Section 1 - Thread id=0, a=1
Section 2 - Thread id=0 a=2

# Results of Practica3_Construccion_2

**Output**
Itr 0 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=0 a=-64245759
Itr 1 ------------------>
Section 1 - Thread id=1, a=1
Section 2 - Thread id=1 a=2
Itr 2 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758
Itr 3 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758
Itr 4 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758
Itr 5 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758
Itr 6 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758
Itr 7 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758
Itr 8 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758
Itr 9 ------------------>
Section 1 - Thread id=0, a=-64245759
Section 2 - Thread id=0 a=-64245758


# Practica3_paraleliza_practica1

**OUtput**
Enter the size of the vectors (N): 12
Enter the value of beta: 0.3
Elapsed time: 0.000983869 seconds

Enter the size of the vectors (N): 100
Enter the value of beta: 0.3
Elapsed time: 0.000892038 seconds