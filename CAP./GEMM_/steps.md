 follow these steps:

Understand the Structure:

gemm_driver.c is likely the main file that you need to modify and extend to parallelize the matrix multiplication.
Makefile is used to compile the project.
my_gemm.c is probably where you will implement your parallelized matrix multiplication function.
run_driver.sh is likely a script to execute the compiled program with some parameters.
lib folder may contain libraries or additional code required for the project.
include folder may contain header files required for the project.
build folder might be used to store compiled object files or the final executable.
Implement Parallel Matrix Multiplication:

Open my_gemm.c and implement the parallelized version of the matrix multiplication function, which is likely the gemm_op function in your case. You already have the parallel version provided.
Make sure to include necessary header files and any additional dependencies.
Modify gemm_driver.c:

Update gemm_driver.c to include your parallelized matrix multiplication function from my_gemm.c.
You might need to adjust any function calls or parameters to fit the structure of gemm_driver.c.
Compile the Project:

Use the Makefile to compile the project. If necessary, update the Makefile to include your new source files.
Execute make command in the terminal to compile the project. This should create an executable file.
Test the Program:

Run the run_driver.sh script or execute the compiled program directly to test the functionality.
Ensure that the program executes without errors and produces correct results.
Verify Parallelization and Performance:

Verify that the parallelized version of matrix multiplication is functioning correctly and providing improved performance compared to the sequential version.
You can check the performance metrics provided by the gemm_driver.c or use other profiling tools to measure the speedup achieved by parallelization.
Create the Compressed Document:

Once you have verified that the program works as expected, compress all the relevant files (source code, Makefile, etc.) into a single zip archive.
Ensure that the compressed document includes all necessary files and directories required to compile and run the program.
Submission:

Submit the compressed document as per the requirements of your assignment or task.