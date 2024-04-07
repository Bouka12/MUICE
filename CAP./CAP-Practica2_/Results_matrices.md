# Multiplication of matrices [A(m.k) * B(k,n) = C(m,n)]

`practica1_matrices`: The multiplication using nested `for` loops
* m=100, k=100, n=100  -> duration= 25 milliseconds
* m=50, k=50, n=50      -> duration= 2 milliseconds

`practica2_matrices`: The multiplication using the `SSE` instructions
* m=100, k=1100, n=100  -> duration=  7 milliseconds
* m=50, k=50, n=50      -> duration=  1 milliseconds

**Summary Table**

| Practica            | Description                                                | m,n,k      | Duration (milliseconds)    |
|---------------------|------------------------------------------------------------|------------|----------------------------|
| practica1_matrices  | The multiplication of vmatrices using nested for loops     | 100,100,100| 25                         |
|                     |                                                            | 50, 50, 50 | 2                          |

| practica2_matrices  | The multiplication of matrices using the SSE instructions  | 100,100,100| 7                          |
|                     |                                                            | 50,50,50   | 1                          |


