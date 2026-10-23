| Case | 4090 µs | 4090 MFU % | 5090 raw µs† | 5090 check |
|---|---:|---:|---:|---|
| p00_scalar_add | 3.072 | 0.00005 | 3.152 | PASS |
| p01_vector_add | 3.120 | 0.00040 | 3.104 | PASS |
| p02_scale_shift | 4.096 | 0.00242 | 3.168 | PASS |
| p03_relu_mul | 4.672 | 0.00053 | 3.104 | PASS |
| r01_array_sum | 4.096 | 0.00030 | 3.104 | PASS |
| r02_big_sum | 7.760 | 0.01022 | 8.832 | PASS |
| r03_mean | 6.144 | 0.00646 | 6.784 | PASS |
| r04_row_sum | 3.264 | 0.01213 | 3.088 | PASS |
| m01_matmul_512 | 25.600 | 12.69463 | 21.696 | FAIL |
| m02_matmul_rect | 5.120 | 0.99177 | 4.992 | PASS |
| f01_rmsnorm | 5.120 | 0.06200 | 4.768 | PASS |
| f02_row_softmax | 3.072 | 0.01935 | 3.104 | PASS |
| f03_l2norm | 4.096 | 0.02904 | 3.184 | PASS |
| a01_attention | 18.432 | 0.06994 | 19.072 | PASS |
| a02_attention_128 | 23.552 | 0.11116 | 25.200 | PASS |
