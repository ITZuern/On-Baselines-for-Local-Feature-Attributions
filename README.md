Source code for the experiments from the paper 'On Baselines for Local Feature Attributions' by Johannes Haug, Stefan Zürn, Peter El-Jiz and Prof. Gjergji Kasneci.

Computation Time (seconds). We exhibit the average computation times for generating local attribution scores on a test set (20\% of observations) from every data set. Strikingly, the stochastic baselines are significantly slower, since they involve sampling. The lowest computation time per attribution model is highlighted in bold. All experiments were conducted on an NVIDIA GeForce GTX 1050 TI GPU with Intel i5 7500 CPU and 16Gb RAM. Our machine ran Linux Fedora 32 and Python 3.7.6.

| Attribution Model | DeepSHAP | DeepLIFT | IG    | KernelSHAP |
|-------------------|----------|----------|-------|------------|
| Constant          | 5.90     | **0.11**     | 2.17  | 81.74      |
| Maximum Distance  | **17.81**    | 29.30    | 26.36 | 101.80     |
| Blurred           | 66.77    | **0.80**     | 19.43 | 324.65     |
| Uniform           | 66.67    | **0.82**     | 19.74 | 327.71     |
| Gaussian          | 74.05    | **0.80**     | 19.60 | 327.86     |
| Expectation       | 69.47    | **0.80**     | 19.81 | 323.35     |