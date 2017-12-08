# CASED-Tensorflow
Tensorflow implementation of [Curriculum Adaptive Sampling for Extreme Data Imbalance](https://www.researchgate.net/publication/319461093_CASED_Curriculum_Adaptive_Sampling_for_Extreme_Data_Imbalance) with **multi GPU** using [*LUNA16*](https://luna16.grand-challenge.org/)

## Preprocessing
```python
> python all_in_one.py
```
* Check `src_root` and `save_path`
## Usage
```python
> python main.py
```
* See `main.py` for other arguments.

## Issue
* *The hyper-parameter information is not listed in the paper, so I'm still testing it.*
* Use ***[Snapshot Ensemble](https://arxiv.org/pdf/1704.00109.pdf)*** (M=10, init_lr=0.1)
* Or Fix learning rate **0.01**

![snapshot](./assests/lr.JPG)
```python
def Snapshot(t, T, M, alpha_zero) :
    """
    t = # of current iteration
    T = # of total iteration
    M = # of snapshot
    alpha_zero = init learning rate
    """

    x = (np.pi * (t % (T // M))) / (T // M)
    x = np.cos(x) + 1

    lr = (alpha_zero / 2) * x

    return lr
 ```

## Idea
### Preprocessing
* Hounsfield
```python
> minHU = -1200
> maxHU = 600
```

* Resample
```bash
> 1mm (In paper, use 1.25mm)
```

### Network Architecture
![network](./assests/network.JPG)

### Algorithm
![framework](./assests/framework.JPG)

## Result
![result2](./assests/result2.JPG)


## Author
Junho Kim / [@Lunit](http://lunit.io/)
