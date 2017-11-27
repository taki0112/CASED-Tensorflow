# CASED-Tensorflow
Tensorflow implementation of [Curriculum Adaptive Sampling for Extreme Data Imbalance](https://www.researchgate.net/publication/319461093_CASED_Curriculum_Adaptive_Sampling_for_Extreme_Data_Imbalance) with **multi GPU** using [*LUNA16*](https://luna16.grand-challenge.org/)

## Usage
```python
> python main.py
```
* See `main.py` for other arguments.

## Idea
### Preprocessing
* Hounsfield
```python
minHU = -1000
maxHU = 400
```

* Resample
```bash
1.25mm
```

### Network Architecture
![network](./assests/network.JPG)

### Algorithm
![framework](./assests/framework.JPG)

## Result
![result2](./assests/result2.JPG)


## Author
Junho Kim
