# Improved Adaptive Gamma Correction
This is an python implementation of the paper "Contrast enhancement of brightness-distorted images by improved adaptive gamma correction." at https://arxiv.org/abs/1709.04427. The purpose of the algorithm is to improve the contrast of the image adaptively.

## System Environment
- Ubuntu 18.04.1 LTS
- Python >= 3.6
- Opencv >= 3.4.1
- Numpy >= 1.14.4

## Gamma Correction Results
![results](docs/imgs/test_results.png)

## Running the tests
```
python IAGCWD.py --input ./input --output ./output
```

## References
Cao, Gang, et al. "Contrast enhancement of brightness-distorted images by improved adaptive gamma correction." Computers & Electrical Engineering 66 (2018): 569-582.
