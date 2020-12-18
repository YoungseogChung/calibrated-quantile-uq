# Quantile Methods for Calibrated Uncertainty Quantification
This is the repo for the paper [Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification](https://arxiv.org/abs/2011.09588).

The core algorithm aims to learn the conditional quantiles of the data distribution by 
optimizing directly for calibration, sharpness, and adversarial group calibration.


## Basic Usage
```python
python main.py --loss scaled_batch_cal
```

There are currently 4 losses implemented: ["calibration loss"](https://arxiv.org/abs/2011.09588), 
["scaled calibration loss"](https://arxiv.org/abs/2011.09588),
["interval score"](https://arxiv.org/abs/2011.09588), and "qr", which is simply the pinball loss.
The versions of the losses with the prefix "batch" means the code for calculating the loss is  
vectorized.




## Bibliography
If you find this work useful, please consider citing the corresponding paper:
```
@article{chung2020beyond,
  title={Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification},
  author={Chung, Youngseog and Neiswanger, Willie and Char, Ian and Schneider, Jeff},
  journal={arXiv preprint arXiv:2011.09588},
  year={2020}
}
```

## Related Repo
This work led to the development of a general uncertainty quantification repo, called
[Uncertainty-Toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox).
This toolbox evaluates a quantification of uncertainty based on a suite of metrics, including calibration,
sharpness, adversarial group calibration, proper scoring rules, and accuracy.
