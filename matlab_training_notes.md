- Generally, anything above 0.7 is pretty crap for training RMSE, not sure yet
  about test RMSE.

# Cross validation:
## All quick to train:

1.1 Linear regression:
- Useless
- Training RMSE: 3.518
- Tested RMSE: 1.9571
1.2 Fine tree:
- Best, pretty accurate
- Training RMSE: 0.37108
- Tested RMSE: 1.7588
1.3 Medium tree:
- Second best, kinda accurate
- Training RMSE: 0.55611
- Tested RMSE: 1.7532
1.4 Coarse tree:
- Not too bad, somewhat accurate, clumps a bit much.
- Training RMSE: 1.089
- Tested RMSE: 1.6709

## All neural networks
- NOTE: I got very different results when I switched to linux for some of the
  neural networks.  Everything except bilayered neural networks performed
  better?

2.1  Narrow Neural netork:
- spastic
- Training RMSE: 2.8962
- Tested RMSE: 3.9665
2.1  Medium Neural netork:
- closer, but still weird
- Training RMSE:  1.4944
- Tested RMSE: 3.7951
2.1  Wide Neural netork:
- Good, very accurate
- Training RMSE:  0.4759
- Tested RMSE: 3.8476
2.1  Bilayered Neural netork:
- Pretty good
- Training RMSE: 0.67125
- Tested RMSE: 7.7016
2.1  Trilayered Neural netork:
- Pretty off
- Training RMSE: 0.7282
- Tested RMSE: 3.4846

# Something else
# No validation


