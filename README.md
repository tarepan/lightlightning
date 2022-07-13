<div align="center">

# light Lightning : the "Lightning" ML wrapper <!-- omit in toc -->

</div>

Light-weight wrapper of PyTorch Lightning (Machine Learning framework based on PyTorch).  

Current APIs:  

- `ll.train` & `ll.ConfTrain`: Common training workload (Mixed precision training on fast accelerator)

## Install
```bash
pip install git+https://github.com/tarepan/lightlightning.git
```

## Usage
```python
import lightlightning as ll

model, conf, datamodule = ...
ll.train(model, conf, datamodule)
```
