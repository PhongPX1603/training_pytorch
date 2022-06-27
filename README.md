# Training Template

# 1. Structure

# 2. Usage
* Training
```bash
python -m core config/MNIST/mnist_training.yaml --num-epoch <x> --num-gpus <y>
```

* Retrain
```bash
python -m core config/MNIST/mnist_training.yaml --num-epoch <int> --num-gpus <int> --checkpoint-path <str>
```

* Resume
```bash
python -m core config/MNIST/mnist_training.yaml --num-epoch <int> --num-gpus <int> --resume-path <str>
```

* Testing
```bash
python -m core config/MNIST/mnist_testing.yaml --num-gpus <int>
```
