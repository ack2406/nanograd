# nanograd

a tiny autograd engine

inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)

#

run examples using [`uv`](https://docs.astral.sh/uv/)

```bash
uv run examples/<example_file>
```

#


to train a simple MLP on the XOR problem run

```bash
uv run examples/xor.py
```

expected output

```txt
Training on 4 examples

Epoch [    0/10000] | Loss [0.770820]
Epoch [ 1000/10000] | Loss [0.641812]
Epoch [ 2000/10000] | Loss [0.526870]
Epoch [ 3000/10000] | Loss [0.324753]
Epoch [ 4000/10000] | Loss [0.173265]
Epoch [ 5000/10000] | Loss [0.104161]
Epoch [ 6000/10000] | Loss [0.071048]
Epoch [ 7000/10000] | Loss [0.052791]
Epoch [ 8000/10000] | Loss [0.041535]
Epoch [ 9000/10000] | Loss [0.034008]
Epoch [10000/10000] | Loss [0.028666]

Raw results:
Tensor([[0.02753704]
 [0.9720824 ]
 [0.9666618 ]
 [0.02420251]])

Rounded results:
Tensor([[0.]
 [1.]
 [1.]
 [0.]])
```