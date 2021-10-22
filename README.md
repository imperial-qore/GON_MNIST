# Using GON for hand-written digit generation

Hypothesis: "A discriminator is all you need". A sufficiently trained discriminator could not only indicate whether an input belongs to a data distribution but also how to tweak the input to make it resemble more closely to the target distribution. Thus, we can use only a discriminator for data generation and use those as fake samples in a self-adversarial training fashion. This allows us to reduce the parameter size significantly compared to traditional GANs.

Thus, a good discriminator can also act as a generator. This is because a discriminator can not 
only tell me if this is a real image or a fake image, but also tell me which direction i should move along
to make it closer to a real image.  Up till now, people have been using discirminator only to discriminate, but neural networks are closed form functions, allowing us to backpropagate gradients to input and run optimization in the input space. The only limitation of GON is that it is slower to train that GANs due to the optimization loop in each epoch.

In this repo, show the broader impact of GONs on how they can be used to generate data from any distribution (assuming sufficient training times). Specifically, we use GONs to generate hand-written digits by training on the MNIST dataset.

The performance of the model can be visualized from the following figure that shows generated zeros in a few-shot setting.

![Alt text](save.png?raw=true "results")


## Installation
This code needs Python-3.7 or higher.
```bash
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

## Generating digits with GONs
Change the digit you want to generate by replacing the `CLASS` variable. Then run the following command
```bash
python3 main.py mnist train
```

Some insights.
- Crossentropy loss works better than MSE.
- Adding negative (random images) examples helps.
- Tanh() is better than relu or prelu.

## Arxiv preprint
https://arxiv.org/abs/2110.02912.

## Cite this work
Our work is published in NeurIPS 2021, Workshop on ML for Systems. The main GitHub repo for this work is here: https://github.com/imperial-qore/GON.
```bibtex
@article{tuli2021generative,
  title={Generative Optimization Networks for Memory Efficient Data Generation},
  author={Tuli, Shreshth and Tuli, Shikhar and Casale, Giuliano and Jennings, Nicholas R},
  journal={Advances in Neural Information Processing Systems, Workshop on ML for Systems},
  year={2021}
}

```

## License

BSD-2-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
