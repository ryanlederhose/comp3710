# Generative Adversarial Networks
Realisitic face generation using GANs of the CelebA dataset.

A GAN is a ML model where two neural networks (NN) compete with each other by using deep learning methods to be come more accurate in their predictions. A GAN uses a discriminator and generator model:
* The generator is a CNN used to artificially manufacture outputs that could be mistaked for real images
* The discriminator is used between real training data and manufactured images

If the discriminator rapidly recognises the generated data produced by the generator, the generator suffers a penalty. As this continues, the generator will produce higher quality images.

## Results

### Face Generation
Epoch 0:

![0000](https://github.com/ryanlederhose/comp3710/assets/112144274/4ff34ac7-1832-4263-95b6-423022d12996)

Epoch 5:

![0005](https://github.com/ryanlederhose/comp3710/assets/112144274/3c5647bb-0039-48a1-892a-5e177e24fe81)

Epoch 20:

![0020](https://github.com/ryanlederhose/comp3710/assets/112144274/e890065c-a360-49c2-bad1-89813fafc030)

Epoch 60:

![0060](https://github.com/ryanlederhose/comp3710/assets/112144274/e57baef6-4d98-40ac-b8af-c37acf19ed8b)

### Losses
![discrim_gen_losses](https://github.com/ryanlederhose/comp3710/assets/112144274/2d15990c-ed08-4bf0-a7d6-39535f4dc15e)

![real_fake_scores](https://github.com/ryanlederhose/comp3710/assets/112144274/a3cceb59-7135-4081-bd53-f13f92e03b19)

# References
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
