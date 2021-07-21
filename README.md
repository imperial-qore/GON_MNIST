# GenBPTI

A discriminator is all you need: Self Adversarial Network for Fault Tolerant Distributed Computing.

Using BPTI to create a generative network
- Crossentropy loss works better
- Adding negative (random images) examples helps
- Tanh() is better than relu or prelu

seans - self adversarial generative network. 
Try with only 2 classes: (1, not 1).
Hope that this is better by training time, performance, parameters.

Hypohesis = "a very good discriminator can also act as a generator". This is because a discriminator can not 
only tell me if this is a real image or a fake image, but also tell me which direction i should move along
to make it closer to a real image. 

Up till now, people have been using discirminator only to discriminate (but NN are closed form functions!).

Systems problem: Optimization of federated graph using GANs or scheduling decisions. A pseudo-neighbor of the current state can be obtained by taking that as the initial value + random noise, and apply the self-adversarial generator network. Or anomaly detection.

Pipeline: train disc using real data --> generate images --> train disc using fake data --> repeat.

Visualizations: change of random input to sample. Generated images. Pipeline fig.

Use case: Memory constrained edge devices.

Limitation: Creating real/fake images for each epoch is super slow. So only suitable for simple structured data.
