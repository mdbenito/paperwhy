---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: 2017-05-10
title: Greedy layer-wise training of Deep Networks 
tags: ["deep-belief-networks", "boltzmann-machines"]

paper_authors: ["Bengio, Yoshua", "Lamblin, Pascal", "Popovici, Dan", "Larochelle, Hugo"]
paper_date: 2007
paper_key: bengio_greedy_2007
---

Back in the dark days of 2006, neural networks were not properly
initialised (no batchnorm), not properly regularised (no
**dropout**,[^1] no **maxout**[^2]), mostly **still using
sigmoids**[^3], not properly trained (no momentum, no adam, no
wildhog!). Random initialisation of weights often led to poor local
minima. This paper took an idea of Hinton, Osindero, and Teh (2006)
for pre-training of Deep Belief Networks: greedily (one layer at a
time) pre-training in unsupervised fashion a network kicks its weights
to regions closer to better local minima,

>giving rise to internal distributed representations that are
>high-level abstractions of the input, bringing better generalization.

The authors

>performed experiments which support the hypothesis that the greedy
>unsupervised layer-wise training strategy helps to optimize deep
>networks, but suggest that better generalization is also obtained
>because this strategy initializes upper layers with better
>representations of relevant high- level abstractions. These
>experiments suggest a general principle that can be applied beyond
>DBNs, and [they] obtained similar results when each layer is
>initialized as an auto-associator [autoencoder] instead of as an RBM.

After a brief description of DBNs and RBMS, as well as their
optimization with maximum likelihood and Gibbs Markov chains (which
does not belong here), the paper proceeds to a description ofthe
greedy layer-wise training of a DBN:

>Note that a 1-level DBN is an RBM. The basic idea of the greedy
>layer-wise strategy is that after training the top-level RBM of a
>l-level DBN, one changes the interpretation of the RBM parameters to
>insert them in a $(l+1)$-level DBN: the distribution $P(g^{l-1}|g^l)$
>from the RBM associated with layers $l−1$ and $$ is kept as part of
>the DBN generative model.

More technical material follows, pertinent only to the extension of
DBMs to continuous-valued inputs. But:

### Why does the layer-wise strategy work?

It seems that weights are set to better initial values. Therefore it
makes sense to attempt the same scheme with simpler networks, like
autoencoders. Indeed the experiments showed this to be the case and
the authors report high quality test errors comparable to those of the
DBNs on the dataset studied (MNIST), when supervised training is done
after the unsupervised pre-training. However: what stops an
autoencoder layer from just learning the identity function? Is it then
necessary to always decrease the size of the layers? The answer is in
the negative:

>our experiments suggest that networks with non-decreasing layer sizes
>generalize well. This might be due to weight decay and stochastic
>gradient descent, preventing large weights: optimization falls in a
>local minimum which corresponds to a good transformation of the input
>(that provides a good initialization for supervised training of the
>whole net).

There is a subtlety here: if the top layers are big enough, training
error can be zero even without pretraining, which was supposed to aid
optimization. The key is the generalisation error, which will be worse
since the lower layers will be poorly initialised. An experiment with
reduced top layer size shows this to be a plausible explanation.

>Consider the top two layers of the deep network with pre-training: it
>presumably takes as input a better representation, one that allows
>for better generalization. Instead, the network without pre-training
>sees a “random” transformation of the input, one that preserves
>enough information about the input to fit the training set, but that
>does not help to generalize.

It seems that indeed pretraining might initialize

>the hidden layers so that they represent more meaningful
>representations of the input, which also yields to better
>generalization.

Finally the authors propose a semisupervised pre-training step for
tasks with input distributions unrelated to the function to learn,
e.g. in regression $Y=f(X)+\epsilon$ with $X \tilde p$ and no relation
between $p$ and $f$:

>In such settings we cannot expect the unsupervised greedy layer-wise
>pre-training procedure to help (…) we propose to train each layer
>with a mixed training criterion that combines the unsupervised
>objective (modeling or reconstructing the input) and a supervised
>objective (helping to predict the target).

[^1]: {{< cite hinton_improving_2012 >}}
 
[^2]: {{< cite goodfellow_maxout_2013 >}}

[^3]: {{< cite glorot_deep_2011 >}}
