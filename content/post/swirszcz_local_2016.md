---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: 2017-05-09
title: "Local minima in training of neural networks"
tags: ["deep-learning", "optimization"]

paper_authors: ["Swirszcz, Grzegorz", "Czarnecki, Wojciech Marian", "Pascanu, Razvan"]
paper_date: 2017-02-17
paper_key: swirszcz_local_2016
---

**tl;dr**: The goal is to construct elementary examples of datasets
such that some neural network architectures get stuck in very bad
local minima. The purpose is to better understand why NNs seem to work
so well for many problems and what it is that makes them fail when
they do. The authors conjecture that their examples can be generalized
to higher dimensional problems and therefore that the good learning
properties of deep networks rely heavily on the structure of the
data.[^1]

---

**Literature review**: Besides the examples themselves, the authors
provide a valuable review, citing, among others:

> In particular Fyodorov & Williams (2007); Bray & Dean (2007), for
> random Gaussian error functions (…) all points with a low index
> \[number of negative eigenvalues of the Hessian\] (note that every
> minimum has this index equal to 0) have roughly the same performance,
> while critical points of high error implicitly have a large number of
> negative eigenvalue which means they are saddle points.

> The claim of Dauphin et al. (2013) is that the same structure holds
> for neural networks as well, when they become large enough.

> Goodfellow et al. (2016) argues and provides some empirical evidence
> that while moving from the original initialization of the model along
> a straight line to the solution (found via gradient descent) the loss
> seems to be only monotonically decreasing, which speaks towards the
> apparent convexity of the problem. Soudry & Carmon (2016); Safran &
> Shamir (2015) also look at the error surface of the neural network,
> providing theoretical arguments for the error surface becoming
> well-behaved in the case of overparametrized models.

> A different view, presented in Lin & Tegmark (2016);[^1]
> Shamir (2016), is that the underlying easiness of optimizing deep
> networks does not simply rest just in the emerging structures due to
> high-dimensional spaces, but is rather tightly connected to the
> intrinsic characteristics of the data these models are run on.

**Contributions**: In Theorem 1 they construct what they conjecture to
be the smallest dataset (10 points) such that a 2-2-1 fully connected
NN with sigmoid activations is “deadlocked” into a local minimum with
an accuracy significantly below the optimum (50% that of another point
they explictly show).

In Section 3.2 they provide 3 examples for a single layer network with
ReLUs for regression. Note that the region in input space where each
unit saturates (the “blind spot” where the gradient vanishes) is the
whole $\mathbb{R}^-$. However they are able to devise examples showing
that:

> (…) blind spots are not the only reason a model can be stuck in a
> suboptimal solution. Even more surprisingly, (…) blind spots can be
> completely absent in the local optima, while at the same time being
> present in the global solution.

Essentially the construction is based on the idea that

> (…) if, due to initial conditions, the model partitions the input
> space in a suboptimal way, it might become impossible to find the
> optimal partitioning using gradient descent.

Crucially, they conjecture that this idea can be non-trivially
**generalized to more interesting higher dimensional problems**.

Finally, in Section 4 they construct a dataset for regression with a
bad local minimum, based on the observation that, since the dataset is
necessarily finite, it is possible to

> (…) compute conditions for the weights of any given layer of the
> model such that for any datapoint all the units of that layer are
> saturated \[and learning stops\]. Furthermore, we show that one can
> obtain a better solution than the one reached from such a state.

[^1]: See {{< cite lin_why_2016 >}}.
