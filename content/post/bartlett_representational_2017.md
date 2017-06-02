---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: 2017-05-02
title: Representational and optimization properties of Deep Residual Networks
tags: ["deep-learning", "deep-residual-networks", "optimization", "approximation"]

paper_authors: ["Bartlett, Peter"]
paper_key: bartlett_representational_2017
---

**tl;dr:** Sufficiently regular functions (roughly: having Lipschitz,
invertible derivatives) can be represented as compositions of
increasingly small perturbations of the identity. Furthermore,
critical points of the quadratic loss for these target functions are
proven to be always minima, thus ensuring loss-reducing gradient
descent steps. This makes this class of functions "easily"
approximable by Deep Residual Networks.

{{< youtube UlnYEWXoxOY >}}

Deep networks are deep compositions of non-linear functions 

\\[ h = h\_m \circ h\_{m - 1} \circ \ldots \circ h\_1 . \\]

Depth provides effective, parsimonious representations of features and
nonlinearities provide better rates of approximation (known from
representation theory). Even more, shallow repesentations of some
functions are necessarily more complex than deeper ones (*no
flattening theorems*). But optimization is hard with many layers:
conventional DNNs show increasingly poorer performance at the same
task with growing depth, even though they can approximate a strictly
larger set of functions.

**D**eep **R**esidual **N**etworks[^3] overcome this limitation by
introductig skip-paths connecting the input of layer $j$ to the output
of layer $j+1$:

\\[ f\_{j + 1} (x\_j) = w\_{j + 1} \sigma (w\_j x\_j) + x\_j, \\]

where $\sigma$ is typically a ReLU.[^4]

**Why?** First consider the following fact about linear maps:

> **Theorem 1:**[^1] Any $A$ with $\det A > 0$ can be written as a product
> of perturbations of the identity with decreasing norm, i.e.
> 
> \\[ A = (I + A\_m) \cdots (I + A\_1)\\]
> 
> with the spectral norm of each perturbation fulfilling $\| A_i \|
> =\mathcal{O} (1 / m)$.

Now, for linear Gaussian models $y = Ax + \varepsilon$ with
$\varepsilon \sim \mathcal{N} (0, \sigma^2 I)$ if one sets to find
$A_i$ which minimize the quadratic loss

$$ \mathbb{E} \| (I + A\_m) \cdots (I + A\_1) x - y \|^2, $$ 

over all $A\_i$ near the identity, i.e. among all $ \| A\_i \| < 1$,
then it can be shown that *every stationary point is a global
optimum*, that is: if for every $i$ one has $\nabla_{A\_i} \mathbb{E}
\ldots = 0$, then $A = (I + A\_m) \cdots (I + A\_1)$. Note that this a
property of *stationary points in this region*, it does not say
that one can attain these points by some particular instance of
gradient descent.

The goal of the talk is to show that similar staments hold in the
non-linear case as well. The **main result** is that, under some
conditions detailed below:

> **Result 2**: The computation of a "*smooth invertible map*" $h$ can
> be spread throughout a deep network
>
> $$ h = h\_m \circ h\_{m - 1} \circ \ldots \circ h\_1, $$
>
> so that all layers compute near-identity functions:
>
> $$ \| h\_i - Id \|\_L =\mathcal{O} \left( \frac{\log m}{m} \right)
> . $$
>
> [Where the $\| f \|\_L$ semi-norm is the optimal Lipschitz constant
> of $f$.]

How does this relate to DRNs? Recall that a standard unit of the kind
$A\_i \sigma (B\_i x)$ is known to be a universal approximator so it
can approximate the Lipschitz perturbation to the identity of the
target $h\_i$. In other words, the $h\_i$ can be approximated by units
of DRNs of the form

\\[ \hat{h}\_i(x) = x + A\_i \sigma (B\_i x), \\]

with the perturbations "becoming flatter and flatter" with $i$. This
means that DRNs allow for compositional representation of functions
where terms are increasingly "flat" as one goes deeper. Note however
that this is only proved for functions which are invertible and
differentiable, with Lipshitz derivative and inverse.

The functions $h\_i$ of the theorem can be explicitly constructed via
adequate scalings $a\_1, \dots, a\_m \in \mathbb{R}$ such that:

\\[ h\_1 (x) = h (a\_1 x) / a\_1, h\_2 (h\_1 (x)) = h (a\_2 x) / a\_2, \ldots,
h\_m \circ \cdots \circ h\_1 (x) = h (a\_m x) / a\_m, \\]
   
and the $a\_i$ small enough that $h\_i \approx Id$.

Analogously to the linear case, for the class of functions which may
be represented as such nested, near-identity compositions of maps,
**stationary points of the risk function**

\\[ Q (h) = \frac{1}{2} \mathbb{E} \| h (X) - Y \|\_2^2 \\]

are global minima.[^2] Then

> **Theorem 3**: for any function
>
> \\[ h = h\_m \circ h\_{m - 1} \circ \ldots \circ h\_1, \\]
>
> where $\| h\_i - Id \|\_L \leqslant \varepsilon < 1$, it holds that 
> for all $i$:
>
> \\[ \| D\_{h\_i} Q (h) \| \geqslant \frac{(1 - \varepsilon)^{m -
> 1}}{\| h - h^{\ast} \|} (Q (h) - Q (h^{\ast})) . \\]

This means that if we start with any $h$ in this class of functions near
the identity which is *suboptimal* (i.e. $Q (h) - Q (h^{\ast}) > 0$), then the
(Fréchet) gradient is bounded below and a gradient descent step can be
taken to improve the risk.

Note that this is in the whole space of such nested functions, not in
the particular parameter space of some instance $\hat{h}$: it can
happen that the optimal direction in the whole space is "orthogonal"
to the whole subspace allowed by changing weights in the layers of
$\hat{h}$. Or it can (and will!) happen that there are local minima
among all possible parametrizations of any layer $\hat{h}$. The
following statement remains a bit unclear to me:

> We should expect suboptimal stationary points in the ReLU or sigmoid
> parameter space, but these cannot arise because of interactions
> between parameters in different layers; they arise only within a
> layer.

Basically: if we are able to optimize in the space of architectures,
we should always be able to improve performance (assuming invertible
$h$ with Lipschitz derivative and so on)


[^1]: {{< cite hardt_identity_2016 >}}.

[^2]: Recall that the minimizer of the risk with quadratic loss is the $L^2$ projection, i.e. the conditional expectation $h^{\ast} (x) =\mathbb{E} [Y|X = x]$.

[^3]: {{< cite he_deep_2016 >}}.

[^4]: {{< cite glorot_deep_2011 >}}.
