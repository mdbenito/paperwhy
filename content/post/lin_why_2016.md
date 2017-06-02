---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: 2017-05-13
title:  Why does deep and cheap learning work so well?
tags: ["deep-learning", "foundations"]

paper_authors: ["Lin, Henry W.", "Tegmark, Max"]
paper_date: 2016-09-29
paper_key: lin_why_2016
---

This paper addresses two fundamental questions for deep networks. That
of **efficiency**: why do networks with relatively so few parameters
(millions as opposed to gazillions) work? And that of **depth**: why
do deeper architectures perform better in some tasks? and is it
possible to perform as well with shallower ones?

**tl;dr**: (Hard to judge) physical motivation for the success of
shallow networks as approximators of Hamiltonians. Proof that fixed
size networks can approximate polynomials arbitrarily well and
implication for typical Hamiltonians. Proof that the inference
(reconstruction of initial parameters) of hierarchical / sequential
*Markovian* processes (argued to be pervasive in nature) is learnable
by deep architectures but not by shallower ones (**no-flattening
theorem**).

### Efficiency 

Against the overwhelming size of the search space, neural networks
(NNs) perform remarkably well in tasks where their input stems from
physical processes. The authors argue that this is due to their
ability to approximate the very simple Hamiltonians (oftentimes
polynomials of lower order) that govern naturally phenomena:[^1] Even
though “the set of all possible functions is exponentially larger than
the set of practically possible networks”, the set of functions
interesting to machine learning is vastly simpler.

Consider **image classification**: the training input consists of
pairs $(x\_i,y\_i)$ of image array and class and the desired output
for a test image $x$ is the probability $p(y|x)$ for all possible
classes $y \in Y$, with $\|Y\| \< \infty$. Bayes' theorem states

$$ p (y|x) = \frac{p (x|y) p (y)}{\sum\_{y' \in Y} p (x|y') p (y')} . $$

Setting

\begin{eqnarray}
  H\_y (x) & := & - \ln p (x|y) \hskip3em \text{(Hamiltonian)} \\\\\\
  \mu\_y & := & - \ln p (y) \hskip3em \text{(self-information)} \\\\\\
  N (x) & := & \sum\_{y \in Y} \mathrm{e}^{- [H\_y (x) + \mu\_y]},
\end{eqnarray}

Bayes' theorem transforms into the *Boltzmann form* common in
statistical physics:

$$ p (y|x) = \frac{\mathrm{e}^{- [H\_y (x) + \mu\_y]}}{N (x)} . $$

If we use vector notation, $p (x) := (p (y\_1 |x), \ldots, p (y\_{| Y
|} \|x))$ and analogously for $H$ and $\mu$, we can write

$$ p (x) = \frac{\mathrm{e}^{- [H (x) + \mu]}}{N (x)} = \sigma (- H
(x) - \mu), $$

where $\sigma (x) := \mathrm{e}^x / \sum\_j \mathrm{e}^{x\_j}$ is the
softmax function. Now recall that any feed-forward neural network $f$
can be succintly described as a composition of linear $A\_j$ and
non-linear maps $h\_j$:

$$ f (x) = (h\_n \circ A\_n \circ \cdots \circ h\_1 \circ A\_1) (x) $$

and typically the last non-linearity is chosen to be a soft-max: $h\_n
= \sigma$. The key insight here is that, if the rest of the network,
$A\_n \circ \cdots \circ h\_1 \circ A\_1$, can approximate well the
Hamiltonian $H$, the application of $\sigma$ will yield an accurate
prediction of the desired posterior $p (y|x)$.

It turns out that many relevant Hamiltonians are:

* Polynomials of very low order d (e.g. the standard model of physics
  has $d=4$).
* Invariant under some group of transformations (and thus describable
  with even fewer parameters).
* Limited in degree by locality properties (cf. Ising model).

The main contribution of the paper wrt. the previous idea consists in
the following two results:

>**Theorem:** Let $f$ be a neural network of the form $f = A_2
>\circ \sigma \circ A_1$, where $\sigma$ acts elementwise by applying
>some smooth non-linear function $\sigma$ to each element. Let the
>input layer, hidden layer and output layer have sizes 2, 4 and 1,
>respectively. Then f can approximate a multiplication gate
>arbitrarily well.

>**Corollary:** For any given multivariate polynomial and any
>tolerance $\varepsilon > 0$, there exists a neural network of fixed
>finite size $N$ (**independent of $\varepsilon$**) that approximates
>the polynomial to accuracy better than $\varepsilon$. Furthermore,
>$N$ is bounded by the complexity of the polynomial, scaling as the
>number of multiplications required times a factor that is typically
>slightly larger than 4.

So computing polynomials is cheap in terms of parameters: the accuracy
of the approximation does not have an effect on the size of the
network, in contrast to previous approximation results which could not
rule out an exponential increase in the number of
parameters. Therefore a NN should perform well when approximating
simple Hamiltonians, even with only “few” parameters. One obvious
point of contention is precisely this required simplicity, but:

Although we might expect the Hamiltonians describing human-generated
data sets such as drawings, text and music to be more complex than
those describing simple physical systems, we should nonetheless expect
them to resemble the natural data sets that inspired their creation
much more than they resemble random functions.

It is not yet entirely clear that this is enough...

### Depth

Why is the performance of NNs typically improved when more layers are added?

Many complicated physical processes amount to the concatenation of simpler
transformations. These can be modeled as Markov processes $y\_1 \mapsto y\_2
\mapsto \ldots \mapsto y\_n$. Denote the probability of transitioning to state
$y\_i$ given the last state $y\_n$ by $p (y\_i |y\_n)$. Then

>**Theorem:** Let $T\_i$ be a minimal sufficient statistic of $p (y\_i
>|y\_n)$.[2^] Then there exists some function $f\_i$ such that $T\_i =
>f\_i \circ T\_{i + 1}$.

That is, there are functions $f\_i$ that unravel these hierarchical
processes "backwards in time" without losing any information beyond
that which was lost by the Markov process itself (i.e. that left by
the sufficient statistic).

>**Corollary:** Define $f\_0 (T\_0) = p (y\_0 |T\_0)$ and $f\_n =
>T\_{n - 1}$. Then
>
>$$ p (y\_0 |y\_n) = (f\_0 \circ f\_1 \circ \cdots \circ f\_n) (y\_n) . $$

Which says that “the structure of the inference problem reflects the
structure of the generative process”, meaning that to go backwards in
the generative process, a NN must approximate a composition of
functions, a task at which they excel.

One immediate problem with these results is that sufficient statistics may not
be (easily) computable. However, it is possible to work with "almost
sufficient statistics" $f$ in the sense that they do not preserve the full
mutual information $I$, that is: $I (y|y\_n) > I (y|f (y\_n))$:

> For example, it may be possible to trade some loss of mutual
> information with a dramatic reduction in the complexity of the
> Hamiltonian; e.g., $H\_y (f (x))$ may be considerably easier to
> implement in a neural network than $H\_y (x)$.  Precisely this
> situation applies to (...), where a hierarchy of efficient
> near-perfect information distillers [$f\_0, \ldots, f\_3$] have been
> found, [whose] numerical cost [scales] with the number of inputs
> parameters $n$ as $O (n)$, $O (n^{3 / 2})$, $O (n^2)$ and $O (n^3)$,
> respectively.

However it is not exactly clear, nor quantified, how using these
almost sufficient statistics affects the quality of the approximation
of the prior distribution $p (y\_0 |y\_n)$.

Finally it is not possible to "flatten" these architectures without
losing performance or suffering an exponential increase in the number
of parameters.

### No-flattening theorems

Fix some network $f$ with $l$ layers and an approximation accuracy
$\varepsilon > 0$ for $f$ in some sense. Denote by
$\mathcal{F}^l\_{\varepsilon}$ some family of networks which
approximate $f$ with fewer than $l$ layers and with $\varepsilon$
accuracy. Define the **flattening cost** of $f$ as

$$ C\_n (f, \mathcal{F}^l\_{\varepsilon}) :=  \underset{f' \in
   \mathcal{F}^l\_{\varepsilon}}{\min}  \frac{N\_n (f')}{N\_n (f)} $$
   
for the number of neurons and

$$ C\_s (f, \mathcal{F}^l\_{\varepsilon}) := \underset{f' \in
   \mathcal{F}^l\_{\varepsilon}}{\min}  \frac{N\_s (f')}{N\_s (f)} $$
   
for the number of synapses or non-zero weights. A result where $C\_n > 1$ or
$C\_s > 1$ for some class $\mathcal{F}^l\_{\varepsilon}$ is referred to as a
**no-flattening theorem**.

After presenting a little zoo of results (no-flattening for
polynomials or composition of functions in Sobolev spaces, as well as
exponential neuron-inefficiency for certain architectures) the authors
proceed with their own:

Simple no-flattening example: composition of linear maps. The idea
behind the FFT is to decompose the linear map performing the discrete
Fourier transform into $\log n$ many which are sparse, thus reducing
an $\mathcal{O} (n^2)$ operation to an $\mathcal{O} (n \log n)$. So
even though these simpler maps could be flattened (by multiplication)
into just one, the computational cost would dramatically grow. The
synapse flattening cost (the increase in non-zero weights in the
network) is $C\_s =\mathcal{O} (n / \log n) \approx \mathcal{O} (n)$.

Another example: it has been proved that matrix multiplication can be
performed in roughly $\mathcal{O} (n^{2.37})$ instead of $\mathcal{O}
(n^3)$ by stacking operations in a deep architecture (ref 46 in the
paper).

Finally, polynomials are proven to be exponentially expensive to
flatten:

> **Theorem:** No neural network can implement an $n$-input
> multiplication gate using fewer than $2^n$ neurons in the hidden
> layer.

For example:

> (...) a deep neural network can multiply $32$ numbers using $4 n =
> 160$ [sic] neurons while a shallow one requires $2^{32} = 4, 294,
> 967, 296$ neurons.  Since a broad class of real-world functions can
> be well approximated by polynomials, this helps explain why many
> useful neural networks cannot be efficiently flattened.


### Talk

The first half of the following talk by Tegmark is dedicated to this paper.

{{< youtube 5MdSE-N0bxs>}}

[^1]: It is well known that it is possible to fool any deep network into incorrectly classifying its input , but this is only achieved by optimising the inputs away from “physical” values to generate so-called adversarial examples.
[^2]: Meaning that $p (y_i |y_n) = p (y_i |T_i (y_n))$ and for any other such $T$ there exists some $f$ such that $T_i = f \circ T$. That is, $T_i$ retains all information about the previous state which is relevant to the Markov chain.
