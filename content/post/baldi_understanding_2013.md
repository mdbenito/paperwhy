---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: "2017-05-05"
tags: ["dropout", "deep-learning"]
title: "Understanding Dropout"
paper_authors: ["Baldi, Pierre", "Sadowski, Peter J."]
paper_key: baldi_understanding_2013
---

The authors set to study the "averaging" properties of dropout in a
quantitative manner in the context of fully connected, feed forward
networks understood as DAGs. In particular, architectures other than
sequential are included, cf. [Figure 1](#figure1).  In the linear case
with no activations, the output of some layer $h$ (no dropout yet) is:

$$ S^h\_i = \sum\_{l < h} \sum\_j w^{h l}\_{i j} S^l\_j . $$

And if activations are included:

\begin{equation}
  O^h\_i = A (S\_i^h) = A \left( \sum\_{l < h} \sum\_j
           w^{h l}\_{i  j} O^l\_j \right),
  \label{dag-network}
  \tag{1}
\end{equation}

with $O^0\_i = I\_i$. The authors consider only sigmoid and
exponential activation functions $A$.

<a name="figure1"></a>
{{< figure src="/img/understanding-dropout-nips-fig1.jpg"
    title="The feed forward network described by (1)">}}

Recall that dropout consists of randomly disabling (i.e. setting to 0)
some fraction of the outputs at each layer.[^1] This means that for
some fixed input, randomness is introduced in the model by the dropout
scheme. The authors only explicitly consider i.i.d. "Bernoulli gating
variables" $\delta^l\_j$ (at layer $l$, output $j$) which disable
outputs with probability $p^l\_j$ (but mention that the results extend
to other distributions):


\begin{equation}
  O^h\_i = \sigma (S\_i^h) = \sigma \left( \sum\_{l
  < h} \sum\_j w^{h l}\_{i j} \delta^l\_j O^l\_j \right) .
  \label{eq:dropout-bernoulli} 
  \tag{2}
\end{equation}

Note that probabilities and expectations are therefore always over the
set of all possible subnetworks, not over the input data.

The **key result** is the following estimate on the expected value of
an output, using the **N**ormalized **W**eighted **G**eometric
**M**ean:

\begin{equation}
  \mathbb{E} (O^h\_i) \overset{(\dagger)}{\approx}
  \text{NWGM} (O^h\_i) \overset{(\ast)}{=} A\_i^h (\mathbb{E} [S\_i^h])
  \overset{(\triangle)}{=} A\_i^h (\sum\_{l < h} \sum\_j w^{h  l}\_{i
   j} p^l\_j \mathbb{E} (O^l\_j)),
   \label{eq:main-estimate} 
   \tag{3}
\end{equation}

where the NWGM is defined as the quotient $\text{NWGM} (x) = G (x) /
(G (x) - G' (x))$ where $G (x) = \prod\_i x\_i^{p\_i}$ is the weighted
geometric mean of the $x\_i$ with weights $p\_i$, and $G' (x) =
\prod\_i (1 - x\_i)^{p\_i}$ the weighted geometric mean of their
complements.

In (3), $(\ast)$ holds exactly only for sigmoid
and constant functions (p. 2) and $(\triangle)$ follows from
independence. The approximation $(\dagger)$ (Section 4) is shown to be
exact for linear layers and to hold to first order in general. An
interesting observation is that
the
[Ky Fan inequality](https://en.wikipedia.org/wiki/Ky_Fan_inequality)
tells us:

$$ G \leqslant \frac{G}{G + G'} \leqslant E \text{, if } 0 < O_i \leqslant 0.5
   \text{ for all } i, $$
   
and empirical tests show that:

>In every hidden layer of a dropout trained network, the distribution
>of neuron activations $O^∗$ is sparse and not symmetric.<a name="figure2"></a> {{< figure src="/img/baldi2013-fig3.jpg" title="Figure 3 in the paper" >}}

This seems to indicate that the NWGM is in practice a good
approximation when using sigmoidal units. Note however that the bound
in eq. (22) $| \mathbb{E}- \text{NWGM} | \leqslant 2\mathbb{E} (1
-\mathbb{E}) | 1 - 2\mathbb{E} |$ seems rather rough,
as [Figure 3](#figure3) shows:

<a name="figure3"></a>
{{< figure src="/img/understanding-dropout-nips-fig3.svg"
    title="The upper bound is quite loose">}}
    
**Analysis of gradient descent**: Using dropout means optimizing
simultaneously over the training set and the whole set of possible
networks. Therefore, two quantities of interest are the **ensemble
error**

$$ E\_{\text{ENS}} = \frac{1}{2} \sum\_i (t\_i - O^i\_{\text{ENS}})^2$$

and the **dropout error** 

$$ E\_D = \frac{1}{2} \sum\_i (t\_i - O^i\_D)^2 . $$

In the case of a single linear unit (!) it is show that:

$$ \mathbb{E} (\nabla E\_D) = \nabla (E\_{\text{ENS}} +
R\_{\text{ENS}}) $$

with the usual $l^2$ regularizer (here for just one training sample $I$)

$$ R\_{\text{ENS}} = \frac{1}{2} \sum\_j w^2\_j I\_j^2 \text{Var}
(\delta\_j) . $$

So *in expectation, the gradient of the dropout network is the
gradient of a regularized ensemble*. Observe that:

> Dropout provides immediately the magnitude of the regularization
> term which is adaptively scaled by the inputs and by the variance of
> the dropout variables. Note that $p\_i=0.5$ is the value that
> provides the highest level of regularization.

Analogously, *for a single sigmoid unit: the expected value of the
gradient of the dropout network is approximately the gradient of a
regularized ensemble network*:

$$ \mathbb{E} (\nabla E\_D) \approx \nabla E\_{\text{ENS}} + \lambda
   \sigma' (S\_{\text{ENS}}) w\_j I\_j^2 \text{Var} (\delta\_j) . $$
   
These results are extended to deeper networks in: {{< cite
baldi_dropout_2014 >}}

**Simulations**: the validity of the bounds is tested using Monte
Carlo approximations to the ensemble distribution. It is shown in
several examples how dropout favours the sparsity of activations and
"increases the consistency of layers" after dropout layers.

[^1]: See {{< cite hinton_improving_2012 >}} for the introduction of dropout.
