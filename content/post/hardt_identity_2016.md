---

title: "Identity matters in Deep Learning"
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: "2017-06-07"
tags: ["deep-learning", "deep-residual-networks"]
paper_authors: ["Hardt, Moritz", "Ma, Tengyu"]
paper_key: "hardt_identity_2016"

---

**tl;dr:** vanilla residual networks are very good approximators of functions 
which can be represented as linear perturbations of the identity. In the linear 
setting, optimization is aided by a benevolent landscape having only minima in 
certain (interesting) regions. Finally, very simple ResNets can completely 
learn datasets with $\mathcal{O} (n \log n + \ldots)$ parameters. All this 
seems to indicate that deep and simple architectures might be enough to achieve 
great performance.

---

In general, it is hard for classical deep nets to "preserve features which 
are good": initialization with zero mean and small gradients make it hard to 
learn the identity at any given layer. Even though **batchnorm**[^1] seeks to 
alleviate this issue, it has been **residual networks** which have most 
improved upon it.[^2] In a residual net

> (…) each residual layer has the form $x + h (x)$, rather than $h (x)$. This 
> simple reparameterization allows for much deeper architectures largely 
> avoiding the problem of vanishing (or exploding) gradients.

### Identity parametrizations improve optimization

The authors work first in the linear setting, i.e. they consider only networks 
which are compositions of linear perturbations of the identity:[^3]

\\[ h (x) = (I + A\_{l}) \cdots (I + A\_{2})  (I + A\_{1}) x \\]

The objective function to mimize is the **population risk** with quadratic 
loss:

\\[ f (A) = f (A\_{1}, \ldots, A\_{l}) := \mathbb{E}\_{X, Y} | Y - (I + 
A\_{l}) \cdots (I + A\_{1}) X |^2 . \\]

Labels are assigned with noise, that is $Y = RX + \xi$, with $\xi \sim 
\mathcal{N} (0, I\_{d})$. Note that the problem over the variables $A = 
(A\_{1}, \ldots, A\_{l})$ is non-convex.

The first result of the paper states that deep networks with many layers have 
minima of proportionally low (spectral) norm.[^4] Because of this, it makes 
sense to study critical points with small norm, and it turns out that there are 
only minima. The main result of this section is the following one, where one 
can think of $\tau$ as being $\mathcal{O} (1 / l)$:

> **Theorem 1:** For every $\tau < 1$, every critical point $A$ of the objective 
> function $f$ with
> \\[ \underset{1 \leqslant i \leqslant l}{\max}  \| A\_{i} \| \leqslant \tau 
> \\]
> is a global minimum.

This is good news since, under the assumption that the model is correct, a 
"good" (in some sense) optimization algorithm will converge to it.[^5] The 
proof is relatively straightforward too: rewrite the risk as the norm of a 
product by the covariance: $f (A) = \| E (A) \Sigma^{1 / 2} \|^2 + C$, then, 
using that $\| A\_{i} \|$ are small, show that if $\nabla f (A) = 0$ this can 
only be if $E (A) = 0$, where $E$ precisely encodes the condition of being at 
an optimum: $E (A) = (I + A\_{l}) \cdots (I + A\_{1}) - R$.

### Identity parametrizations improve representation

The authors consider next non-linear simplified residual networks with each 
layer of the form

\begin{equation}
  \label{eq:residual-unit}\tag{1} h\_{j} (x) = x + V\_{j} \sigma (U\_{j} x +
  b\_{j})
\end{equation}

where $V\_{j}, U\_{j} \in \mathbb{R}^{k \times k}$ are weight matrices, 
$b\_{j} \in \mathbb{R}^k$ is a bias vector and $\sigma (z) = \max (0, z)$ is a 
ReLU activation.[^6] Note that *the layer size is constant*. No batchnorm is 
applied. The problem is $r$-class classification. Assuming (the admittedly 
natural condition) that all the training data are uniformly separated by a 
minimal distance $\rho > 0$, they prove that perfectly learning $n$ data points 
is possible with $n \log n$ parameters:

> **Theorem 2:** There exists a residual network with $\mathcal{O}(n \log n + 
> r^2)$ parameters that perfectly fits the training data.

By choosing the hyperparameter $k \in \mathcal{O} (\log n)$ and $l = \lceil n 
/ k \rceil$ the complexity stated is obtained with a bit of arithmetic. The 
proof consists then of a somewhat explicit construction of the network. Very 
roughly, the weight matrices of the hidden layers are chosen as to assign each 
data point to one of $r$ **surrogate label vectors** $q\_{1}, \ldots, q\_{r} 
\in \mathbb{R}^k$, then the last layer converts these to 1-hot label vectors 
for output. The main ingredient is therefore the proof that it is possible to 
map the inputs $x\_{i}$ to the surrogate vectors in a way such that the final 
layer has almost no work left to do.[^7]

This is achieved by showing that:

> (…) for an (almost) arbitrary sequence of vectors $x\_{1}, \ldots, x\_{n}$ 
> there exist [weights $U, V, b$] such that operation (1) transforms $k$ [of 
> them] to an arbitrary set of other $k$ vectors that we can freely choose, and 
> maintains the value of the remaining $n - k$ vectors.

### Experiments

{{< figure src="/img/hardt_identity_2016-fig1.jpg"
         title="Convergence plots of best model for CIFAR10 (left) and CIFAR (100) right." >}}

Working on CIFAR10 and CIFAR100, the authors tweaked a [standard ResNet 
architecture](https://github.com/tensorflow/models/tree/master/resnet) in 
Tensorflow to have constant size $c$ of convolution, no batch norm and smaller 
weight initialization.[^8] Several features stand out:

* The last layer is a fixed random projection. Therefore all parameters are in 
  the convolutioms.
* Lack of batchnorm or other regularizers seemed not to lead to serious 
  overfitting, even though the model had $\sim 13.6 \times 10^6$ parameters.
* Both problems where tackled with the same networks and the convolutions 
  where of constant size.

Finally, results on **ImageNet** were not as bright, though the authors seem 
confident that this is due to lack of tuning of hyperparameters and learning 
rate. It would be interesting to find out how true this is and **how much 
harder this tuning becomes by having discarded regularization techniques** like 
dropout or additional data processing.

### Extensions

For an extension of these results to the non-linear case (which as of this 
writing is reported to be work in progress), be sure to check out Bartlett's 
talk: {{< cite bartlett_representational_2017 >}}.


[^1]: See {{< cite ioffe_batch_2015 >}}.
[^2]: See {{< cite he_deep_2016 >}}.
[^3]: For extensions to the non-linear setting see {{< cite bartlett_representational_2017 >}}. Note that even though in principle the network can be flattened to only one linear map by taking the product of all $A\_{i}$ with no loss in its representational capacity, a great cost in efficiency can be incurred in doing so, see {{< cite lin_why_2016 >}} (§G). The dynamics of optimization also are affected by the stacking of purely linear layers, see e.g. {{< cite saxe_exact_2013 >}}.
[^4]: Essentially, the theorem states that there exists a constant $\gamma$ depending on the largest and smallest singular values of $R$ such that there exists a global minimum $A^{\star}$ of the population risk fulfilling $\underset{1 \leqslant i \leqslant l}{\max}  \| A^{\star}\_{i} \| \leqslant \frac{1}{l}  (4 \pi + 3 \gamma)$ whenever the number of layers $l \geqslant 3 \gamma$.
[^5]: Equation (2.3) of the paper provides a lower bound on the gradient which does guarantee convergence under the assumption that iterates don't jump out of the domain of interest. See the reference in the paper.
[^6]: This is a deliberately simpler setup than the original ResNets with two ReLU activations and two instances of batch normalization. See {{< cite he_identity_2016 >}}.
[^7]: An interesting point is the use of the **Johsonn-Lindestrauss lemma** to ensure that an initial random projection of input data onto $\mathbb{R}^k$ by the first layer does not violate the condition that it remains separated, with high probability.
[^8]: Gaussian with $\sigma \sim (k^{- 2} c^{- 1})$ instead of $\sigma \sim (k^{- 1} c^{- 1 / 2})$. For more on proper initialisation see e.g. {{< cite sutskever_importance_2013 >}}.
