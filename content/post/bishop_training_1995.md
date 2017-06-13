---

title: "Training with noise is equivalent to Tikhonov regularization"
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: "2017-06-12"
tags: ["regularization", "input-noise"]
paper_authors: ["Bishop, Chris M."]
paper_key: "bishop_training_1995"

---


**tl;dr:** Adding noise to training inputs changes the risk function. A Taylor 
expansion shows that up to a term quadratic in the noise amplitude, the 
empirical risk is the same as without noise but with an additional term 
involving 1st derivatives of the estimator.

---

In our quest to understand all things regularization, today we review an old 
piece by Christopher Bishop no less!

### The bias-variance tradeoff

We begin with a classical observation: for any statistical model we develop 
(i.e. for any choice of estimator $T = T (X\_{1}, \ldots, X\_{N})$ as a 
function of the data), we will always face the proverbial tradeoff between bias 
and variance of the statistic wrt. different datasets. Roughly: low bias 
$\mathbb{E}\_{\mathbf{X}} T$ implies high variance $V\_{\mathbf{X}} T$ and 
viceversa. One way to see it is fhe following: We desire low model complexity 
(fewer parameters for the description of $T$) but this typically results in 
high bias and low variance, so we can increase the complexity for a lower bias 
but higher variance. There are many ways to tackle this connundrum: a few 
examples are **structural stabilization** to reduce the bias, **ensembling** of 
poor (*weak*) estimators to reduce the variance, or **regularization** of the 
objective function to achieve the same goal. The latter consists in adding a 
penalty term to the risk which *regularizes* the estimator in the sense that 
the problem of computing a good estimator from the data becomes well-posed, 
i.e. it depends smoothly on the training set. This leads to classical [Tikhonov 
regularization](en.wikipedia.org/tikhonov_regularization).

### Adding noise to the input

The focus of the paper is in the related technique of **adding noise to the 
training samples** and how (up to second order) it can be regarded as just 
adding a penalty term to the risk function. This noise could for instance be 
simply Gaussian, or *salt and pepper* noise (a binary mask), common in image 
recognition tasks.

To fix ideas, suppose that we are training an estimator $x \mapsto \hat{y} 
(x)$ via **empirical risk minimization**.[^1] We have training data $(x\_{i}, 
y\_{i})$ being realizations of the random variable $(X, Y)$ and we use 
quadratic loss and add some noise $\xi$ to the input $X$. This changes the risk 
function from $\mathbb{E}\_{X, Y} [| \hat{y} (X) - Y |^2]$ to $\mathbb{E}\_{X, 
Y, \xi} [| \hat{y} (X + \xi) - Y |^2]$, so that the **empirical risk** to 
minize is

\begin{equation}
  \label{eq:empirical-risk-noise}\tag{1} \hat{R}\_{\xi} (\hat{y}) =
  \frac{1}{N}  \sum\_{i = 1}^N | \hat{y} (x\_{i} + \xi \_{i}) - y\_{i} |^2 .
\end{equation}

Assuming that the amplitude of the noise $| \xi |$ is small, a second order 
Taylor expansion of $\hat{y} (X + \xi)$ around $X$ yields, after some 
computations on the population risk and bringing them back to the empirical 
one, a new (1):

\\[ \hat{R}\_{\xi} (\hat{y}) = \hat{R} (\hat{y}) + \eta^2 \rho (\hat{y}) +
   \text{h.o.t.} \\]

where the term $\rho (\hat{y})$ is the **(empirical) regularizer** and $\eta^2 
=\operatorname{Var} (\xi)$.[^2] The expression that pops out for $\rho$ (which 
we don't reproduce here) has the big disadvantage of being **not bounded from 
below** so that $\hat{R}\_{\xi}$ is a rather poor choice for an objective 
function to minimise.

### A quadratic approximate regularizer

However, one can rewrite the equations in term of the conditional expectations 
$\mathbb{E} [Y|X]$ and $\mathbb{E} [Y^2 |X]$ to obtain equivalent ones where it 
becomes apparent that, for small variances $\eta^2$, the (empirical) 
regularizer can be approximated by

\\[ \tilde{\rho} (\hat{y}) = \frac{1}{2 N}  \sum\_{i = 1}^N \| \nabla \hat{y}
   (x\_{i}) \|^2 . \\]

This is now much better: being quadratic and bounded below by 0 it is a 
“good” term for the objective. The derivations in the paper show that it 
leads to the same minima (up to $\mathcal{O} (\eta^2)$) as the “true” 
regularizer $\rho$.

The computations are next repeated for the **cross-entropy error** to obtain a 
similar approximate regularizer, this time with an additional factor breaking 
its nice quadratic and Tikhonov-like form. There are efficient ways to compute 
the derivatives involved as part of backpropagation.

The conclusion is then that, at least in these settings one can simply plug 
these regularizers in instead of adding noise to the input. This might not be 
that interesting computationally, but it provides a deeper understanding of 
what it is that we are doing when we perturb inputs, a technique very common 
nowadays e.g. in object classification with convnets.

Finally, the paper concludes with a specific computation of the updates for 
weights in a neural network using the quadratic regularizer. The problem is 
that the Hessian of the error wrt. the weights is required, making the method 
unattractive for modern applications with millions of parameters.


[^1]:  That is: we want $\hat{y} =\operatorname{argmin}\_{y}  \hat{R} (y)$, where $\hat{R}$ is an approximation to the **population risk** $R (y)$, a magnitude encoding the expected failure of $y$ in predicting well from $X$. There are many good introductions to statistical learning theory available: for a very brief one, see {{< cite bousquet_advanced_2004 >}} (p. 169).
[^2]: Note that we are omitting all population quantities here for brevity, but the regularizer is computed for the true expected error (population risk), then approximated using the sample data. The paper is actually a bit confusing in this respect since it tries to gather both population and sample quantities under one notation.
