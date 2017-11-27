---

title: "Dropout training as adaptive regularization"
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: "2017-05-31"
tags: ["dropout", "regularization", "adagrad", "semi-supervised"]
paper_authors: ["Wager, Stefan", "Wang, Sida", "Liang, Percy"]
paper_key: "wager_dropout_2013"

---

**tl;dr:** dropout (of features) for GLMs is a noising procedure equivalent to 
Tykhonov regularization. A first order approximation of the regularizer 
actually scales the parameters with the Fisher information matrix, adapting the 
objective function to the dataset, independently of the labels. This makes 
dropout useful in the context of semi-supervised learning: regularizers can be 
adapted to the unlabeled data yielding better generalization. For logistic 
regression the adaption amounts to favoring features on which the estimator is 
confident.

---

For shallow architectures, there were already some (stated) results on
the averaging properties of dropout.[^10] This was later extended to
multiple layers with sigmoid units: simple weighting of outputs in the
forward pass computes an approximation of the expectation of the
ensemble.[^11] Today's paper predates this work and focuses still on shallow
networks, albeit within the wider scope of Generalized Linear Models.

Since these are shallow models, dropout is performed on the inputs and it can 
be compared to other methods of input perturbation like additive Gaussian 
noise.

There are **3 main contributions** in this paper:

1. A dropout regularizer for a GLM is (up to first order) **equivalent to a 
   classical Tykhonov regularizer** with $L^2$-norm, with a specific scaling. 
   Crucially, this scaling *depends on the data but not on the labels* and 
   makes the regularization adaptive.
1. Incorporating this regularization into a rewriting of SGD as the repeated 
   solution of regularized linear problems leads to an **update similar to 
   AdaGrad**.[^1] A connection between the goals of both is established.
1. In the case of logistic regression the dropout regularizer is shown to 
   **favour confident predictions, regardless of the label** (in the sense that 
   it penalizes less those weights corresponding to features on which the 
   predicted probability is far from $1 / 2$). Therefore it makes sense to 
   **apply it to semi-supervised problems** computing an extra term over 
   unlabeled data.


### (Feature-) Dropout is weighted $L^2$-regularization

Consider any **Generalized Linear Model** with parameters $\beta$, inputs $x 
\in \mathbb{R}^d$ and outputs $y \in Y$, i.e.[^2]

\\[ p (y|x, \beta) = h (y) \mathrm{e}^{yx \beta - A (x \beta)}, \\]

and **negative log likelihood** as the loss. For one training sample $(x, y)$: 
$l (\beta ; x, y) = - \log p (y|x, \beta)$. Now choose some noise vector $\xi 
\in \mathbb{R}^d$ with i.i.d. entries with zero mean and replace $x \mapsto 
\tilde{x}$ where $\tilde{x}$ has been noised with $\xi$ in some way we specify 
later. A couple of computations show that the empirical loss on the (full) 
noised data $\tilde{\boldsymbol{x}} = (\tilde{x}\_{1}, \ldots, 
\tilde{x}\_{n}),\boldsymbol{y}= (y\_{1}, \ldots, y\_{n})$ is the loss on the 
original data plus a new term:

\\[ \hat{L} (\tilde{\boldsymbol{x}}, \boldsymbol{y}, \beta) = \hat{L}
   (\boldsymbol{x}, \boldsymbol{y}, \beta) + R (\beta), \\]

which is the **noising regularizer**

\begin{equation}
  \label{eq:noising-regularizer}\tag{1} R (\beta) := \sum\_{i = 1}^n
  \mathbb{E}\_{\xi} [A (\tilde{x}\_{i} \beta)] - A (x\_{i} \beta) .
\end{equation}

$R (\beta)$ has two key features:

* **It does not depend on the labels**: this will allow for its use in 
  unsupervised setting.
* **It is adapted to the training data**.


But how exactly "adapted"? Definition (1) is quite impenetrable as is, 
even after plugging in a specific $A$. Assuming we can do a Taylor expansion of 
$A$ around $x \beta$ one obtains[^3]

\\[ \mathbb{E}\_{\xi} [A (\tilde{x} \beta)] - A (x \beta) \approx \tfrac{1}{2}
   A'' (x \beta) \operatorname{Var}\_{\xi}  [\tilde{x} \beta] \\]

and substituting into (1) the (approximate) **quadratic noising regularizer**:

\begin{equation}
  \label{eq:quadratic-noising-regularizer}\tag{2} R^q (\beta) := \sum\_{i =
  1}^n \tfrac{1}{2} A'' (x \beta) \operatorname{Var}\_{\xi}  [\tilde{x} \beta]
  .
\end{equation}

Note that (2) is in general non-convex. When questioned by a reviewer about 
this fact, the authors respond

> Although our objective is not formally convex, we have not encountered any 
> major difficulties in fitting it for datasets where n is reasonably large 
> (say on the order of hundreds). When working with LBFGS, multiple restarts 
> with random parameter values give almost identical results. The fact that we 
> have never really had to struggle with local minimas suggests that there is 
> something interesting going on here in terms of convexity.


We can now fix the noising method and look at its variance to gain insight 
into what $R^q$ does, and hopefully, by extension $R$.[^4] The authors 
consider:

* **Additive gaussian noise**: $\tilde{x} = x + \xi$ with $\xi \_{i}$ i.i.d. 
  spherical Gaussians $\mathcal{N} (0, \sigma^2 \operatorname{Id}\_{d})$.
* **Dropout noise**: fix $\delta \in (0, 1)$ to build a (scaled) binary mask 
  $\xi$ with i.i.d entries $\operatorname{Bernoulli} (1 - \delta)$ and set 
  $\tilde{x} = x \odot \xi / (1 - \delta)$ to cancel some of the inputs with 
  probability $\delta$.[^5]


Notice that in both cases $\mathbb{E}\_{\xi} [\tilde{x}] = x$ and the 
expectation of the Taylor expansion of $A$ yields (2) (that's the reason for 
the scaling factor $\delta$). After performing the necessary computations, and 
assuming the design matrix has been normalized to $\Sigma \_{i  j} x^2\_{i  j} 
= 1$, the authors obtain the following neat table:

<table style="width: 100%">
  <tbody><tr>
    <td style="text-align: center; padding-left: 0em; padding-right: 0em"><table style="display: inline; vertical-align: -2.2em">
      <tbody><tr>
        <td style="text-align: right"></td>
        <td style="text-align: center" bgcolor="#dfdfdf">Linear regression</td>
        <td style="text-align: center" bgcolor="#dfdfdf">Logistic regression</td>
        <td style="text-align: center" bgcolor="#dfdfdf">GLM</td>
      </tr><tr>
        <td style="text-align: right" bgcolor="#dfdfdf">$L^2$-penalty</td>
        <td style="text-align: center">$\| \beta \|^2_2$ </td>
        <td style="text-align: center; border-bottom: 1px solid"> $\| \beta \|^2_2$ </td>
        <td style="text-align: center; border-bottom: 1px solid"> $\| \beta \|^2_2$ </td>
      </tr><tr>
        <td style="text-align: right" bgcolor="#dfdfdf">Additive noise</td>
        <td style="text-align: center; border-right: 1px solid">  $\| \beta \|^2_2$ </td>
        <td style="text-align: center; border-top: 1px solid; border-left: 1px solid"> $\| \beta \|^2_2  \sum_i p_i  (1 -
    p_i)$ </td>
        <td style="text-align: center; border-top: 1px solid; border-right: 1px solid"> $\| \beta \|^2_2 \operatorname{tr} (V (\beta))$ </td>
      </tr><tr>
        <td style="text-align: right" bgcolor="#dfdfdf">Dropout noise</td>
        <td style="text-align: center; border-right: 1px solid"> $\| \beta \|^2_2$ </td>
        <td style="text-align: center; border-bottom: 1px solid; border-left: 1px solid"> $\sum_{i, j} p_i  (1 - p_i) x_{i j}^2 \beta_j^2$ </td>
        <td style="text-align: center; border-bottom: 1px solid; border-right: 1px solid"> $\beta^{\top} \operatorname{diag} (X^{\top} V (\beta)
    X) \beta$ </td>
      </tr></tbody>
    </table></td>
  </tr><tr>
    <td style="text-align: center; padding-left: 0em; padding-right: 0em; height: 0.5em"></td>
  </tr><tr>
    <td style="text-align: center; padding-left: 0em; padding-right: 0em; padding-left: 1.5em; padding-right: 1.5em"><p>
      <font size="-1"><p>
        <b>Table 1. </b><a id="auto-1"></a><a id="tab:summary-regs"></a>
        $R^q$ (up to constants) for different  models and noising methods. See below for the definition of $V (\beta)$.
      </p></font>
    </p></td>
  </tr></tbody>
</table>

The first row holds by definition. The first column recovers known results[^6] 
and adds the fact that dropout (after scaling) on linear regression is ridge 
regression. It's the box who tells a more interesting story. First we note that 
the key matrix $V (\beta) \in \mathbb{R}^{n \times n}$ is diagonal with entries 
$V (\beta)\_{i  i} = A'' (x\_{i} \beta)$.

Additive noising for logistic regression penalizes more strongly uncertain 
predictions ($p\_{i} \approx 0.5$). For arbitrary GLMs, $R^q$ is just 
multiplied by a constant.

Dropout in logistic regression has the same feature as additive noise *plus 
selective exclusion of features:* given a training sample $x\_{i}$, $\beta 
\_{j}$ is not penalized if $x\_{i  j} = 0$. In particular $p\_{i}  (1 - 
p\_{i})$ and $\beta \_{j}$ may both be large if the *cross-term* $x\_{i  j}^2$ 
is small. This means that

> (…) dropout regularization should be better than $L^2$-regularization for 
> learning weights for features that are rare (i.e., often 0) but highly 
> discriminative, because dropout effectively does not penalize  $j$ over 
> observations for which $x\_{i  j} = 0$.


And

> dropout rewards those features that are rare and positively co-adapted with 
> other features in a way that enables the model to make confident predictions 
> whenever the feature of interest is active.


In the more general case the insight comes from the fact that

\\[ \tfrac{1}{n} X^{\top} V (\beta^{\star}) X = \tfrac{1}{n} \sum\_{i = 1}^n
   \nabla^2 l (\beta^{\star} ; x\_{i}, y\_{i}) \\]

is an estimator of the Fisher information matrix $\mathcal{I}$. Therefore if 
we write $\beta^{\top} \operatorname{diag} (X^{\top} V (\beta) X) \beta = 
\beta^{\top}D \beta = \| D^{1 / 2} \beta \|\_{2}^2 = \| \tilde{\beta} 
\|\_{2}^2$ we see that dropout is applying an $L^2$ penalty after normalizing 
with an approximation of $\operatorname{diag} (\mathcal{I})^{- 1 / 2}$.

> The Fisher information is linked to the shape of the level surfaces of $l 
> (\beta)$ around $\beta^{\star}$. If $\mathcal{I}$ were a multiple of the 
> identity matrix, then these level surfaces would be perfectly spherical 
> around $\beta^{\star}$.


By normalizing, the feature space is deformed into a shape where "*the 
features have been balanced out*".[^7] The authors provide a very nice 
picture for intuition:

{{< figure src="/img/wager_dropout_2013-figA2.jpg"
         title="Comparison of two $L^2$ regularizers." >}}

> (page 11 in the Appendix)In both cases, the black solid ellipses are
> level surfaces of the likelihood and the blue dashed curves are level
> surfaces of the regularizer; the optimum of the regularized objective
> is denoted by OPT. The left panel shows a classic spherical $L^2$
> regular izer $\| \beta \|\_{2}^2$, whereas the right panel has an
> $L^2$ regularizer $\beta^{\top} \operatorname{diag}(\mathcal{I})
> \beta$ that has been adapted to the shape of the likelihood
> ($\mathcal{I}$ is the Fisher information matrix). The second
> regularizer is still aligned with the axes, but the relative
> importance of each axis is now scaled using the curvature of the
> likelihood function. As argued [above], dropout training is
> comparable to the setup depicted in the right panel.

### Relation to AdaGrad

By rewriting standard SGD into an iterative solution of linear $L^2$-penalized 
problems

\\[ \hat{\beta}\_{t + 1} = \operatorname{argmin}\_{\beta}  \left \lbrace  l
   (\hat{\beta}\_{t} ; x\_{t}, y\_{t}) + \nabla l (\hat{\beta}\_{t})  (\beta -
   \hat{\beta}\_{t}) + \frac{1}{2 \eta \_{t}}  \| \beta - \hat{\beta}\_{t}
   \|\_{2}^2 \right \rbrace  \\]

and substituting the dropout penalty for the penalty in this formulation, one
obtains the update rule

\\[ \hat{\beta}\_{t + 1} =\operatorname{argmin}\_{\beta}  \left \lbrace  l
   (\hat{\beta}\_{t} ; x\_{t}, y\_{t}) + g\_{t}  (\beta - \hat{\beta}\_{t}) +
   R^q (\beta - \hat{\beta}\_{t} ; \hat{\beta}\_{t}) \right \rbrace  \\]

with the **centered quadratic dropout penalty**, similarly to the entry in 
Table 1:

\\[ R^q (\beta - \hat{\beta}\_{t} ; \hat{\beta}\_{t}) = (\beta -
   \hat{\beta}\_{t})^{\top} \operatorname{diag} (X^{\top} V (\hat{\beta}\_{t})
   X)  (\beta - \hat{\beta}\_{t}) . \\]

This is effectively solving the problem of SGD has learning weights for 
"rare but highly discriminative features", by using the update

\\[ \hat{\beta}\_{t + 1} = \hat{\beta}\_{t} - \eta \_{t} A\_{t}^{- 1} \nabla l
   (\hat{\beta}\_{t}) . \\]

AdaGrad[^8] uses $A\_{t} =\operatorname{diag} (\nabla^{\top} l 
(\hat{\beta}\_{t}) \nabla l(\hat{\beta}\_{t}))^{- 1 / 2}$, warping the 
gradient by some sort of intrinsic metric, whereas dropout uses its estimate of 
the Fisher information.[^9] However, in the limit $\hat{\beta}\_{t} \rightarrow 
\beta^{\star}$ for GLMs the expectations of both matrices are equal to 
$\mathcal{I}$, meaning that the SGD updates when using feature dropout in GLMs 
are "converging" in some sense to AdaGrad updates.

### Semi-supervised tasks

As we said above, the dropout regularizer is shown to change the loss function 
with the Fisher information matrix in a way that focuses on weights relevant 
for discriminative features, *without recourse to the labels $y\_{i}$*. 
Therefore in a semi-supervised context, we can use unlabeled data to improve 
the regularizer:

\\[ R\_{\ast} (\beta) := \frac{n}{n + \alpha m}  \left( R (\beta) + \alpha
   R\_{\text{unlabeled}} (\beta) \right), \\]

where $n$ is the size of the labeled dataset, $m$ that of the unlabeled one 
and $\alpha$ a "discount factor" for the latter which is a hyperparameter. 
Unlike other semi-supervised approaches relying on generative models, the 
authors' approach

> is based on a different intuition: we'd like to set weights to make confident 
> predictions on unlabeled data as well as the labeled data, an intuition 
> shared by entropy regularization [24] and transductive SVMs [25].



[^1]: See {{< cite duchi_adaptive_2011 >}}.

[^2]: Recall that in a GLM one uses a so-called **link function** $h$ to relate a linear predictor $x \beta$ with the posterior $p (y|x)$ by means of the relationship $\mathbb{E} [y|x] = h^{- 1} (x \beta)$. In our notation, $h = A'$. To fix ideas think of logistic regression, where $p (y|x) = (1 + \mathrm{e}^{- x \beta})^{- 1}$. In this case we assume $y \in \lbrace  0, 1  \rbrace $, the log likelihood is $p (\boldsymbol{y}|\boldsymbol{x}) = \prod\_{i} p\_{i}^{y\_{i}} (1 -p\_{i})^{1 - y\_{i}}$, with $p\_{i} := (1 + \mathrm{e}^{- x\_{i} \beta})^{- 1}$ and the negative log likelihood is the **cross entropy loss**: $\log p (y|x) = - \sum\_{i} y\_{i} \log p\_{i} + (1 - y\_{i}) \log (1 -p\_{i})$.

[^3]: Indeed $A (\tilde{x} \beta) - A (x \beta) = A' (x \beta)  (\tilde{x} \beta - x \beta)+ \frac{1}{2} A'' (x \beta)  (\tilde{x} \beta - x \beta)^2 + \text{h.o.t.}$ and taking expectations: $\mathbb{E}\_{\xi} [A (\tilde{x} \beta)] - A (x \beta) = A' (x \beta) (\mathbb{E} [\tilde{x} \beta] - x \beta) + \frac{1}{2} A'' (x \beta)\mathbb{E} (\tilde{x} \beta - x \beta)^2 + \text{h.o.t}$.

[^4]: There is a handwavy discussion in the paper on the error $| R - R^q |$ which is not worth discussing here. Suffice to say: it works "well" in practice.

[^5]: Here $\odot$ stands for the entrywise or [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)).

[^6]: See {{< cite bishop_training_1995 >}} for more on additive noise leading to ridge regression.

[^7]: Notice that we could use any quadratic form to redefine the norm in which weights are measured. There are surely many other interesting possibilities!

[^8]: See {{< cite duchi_adaptive_2011 >}}.

[^9]: This looks like a nice connection to second order methods: warp the update step with information on the target function or warp feature space with information on the data to "improve" it (*very* handwavily put…)

[^10]: See {{< cite hinton_improving_2012 >}}.

[^11]: See {{< cite baldi_understanding_2013 >}}, {{< cite baldi_dropout_2014 >}}.
