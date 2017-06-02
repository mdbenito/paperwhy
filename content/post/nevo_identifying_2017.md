---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: 2017-04-27
title: Identifying a minimal class of models for high–dimensional data
tags: ["regression", "sparsity", "high-dimension", "consistency"]

paper_authors: ["Nevo, Daniel", "Ritov, Ya'acov"]
paper_key: nevo_identifying_2017
---

**tl;dr**: a technique for feature selection in regression which might
be useful for exploratory analysis and which can provide guidelines
for designing subsequent costly experiments by hinting at which
features need not be collected. The main weaknesses are multiple
non-discoverable hyperparameters, a blind random search for
optimization, and a not so easily actionable output of the algorithm.

Consider sparse regression with a number of features/predictors $p$
greater than the number of datapoints $n$. In this setting it is vital
to extract features relevant to the regression and compute using the
"effective" $p$ of the dataset. The authors define a **model of order
$k$** as a particular choice of $k$ predictors $S$ among the $p$
available.

The **goal** is then to explore as many possible models of order $k$
using some non-greedy search strategy (to avoid similar models in the
results), then use the trajectory of the search to make statements
about the relevance of different predictors. The idea is to be able to
say things like "predictor #xxx was present in all models of size
$6>k>1$" and extract useful information from there for posterior analysis.

A key question is that of the **consistency** of the estimated
models.[^1] Recall that the Lasso (regression with $l^1$ penalty) and
the ElasticNet (which uses a convex combination of $l^1$ and $l^2$
penalties) provide no guarantees (under conditions applicable in
practice) about the consistency of the estimation of the subset of
predictors S with non-zero weights.[^2]

The **key idea** to this respect is to change the problem and not to
try to find the "true" $S_0$ (which is unlikely to exist, and even
more so if $p>n$) but a set of models. The authors define then a
**minimal class of models of size $k$ and efficiency $\eta$** as the
set of all models with risk within $\eta$ margin of the optimal risk
among all models of order $k$. This is clearly an untractable class,
since it is defined in terms of unknown population quantities, but it
can be approximated with sample quantities. There seem to be no bounds
on the quality of approximation.

The **method** is as follows: Start with a reduced range of orders
$\mathcal{M}$, then apply simulated annealing to explore the space of
models. The transition from one model to another is done by changing the
state (in/out) of just one predictor at a time. Which one is taken is
determined by probabilities $p^in,p^out$ depending on scores $\gamma$
for each predictor. These are used in computing a threshold $q$ for the
Metropolis-Hastings criterion. See p.7. in the paper for the details

**Examples**:

1. Expression of $p=4088$ genes in $n=71$ samples of *Bacilus
subtilis*, the target is the production of Riboflavin. The method
picked 112 models providing some (inconclusive) insights into which
genes might be worth exploring by experimentalists (e.g. gene #xxx
appears so and so many times, genes this and that almost never
do…).
2. Air pollution dataset, more of the same idea: try to guess what
predictors are relevant by exploring the search path of the model
estimators.


[^1]: A model estimator $\hat{S}\_n$ is **consistent** if $\underset{n \rightarrow \infty}{\lim} \mathbb{P} (\hat{S}\_n = S_0) = 0$, where $S_0$ is the true set of relevant predictors (i.e. with non-zero coefficients in the true regression function, $\mathbb{E}(Y\|X)$).

[^2]: See e.g. Larry Wasserman's [lecture notes](http://www.stat.cmu.edu/~larry/=sml/) at Carnegie Mellon.
