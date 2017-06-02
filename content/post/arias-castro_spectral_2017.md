---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: 2017-04-25
title: Spectral Clustering based on Local PCA
tags: ["manifold-clustering", "clustering", "pca", "unsupervised"]
paper_authors: ["Arias-Castro, Ery", "Lerman, Gilad", "Zhang, Teng"]
paper_key: arias-castro_spectral_2017
---

*Actually appeared in 2011.*

**tl;dr** This paper develops an algorithm in manifold clustering[^1],
_**C**onnected **C**omponent **E**xtraction_, which attempts to
resolve the issue of intersecting manifolds. The idea is to use a
local version of PCA at each point to determine the "principal" or
"approximate tangent space" at that point in order to compute a set of
weights for neighboring points. Then these weights are used to build a
graph and _**S**pectral **G**raph **P**artitioning_[^2] is applied to
compute its connected componets.

---

Because the goal is to resolve intersections, it is necessary to
assume that the input data is distributed around smooth, smoothly
intersecting manifolds.[^3]

Three variants of CCE are presented: comparing covariances, using
projections the eigenvectors of the covariance matrix and using local PCA.

### Highlights

* In their comparison of the methods to prior work, the authors review
  multiple affinities and their relative weaknesses.
* There is **guaranteed clustering** for CCE using local covariances and
  local projections (but not local PCA), assuming that:
     - the data has been sampled uniformly from smooth $d$-dimensional
       manifolds with bounded additive noise.[^5]
     - the right (hyper-) parameters have been set. Note that these
       values depend on the geometric configuration but in principle
       cross-validation should work.
*  Detailed and self-contained proofs are given: the authors provide a
   **characterization of the covariance matrix** for data as above
   (uniformly sampled, additive noise), as estimates of relevant
   geometric quantities, bounds on distances of covariance functions
   (and those of their push-forwards), as well as a form of
   **[Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)
   for matrices**, among other things.
* Better performance than ... Check the paper for detailed examples.


### Method

All their three variants begin by subsampling the dataset, then
computing the sample covariance matrix at each point.

The first variant then discards points with a heuristic which is
believed to approximate the condition of "being close to an
intersection" and builds an unweighted graph with edges between points
$x\_i, x\_j$ which are both close enough and have similar enough
covariance matrices $| C\_i - C\_j \| \le \eta r^2 $. This is encoded
in the **affinity**:

\\[ W\_{i j} = \mathbb{1}\_{\{ \| x\_i - x\_j \| < \varepsilon \}} 
   \mathbb{1}\_{\{ \| C\_i - C\_j \| < \eta r^2 \}}, \\]

Here the matrix norm is the *spectral norm*: intuitively, the smaller
$\| C\_i - C\_j \|$ is, the "more parallel" the covariance matrices
will be.

This graph is then fed to SGP to compute the connected components and
the data is clustered.

The second method, local projection, skips the heuristic but instead
computes projections $Q\_i$ onto the principal vectors of the $C\_i$
(with eigenvalues $>\sqrt{\eta}\|Q\_i\|$) and uses the affinity

\\[ W\_{i j} = \mathbb{1}\_{\{ \| x\_i - x\_j \| < \varepsilon \}} 
   \mathbb{1}\_{\{ \| Q\_i - Q\_j \| < \eta \}}, \\]

Here is the third and best performing method, using local PCA, in
detail:

1. Create a subsample of adequately spaced data points $S = \{ x\_1,
  \ldots, x\_{n\_0} \}$, which are centers of disjoint balls of radius
  $r > 0$ a fixed hyperparameter.
2. For each $x\_i \in S$, perform *local PCA* in the nhood $N\_r
   (x\_i)$ to compute the principal directions of an approximate
   tangent space at $x\_i$.[^4] Let $Q\_i$ be the projection onto the
   first $d$ vectors of $C\_i$. The local dimension $d$ is another
   (essential) hyperparameter, but the gap in the singular values can
   provide good splitting points for the cross-validation.
3. For each $x\_i, x\_j \in S$ compute the affinities
   \\[ W\_{i j}
   = \exp \left( - \frac{\| x\_i - x\_j \|^2}{\varepsilon^2} \right)
   \exp \left( - \frac{\| Q\_i - Q\_j \|^2}{\eta^2} \right), \\] 
   intuitively, the smaller $\| Q\_i - Q\_j \|$ is, the "more
   parallel" the tangent spaces will be and the closer to 1 the second
   exponential will be. On the contrary, a large norm will make the
   affinity close to 0.
4. Use SGP on $W$. Cluster all data according to closest center.

### Drawbacks 

* Assumes a known number of clusters (required for SGP).
* Assumes the dimension of the manifolds is known. Estimates using the
  jump in eigenvalues after PCA didn't perform well.
* Assumes smooth manifolds intersect at non zero angles (but consider
  the graphs of $x^2$ and $x^3$ intersecting at 0).[^6]

[^1]: See e.g. the review {{< cite vidal_subspace_2011 >}}
[^2]: {{< cite ng_spectral_2002 >}}
[^3]: Manifold clustering is particularly relevant in motion segmentation and some specific cases of face recognition. But what has been the effect of the development of deep convolutional neural networks on the relevance of these techniques?
[^4]: Local PCA amounts to an eigenvalue decomposition of the local covariance matrix $C\_i$.
[^5]: **Check me:** Is this essential for the proofs?
[^6]: Maybe one could use some idea of momentum as one moves through the manifold? Assuming enough regularity it should be possible to determine when a "trajectory" escapes the manifold.
