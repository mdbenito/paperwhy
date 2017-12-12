---

title: "Geometric loss functions for camera pose regression with deep learning"
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: "2017-11-27"
tags: ["computer vision", "camera pose", "multi-task learning"]
paper_authors: ["Kendall, Alex", "Cipolla, Roberto"]
paper_key: "kendall_geometric_2017"

---

**tl;dr:** A Bayesian point of view allows simultaneously training for two 
different losses without hyperparameters. For camera pose estimation, geometric 
reprojection loss can be used to fine tune results.

---

Today's paper application is interesting enough by itself, but perhaps more so 
how it uses Bayesian ideas to train jointly for multiple losses, balancing them 
but without adding hyperparameters.[^1] Since I find this to be the take-away 
message, instead of going through a tedious literature review of the field of 
application, which the paper does much better, let's just dive in with a couple 
of quick comments:

## Goal

Given some photograph, we wish to recover position and orientation of the 
camera (6 dofs). Some application examples are:

* Overlay a 3D model onto an image, e.g. for *augmented reality*.
* Locate pictures in maps, e.g. Google Maps' feature placing users' pictures 
  automatically.
* Estimate full pose of autonomous vehicles with inaccurate positioning: the 
  idea is to fuse GPS and / or inertial sensor data with visual information for 
  full 6 dof estimation in an “absolute” frame of reference. Note that this 
  is easier in a structured (road) environment than e.g. in the air or sea.[^2]

Some examples of what the **training data** can be (links to datasets in the 
paper)

* Flat images and 3D points: costly because it requires careful crafting with 
  LIDAR-like solutions or must be approximated with things like [structure from 
  motion](https://en.wikipedia.org/wiki/Structure_from_motion).
* RGB-D sensor data: best used in indoor applications because depth info in 
  RGB-D degrades with distance (quadratically[^3]).
* Stereoscopic images: this option seems best for outdoor applications: 
  standard techniques can provide relative position and rotation modulo 
  constant factors in the distance to objects: fusing this with sensor data 
  should work well (?).

## Architecture

As is common, the authors use a pretrained CNN model for image 
classification[^4] with the top layer and softmax output removed. In their 
stead two independent fully connected layers perform regression to predict the 
position vector $\boldsymbol{p} \in \mathbb{R}^3$ and rotation quaternion 
$\boldsymbol{q} \in \mathbb{R}^4$. Naturally, one wishes to train for 
$\boldsymbol{p}, \boldsymbol{q}$ jointly since it is to be expected that, 
conditioned on an image, they are strongly related.

> The model learns a better representation for pose when supervised with both 
> translation and orientation labels. (…) branching the network lower down 
> into two separate components (…) was less effective (…): separating into 
> distinct position and orientation features denies each the information 
> necessary to factor out orientation from position, or vice versa.

{{< figure src="/img/kendall_geometric_2017-fig1-fake.jpg" 
           title="Architecture." >}}

Picture shamelessly cannibalized without permission from
{{< cite walch_imagebased_2016 >}}.

## Improving the loss

The simplest loss function combining training for both quantities, is a simple 
linear combination including a hyperparameter $\beta$ interpolating the losses 
at each of the final regression layers:

\\[ \mathcal{L}=\mathcal{L}\_{p} + \beta \mathcal{L}\_{q}, \\]

where both $\mathcal{L}\_{p}$ and $\mathcal{L}\_{q}$ are simple $L^2$ losses, 
modulo a little technicality with quaternions.[^5] However, this extraneous 
$\beta$ can be avoided using so-called **homoscedastic uncertainty**, with a 
Laplace likelihood. Basically it amounts to assuming a model $p 
(\boldsymbol{y}|\boldsymbol{x}) \sim \mathcal{N} (f (\boldsymbol{x}), \sigma)$ 
where $f$ is the neural network, $\boldsymbol{x}$ the input image and 
$\boldsymbol{y}$ is one of $\boldsymbol{p}, \boldsymbol{q}$. We leave the 
details of this idea for a later post on a previous paper,[^6] but for now, 
suffice to say that it boils down to adding two *trainable parameters* 
$\hat{\sigma}\_{p}, \hat{\sigma}\_{q} \in \mathbb{R}$ to the model and 
optimising

\\[ \mathcal{L}\_{\sigma} =\mathcal{L}\_{p} \mathrm{e}^{- \hat{s}\_{p}} +
   \hat{s}\_{p} +\mathcal{L}\_{q} \mathrm{e}^{- \hat{s}\_{q}} + \hat{s}\_{q},
\\]

where for numerical stability reasons the actual parameters learned are 
$s\_{i} := \log \sigma^2\_{i}$. As the notation suggest and was hinted at 
above, these parameters represent the variance in position and rotation of the 
outputs of the model:

> This loss consists of two components; the residual regressions and the 
> uncertainty regularization terms. We learn the variance, $\sigma^2$, 
> implicitly from the loss function. As the variance is larger, it has a 
> tempering effect on the residual regression term; larger variances (or 
> uncertainty) results in a smaller residual loss. The second regularization 
> term prevents the network from predicting infinite uncertainty (and therefore 
> zero loss)

## Geometric reprojection

A better loss function specific to this domain can be computed based on the 
predicted camera coordinates: the transformation between true and predicted 
camera position is computed, then the 3D test data is reprojected onto the 2D 
image from the second and the distance is computed. This is a natural way of 
combining translation and rotation on a per-image basis: in outdoor images the 
balance will be different from indoor images where objects are typically 
closer. Ditto for camera parameters.

If $\pi \_{K} (\boldsymbol{p}, \boldsymbol{q}, \boldsymbol{x}) = (u, v)$ is 
the projection of a point $\boldsymbol{x} \in \mathbb{R}^3$, with $K$ being the 
camera intrisics, the loss on an image $I$, with visible 3D points $G'$ is

\\[ \mathcal{L}\_{g} (I) = \frac{1}{| G' |}  \sum\_{x_i \in G'} \| \pi
   (\boldsymbol{p}, \boldsymbol{q}, \boldsymbol{x}\_{i}) - \pi
   (\hat{\boldsymbol{p}}, \hat{\boldsymbol{q}}, \boldsymbol{x}\_{i}) \| . \\]

We omitted the index $K$ because the same projection is applied using ground 
truth and predicted camera coordinates so the camera intrinsics can be taken to 
be the identity.

It is noteworthy that the authors observed that this loss can fail to converge 
when initialised with random weights (because of high sensitivity to large 
residuals), but provided the best results when used to fine tune a model 
pretrained with the homoscedastic uncertainty loss.

## Results

As usual, check the paper for the benchmarks and the details of how it 
outperforms everyone else ;). We only remark that this model improves the 
already impressive results of a previous paper using PoseNet with spatial
LSTMs,[^7] but since this paper uses a linear combination for the loss,
the fruit is hanging really low! Also, stay tuned:

> For many applications which require localization, such as mobile robotics, 
> video data is readily available. Ultimately, we would like to extend the 
> architecture to video input with further use of multi-view stereo.


[^1]: The basic idea is explained in {{< cite kendall_what_2017 >}}.
[^2]: Note too that LIDAR is not an option for ships due to the long ranges required and radar has too low resolution.
[^3]: {{< cite walch_imagebased_2016 >}}.
[^4]: {{< cite szegedy_going_2015 >}}.
[^5]: For each rotation, there are two quaternions $\boldsymbol{q}$ and $-\boldsymbol{q}$ representing it. For this reason the authors constrain the optimization to happen on a half-space.
[^6]: {{< cite kendall_multitask_2017 >}} but also {{< cite kendall_what_2017 >}}.
[^7]: {{< cite walch_imagebased_2016 >}}.
