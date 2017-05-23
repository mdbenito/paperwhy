---
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: 2017-05-15
title:  Deep Learning using linear Support Vector Machines
tags: ["deep-learning", "svms"]

paper_authors: ["Tang, Yichuan"]
paper_key: tang_deep_2013
---

The author proposes to replace a **linear SVM** for the softmax atop
some architectures, then **backpropagate the error of the primal
problem to the whole network** . This idea had already been proposed
in the literature but with a standard hinge loss instead of the
$L^2$-loss that the author uses.[^1] Because an $L^2$ loss penalizes
mistakes more heavily than the standard hinge loss the author believes
that:

>_the performance gain is largely due to the superior regularization
>effects of the SVM loss function, rather than an advantage from
>better parameter optimization._

Two natural questions pop up:

1.  Will using an SVM instead of a softmax help networks which already
    are heavily regularized? Note for instance that dropout seems
    **not** to have been used for the paper.

2.  If the softmax seems to be an integral part of the reason why deep
    learning works so well (Lin & Tegmark, 2016)[^2], how is it that
    an SVM works in some situations?

As to the implementation details, a _one-vs rest_ multi-class SVM is
directly substituted for the softmax layer:

>_For $K$ class problems, $K$ linear SVMs will be trained
>independently, where the data from the other classes form the
>negative cases._

Then the class with respect to which a given sample has maximal margin
is taken to be the correct one. Note that this has the immediate
disadvantage wrt. softmax, characteristic of SVMs, that the values
obtained cannot be interpreted as probabilities anymore since the
outputs $a_k (x) = w^{\top} x, k \in [K]$ of the SVM are not
normalized.  How does learning proceed? By backpropagating the error
of the (unconstrained, primal) SVM's objective:

$$ l (w ; \mathrm{x}, \mathrm{t}) = \underset{w}{\min} \frac{1}{2} | w
   |^2 + C \sum_{n = 1}^N \max (1 - w^{\top} x_n t_n, 0)^2, $$

where $(x_n, t_n) \in \mathbb{R}^d \times \{ - 1, 1 \}$ are the
outputs from the last layer and the training labels respectively and
$w$ are the weights for the SVM. Note the square after the maximum:
because the arguments of the max are linear, the whole function is
differentiable with respect to each $x_n$. The gradient is:

$$ \nabla_x l (w ; x, t) = - 2 Ct_n w \max (1 - w^{\top} xt, 0) $$

Using this idea the author won the Facial Expression Recognition
challenge at ICML 2013:

>_(…) using a simple Convolutional Neural Network with linear
>one-vs-all SVM at the top. Stochastic gradient descent with momentum
>is used for training and several models are averaged to slightly
>improve the generalization capabilities._

The details of the architecture are not clear, but the author reports

>_(…) using an 8 split/fold cross validation, with a image mirroring
>layer, similarity transformation layer, two convolutional
>filtering + pooling stages, followed by a fully connected layer with
>3072 hidden penultimate hidden units. The hidden layers are all of
>the rectified linear type. other hyperparameters such as weight decay
>are selected using cross validation._

MNIST is another dataset where good results are obtained: first PCA
down the data to 70 dimensions, then a shallow 512-512 network with an
$L^2$-SVM atop (or softmax for comparison). Learning is done as usual
with SGD with minibatch updates. More interestingly, a stronger
regularization than that provided by the $L^2$-SVM alone was needed
here:

>_To prevent overfitting and critical to achieving good results, a lot
>of Gaussian noise is added to the input._

With this setup, the network with $L^2$-SVM performed around 12%
better than with softmax. It is however noteworthy that no other
regularization techniques nor architectures were tested.

We now come to the more interesting question of why the method
works. Is it a form of regularization or is the network easier to
optimize? To test this

>_looked at the two final models' loss under its own objective
>functions as well as the other objective. [Table 3]_

<table style="width: 100%">
  <tbody><tr>
    <td style="text-align: center; padding-left: 0em; padding-right: 0em"><table style="display: inline; vertical-align: -2.2em">
      <tbody><tr>
        <td style="padding: 0.4em"></td>
        <td style="padding: 0.4em; text-align: center" bgcolor="#dfdfdf"><p>
          ConvNet+Softmax 
        </p></td>
        <td style="padding: 0.4em; text-align: center" bgcolor="#dfdfdf">ConvNet+SVM </td>
      </tr><tr>
        <td style="padding: 0.4em" bgcolor="#dfdfdf">Test error</td>
        <td style="padding: 0.4em; text-align: center"><p>
          14.0% 
        </p></td>
        <td style="padding: 0.4em; text-align: center">11.9% </td>
      </tr><tr>
        <td style="padding: 0.4em" bgcolor="#dfdfdf">Avg. cross entropy</td>
        <td style="padding: 0.4em; text-align: center"><p>
          0.072 
        </p></td>
        <td style="padding: 0.4em; text-align: center">0.353</td>
      </tr><tr>
        <td style="padding: 0.4em" bgcolor="#dfdfdf">Hinge loss squared</td>
        <td style="padding: 0.4em; text-align: center"><p>
          213.2 
        </p></td>
        <td style="padding: 0.4em; text-align: center">0.313</td>
      </tr></tbody>
    </table></td>
  </tr><tr>
    <td style="text-align: center; padding-left: 0em; padding-right: 0em; height: 0.5em"></td>
  </tr><tr>
    <td style="text-align: center; padding-left: 0em; padding-right: 0em; padding-left: 1.5em; padding-right: 1.5em"><p>
      <font size="-1"><p>
        <b>Table 3. </b><a id="auto-1"></a>Training objective including the weight
        costs.
      </p></font>
    </p></td>
  </tr></tbody>
</table>

Note how lower cross entropy actually had a higher error. Perhaps more
interestingly the author

>_also initialized a ConvNet+Softmax model with the weights of the
>[ConvNet+SVM] that had 11.9% error. As further training is performed,
>the network's error rate gradually increased towards 14%._

Which suggests that the $L^2$-SVM provides a better objective,
probably through its regularization property.

[^1]: Some other prior work had been to train convnet (un)supervised, then use the output as input features for a SVM (but then training of the convnet is decoupled from the SVM's objective function); train multiple stacked SVMs recursively (without joint fine-tuning).
[^2]: Lin, H. W. & Tegmark, M. Why does deep and cheap learning work so well? _arXiv:1608.08225 [cond-mat, stat]_ (2016).
