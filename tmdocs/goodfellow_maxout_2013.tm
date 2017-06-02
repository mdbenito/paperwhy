<TeXmacs|1.99.4>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;
  </hide-preamble>

  <doc-data|<doc-title|Maxout Networks>|<doc-author|<author-data|<author-name|Goodfellow,
  Ian J.>>>|<doc-author|<author-data|<author-name|Warde-Farley,
  David>>>|<doc-author|<author-data|<author-name|Mirza,
  Mehdi>>>|<doc-author|<author-data|<author-name|Courville,
  Aaron>>>|<doc-author|<author-data|<author-name|Bengio,
  Yoshua>>>|<doc-running-author|Miguel de Benito Delgado>>

  <strong|tl;dr:> this paper introduced an activation function for deep
  convolutional networks which specifically benefits from regularization with
  dropout<\footnote>
    See <cite|hinton_improving_2012> for the introduction of dropout.
  </footnote> and still has a universal approximation property for continuous
  functions. It is hypothesized that, analogously to ReLUs, the locally
  linear character of these units makes the averaging of the dropout ensemble
  more accurate than with fully non-linear units. Although sparsity of
  representation is lost wrt. ReLUs, backpropagation of errors is improved by
  not clamping to 0, resulting in significant performance gains.

  <hrule>

  Recall the intuition behind dropout: for each training batch, it masks out
  around 50% of the units, thus training a different model / network of the
  <math|2<rsup|N>> possible (albeit <em|all having shared parameters>), where
  <math|N> is the total number of units in the network. Consequently it is
  benefficial to use <with|font-series|bold|higher learning rates> in order
  to make each one of the models profit as much as possible from the batch it
  sees. But then at test time one needs either to sample from the whole
  ensemble again by using dropout or to use some averaging trick. We recently
  saw<\footnote>
    <cite|hinton_improving_2012>.
  </footnote> that simply scaling the outputs actually approximates the
  expected output of an ensemble of shallow, toy networks, but at the time
  there was little rigorous work on the averaging properties of
  dropout<\footnote>
    But see later developments, e.g. <cite|baldi_dropout_2014>.
  </footnote>

  <\quotation>
    Explicitly designing models to minimize this approximation error may thus
    enhance dropout's performance
  </quotation>

  Hence the idea of <strong|maxout>: define a new activation function

  <\equation>
    <label|eq:def-maxout>h<rsub|i><around*|(|x|)>=<below|max|j\<in\><around*|[|k|]>>z<rsub|i
    j>,x\<in\>\<bbb-R\><rsup|d>,i\<in\><around*|[|m|]>
  </equation>

  where <math|z\<in\>\<bbb-R\><rsup|m\<times\>k>> is a collection of <math|k>
  affine maps computed as <math|z<rsub|i j>=x<rsub|l>*W<rsub|l i j>+b<rsub|i
  j>> (summation convention) and <math|W\<in\>\<bbb-R\><rsup|d\<times\>m\<times\>k>,b\<in\>\<bbb-R\><rsup|m\<times\>k>>
  are to be learned. So

  <\quotation>
    In a convolutional network, a maxout feature map can be constructed by
    taking the maximum across <math|k> affine feature maps (i.e., pool across
    channels, in addition [to] spatial locations).<\footnote>
      Quick reminder: in a convnet max pooling the input on one channel (or
      <em|slice>) consists of applying the following filter:

      <big-figure|<image|../static/img/goodfellow_maxout_2013-fig2.jpg|700px|||>|From
      <hlink|CS231n|//cs231n.github.io> at Stanford.>
    </footnote>
  </quotation>

  The <strong|key observation> here with respect to the approximation
  properties of maxout networks is the fact that since we are taking the max
  over a family of affine functions, the graph of the resulting function is a
  convex set, so with maxout units we are producing piecewise linear (PWL)
  convex functions:

  <\big-figure|<image|../static/img/goodfellow_maxout_2013-fig1.jpg|961px|||>>
    The dotted lines are the affine filters <math|z<rsub|i>>. The epigraph is
    convex.
  </big-figure>

  Because we have

  <\quotation>
    <\theorem>
      Any continuous PWL function can be expressed as a difference of two
      convex PWL functions [of the form <eqref|eq:def-maxout>]. The proof is
      given in <cite|wang_general_2004>.
    </theorem>
  </quotation>

  and because PWL functions approximate continuous ones over compact sets by
  Stone-Weierstrass, it immediately follows that <strong|maxout networks are
  universal approximators>. Of course we can't say anything about rates of
  convergence, so this statement, though necessary is not exactly powerful.

  It is important to note that because they don't clamp to 0 like ReLUs do,\ 

  <\quotation>
    The representation [maxout units produce] is <strong|not sparse> at all
    (...), though the gradient is highly sparse and dropout will artificially
    sparsify the effective representation during training.
  </quotation>

  After extensive cross-validated benchmarking where maxout basically
  outperforms everyone (see the paper for the now
  not-so-much-state-of-the-art results) at MNIST, CIFAR10, CIFAR100 and SVHN
  we come to the question of why it performs so much better than ReLUs.

  The first aspect is the <strong|number of parameters> required: maxout
  performs better with more filters, while ReLUs with more outputs and the
  same number of filters (say, <math|k>). But since cross-channel pooling
  typically reduces the amount of parameters for the next layer

  <\quotation>
    the size of the state and the number of parameters must be about <math|k>
    times higher for rectifiers to obtain generalization performance
    approaching that of maxout.
  </quotation>

  The second aspect is the <strong|good interplay with dropout> and model
  averaging. The fundamental observation now is that

  <\quotation>
    dropout training encourages maxout units to have large linear regions
    around inputs that appear in the training data.
  </quotation>

  The intuitive idea is that these large regions make it relatively rare that
  the maximal filter selected changes when the dropout mask does, and given
  the conjecture (?) that

  <\quotation>
    dropout does exact model averaging in deeper architectures provided that
    they are locally linear among the space of inputs to each layer that are
    visited by applying different dropout masks,
  </quotation>

  it seems plausible that maxout units improve the ability to optimize when
  using dropout. However, this is also true of ReLUs and indeed

  <\quotation>
    The only difference between maxout and max pooling over a set of
    rectified linear units is that maxout does not include a 0 in the max.
  </quotation>

  The experiments in Ÿ8.1 show however that this clamping impedes the
  optimization process and indicate why maxout units are easier to optimize
  than ReLUs. This observation was already done by <cite|glorot_deep_2011>:
  dropout induces sparsity (saturation at 0 for ReLUs) and backprop stops at
  saturated units, but

  <\quotation>
    Maxout does not suffer from this problem because gradient always flows
    through every maxout unit \Ueven when a maxout unit is 0, this 0 is a
    function of the parameters and may be adjusted.
  </quotation>

  It is interesting to see how the experiments where designed in order to
  single out characteristics of the optimization:

  <\enumerate>
    <item>Train a small network on a large dataset. Lack of parameters will
    make it hard to fit the training set.

    <item>Train a deep and narrow model on MNIST. Vanishing gradients (both
    for numerical reasons and because of the clamping to 0 blocking
    gradients) will make optimization hard.<\footnote>
      A few years later, skip connections as in RNNs where proposed for
      convolutional ones in so-called Residual Networks
      <cite|he_deep_2016|he_identity_2016>.
    </footnote>

    <item>Train two-layer MLPs with 1200 filters per layer and 5-channel
    max-pooling: adding a constant 0 deactivates units and degrades
    performance over simply taking the max.
  </enumerate>

  A final noteworthy test to keep in mind is keeping track of the
  <strong|variances of the activations>. Maxout networks enjoyed much higher
  variance at lower layers than ReLU networks: an indication of the vanishing
  gradient problem.

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|6>
      <bibitem*|1><label|bib-baldi_dropout_2014>Pierre Baldi<localize| and
      >Peter Sadowski.<newblock> The Dropout Learning Algorithm.<newblock>
      <with|font-shape|italic|Artificial intelligence>, 210:78\U122, may
      2014.<newblock>

      <bibitem*|2><label|bib-glorot_deep_2011>Xavier Glorot, Antoine
      Bordes<localize|, and >Yoshua Bengio.<newblock> Deep Sparse Rectifier
      Neural Networks.<newblock> <localize|In
      ><with|font-shape|italic|Proceedings of the Fourteenth International
      Conference on Artificial Intelligence and Statistics April 11-13, 2011,
      Fort Lauderdale, FL, USA>, <localize|volume><nbsp>15, <localize|pages
      >315\U323. Florida, USA, apr 2011.<newblock>

      <bibitem*|3><label|bib-he_deep_2016>Kaiming He, Xiangyu Zhang, Shaoqing
      Ren<localize|, and >Jian Sun.<newblock> Deep Residual Learning for
      Image Recognition.<newblock> <localize|Pages >770\U778. 2016.<newblock>
      Extended version with appendix from the Arxiv.<newblock>

      <bibitem*|4><label|bib-he_identity_2016>Kaiming He, Xiangyu Zhang,
      Shaoqing Ren<localize|, and >Jian Sun.<newblock> Identity Mappings in
      Deep Residual Networks.<newblock> <with|font-shape|italic|ArXiv:1603.05027
      [cs]>, <localize|page >15, mar 2016.<newblock> ArXiv:
      1603.05027.<newblock>

      <bibitem*|5><label|bib-hinton_improving_2012>Geoffrey<nbsp>E.<nbsp>Hinton,
      Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever<localize|, and
      >Ruslan<nbsp>R.<nbsp>Salakhutdinov.<newblock> Improving neural networks
      by preventing co-adaptation of feature detectors.<newblock>
      <with|font-shape|italic|ArXiv:1207.0580 [cs]>, <localize|page >18, jul
      2012.<newblock> ArXiv: 1207.0580.<newblock>

      <bibitem*|6><label|bib-wang_general_2004>Shuning Wang.<newblock>
      General constructive representations for continuous piecewise-linear
      functions.<newblock> <with|font-shape|italic|IEEE Transactions on
      Circuits and Systems I: Regular Papers>, 51(9):1889\U1896, sep
      2004.<newblock>
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|font|stix>
    <associate|font-base-size|11>
    <associate|global-subject|activations dropout regularization
    deep-networks>
    <associate|global-title|Maxout Networks>
    <associate|math-font|math-stix>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|3|?>>
    <associate|bib-baldi_dropout_2014|<tuple|1|?>>
    <associate|bib-glorot_deep_2011|<tuple|2|?>>
    <associate|bib-he_deep_2016|<tuple|3|?>>
    <associate|bib-he_identity_2016|<tuple|4|?>>
    <associate|bib-hinton_improving_2012|<tuple|5|?>>
    <associate|bib-wang_general_2004|<tuple|6|?>>
    <associate|eq:def-maxout|<tuple|1|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnote-3|<tuple|3|?>>
    <associate|footnote-4|<tuple|4|?>>
    <associate|footnote-5|<tuple|5|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
    <associate|footnr-3|<tuple|3|?>>
    <associate|footnr-4|<tuple|1|?>>
    <associate|footnr-5|<tuple|5|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      hinton_improving_2012

      hinton_improving_2012

      baldi_dropout_2014

      wang_general_2004

      glorot_deep_2011

      he_deep_2016

      he_identity_2016
    </associate>
    <\associate|figure>
      <tuple|normal|From <locus|<id|%150DBD9C8-1398AD6A0>|<link|hyperlink|<id|%150DBD9C8-1398AD6A0>|<url|//cs231n.github.io>>|CS231n>
      at Stanford.|<pageref|auto-1>>

      <\tuple|normal>
        The dotted lines are the affine filters
        <with|mode|<quote|math>|z<rsub|i>>. The epigraph is convex.
      </tuple|<pageref|auto-2>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>