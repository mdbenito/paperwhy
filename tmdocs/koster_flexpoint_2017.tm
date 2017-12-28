<TeXmacs|1.99.6>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Flexpoint: an adaptive numerical format for efficient
  training of deep neural networks>|<doc-author|<author-data|<author-name|Köster,
  Urs>>>|<doc-author|<author-data|<author-name|Webb, Tristan
  J.>>>|<doc-author|<author-data|<author-name|Wang,
  Xin>>>|<doc-author|<author-data|<author-name|Nassar,
  Marcel>>>|<doc-author|<author-data|<author-name|Bansal, Arjun
  K.>>>|<doc-author|<author-data|<author-name|Constable, William
  H.>>>|<doc-author|<author-data|<author-name|Elibol, O§uz
  H.>>>|<doc-author|<author-data|<author-name|Gray,
  Scott>>>|<doc-author|<author-data|<author-name|Hall,
  Stewart>>>|<doc-author|<author-data|<author-name|Hornof,
  Luke>>>|<doc-author|<author-data|<author-name|Khosrowshahi,
  Amir>>>|<doc-author|<author-data|<author-name|Kloss,
  Carey>>>|<doc-author|<author-data|<author-name|Pai, Ruby
  J.>>>|<doc-author|<author-data|<author-name|Rao,
  Naveen>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|efficiency|quantization|hardware>

  <strong|tl;dr:> A method for hardware accelerated, transparent quantization
  of networks both for training and testing is proposed (and later
  implemented in Intel's Nirvana architecture). It consists of fixed point
  operations with an N-bit mantissa and an M-bit exponent
  (<em|flexpointN+M>). The latter is shared for all entries in a single
  tensor, and is managed by the host (as opposed to the ASIC) in software: a
  statistical model predicts future over/underflows based on past values and
  shifts the exponent in order to avoid them. Performance is on par with
  float32 on AlexNet, residual nets and a GAN with no changes to the networks
  themselves nor hyperparameter tuning (for the exponent management
  algorithm).

  <hrule>

  Deep learning is compute- and data-hungry: recent models require dozens or
  even hundreds of GPUs and petabytes of data to train, and this makes
  results irreproducible for most researchers and practitioners. But run-time
  (\Pinference\Q) is also expensive with millions of parameters, which makes
  models unsuited for low-power devices. There are several solutions to this
  acute problem: devise more efficient architectures,<\footnote>
    <cite|zhang_shufflenet_2017>
  </footnote> hand-craft features to ease the task for the network (although
  often we are precisely tying to avoid this), quantize scalars to lower
  precision data types for storage and communication in parallel
  implementations,<\footnote>
    <cite|dettmers_8bit_2015a>.
  </footnote> or in inference and even training,<\footnote>
    For the extreme case see <cite|courbariaux_binarized_2016a>, which use
    binary weights and activations for up to x7 speedups at run time with no
    loss in performance. However, gradients in SGD still must be computed in
    higher precision, so speedups during training are only limited to the
    forward pass and are therefore less noticeable. One reason why
    binarization does not lead to catastrophic failure may be that
    dot-products are almost preserved in high dimensions, see
    <cite|anderson_highdimensional_2017>.
  </footnote> or use lower precision all around.

  <section|The idea>

  Today's paper takes the last approach: they propose a hardware
  implementation of a new datatype, which is something they can do because,
  well, they are intel.<\footnote>
    Note that Google already has their TPU ASICs which use lower precision
    for inference with huge speedups.
  </footnote> The key observation leading to the main idea is simple: the
  whole dynamic range of weights in a typical deep network fits into 16 bits,
  and that is enough as long as one chooses the exponent adequately. So one
  can perform integer arithmetic on a 16 bit type and adjust the exponent
  (represented by 5 bits) as needed to avoid under / overflows. This was
  already observed in the past, and computations had been made in fixed point
  16 bit precision or lower but the corresponding exponents had not been
  predicted with a statistical model as now; instead they were changed after
  overflows occurred.<\footnote>
    In <cite|courbariaux_training_2014a>, a fixed point data type is used.
    When the number of entries in a tensor under- or overflowing exceeds some
    threshold, the exponent for the whole tensor is shifted. The paper we
    study improves this reactive behaviour turning it into a predictive one.
  </footnote>

  <big-figure|<image|../static/img/koster_flexpoint_2017_blog_fig_3.png|0.9par|||>|Dynamic
  range of tensor entries of a ResNet on CIFAR10.>

  Note that even though techniques like Batchnorm<\footnote>
    <cite|ioffe_batch_2015>.
  </footnote> will improve this situation for activations, this phenomenon of
  concentration of tensor entries around their mean happens for weights and
  weight updates as well.

  <section|flexpointN+M>

  In a nutshell, the format <dfn|flexpointN+M> is designed for whole tensors:
  each scalar is represented by

  <\quotation>
    (...) an N-bit mantissa storing an integer value in two's complement
    form, and an M-bit exponent e, shared across all elements of a tensor.
  </quotation>

  It is essential that the exponent is the same for all scalars in a tensor.
  It is updated after each write to the tensor to adjust for possible
  over/underflows. A predictive model for this called Autoflex is implemented
  in a library on the host. The cost of doing this is paid only once per
  tensor, resulting in great gains:

  <\quotation>
    Compared to 32-bit floating point, Flexpoint reduces both memory and
    bandwidth requirements in hardware, as storage and communication of the
    exponent can be amortized over the entire tensor. Power and area
    requirements are also reduced due to simpler multipliers compared to
    floating point. Specifically, multiplication of entries of two separate
    tensors can be computed as a fixed point operation since the common
    exponent is identical across all the output elements. For the same
    reason, addition across elements of the same tensor can also be
    implemented as fixed point operations.
  </quotation>

  So, what this technique achieves is almost fixed point performance for most
  of the operations carried while training and running a network. The authors
  show that 16 bit integer arithmetic plus a dynamically adjusted 5 bit
  exponent provide great speedups while retaining transparency for the
  network designer. \PAll\Q that is required is a library talking to the ASIC
  which changes exponents as required.

  <big-figure|<image|../static/img/koster_flexpoint_2017_blog_fig_2.png|0.9par|||>|Flexpoint16+5.
  >

  <section|Autoflex>

  For each tensor, a deque of maximum absolute values <math|\<Gamma\>> of the
  mantissas is kept. After an update to the tensor at timestep <math|t>
  (either forward or backward pass for a minibatch) the deque is updated and
  the rescaled quantity <math|\<phi\><rsub|t>=\<Gamma\><rsub|t>*\<kappa\><rsub|t>>,
  with <math|\<kappa\><rsub|t>=2<rsup|-e<rsub|t>>> and <math|e<rsub|t>> the
  exponent, is computed. This value is stored and

  <\quotation>
    We maintain a fixed length dequeue <strong|f> of the maximum floating
    point values encountered in the previous <math|l> iterations, and predict
    the expected maximum value for the next iteration based on the maximum
    and standard deviation of values stored in the dequeue. If an overflow is
    encountered, the history of statistics is reset and the exponent is
    increased by one additional bit.
  </quotation>

  Initialisation is a bit tricky since no history is available, so multiple
  guesses are required until valid exponents without overflows and maximising
  utilisation of the mantissa are found.

  The paper concludes with simulations run on GPUs comparing performance. As
  announced flex16+5 is on par with float32.

  <big-figure|<image|../static/img/koster_flexpoint_2017_blog_fig_7.png|0.9par|||>|Flexpoint
  versus floating point on AlexNet, ResNet 110 and a Wasserstein GAN.>

  One can only wait expectantly for intel to deliver this awesome piece of
  tech! Hopefully they will be able to integrate Autoflex into known deep
  learning frameworks.

  If this has piqued your interest, do read their \ <hlink|blog post
  accompanying the paper|https://www.intelnervana.com/flexpoint-numerical-innovation-underlying-intel-nervana-neural-network-processor/>.
  It is very nicely written and provides very clear explanations to the ideas
  in the paper.

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|6>
      <bibitem*|1><label|bib-anderson_highdimensional_2017>Alexander<nbsp>G.<nbsp>Anderson<localize|
      and >Cory<nbsp>P.<nbsp>Berg. <newblock>The High-Dimensional Geometry of
      Binary Neural Networks. <newblock><with|font-shape|italic|ArXiv:1705.07199
      [cs]>, may 2017.<newblock>

      <bibitem*|2><label|bib-courbariaux_training_2014a>Matthieu Courbariaux,
      Yoshua Bengio<localize|, and >Jean-Pierre David. <newblock>Training
      deep neural networks with low precision multiplications.
      <newblock><with|font-shape|italic|ArXiv:1412.7024 [cs]>, dec
      2014.<newblock>

      <bibitem*|3><label|bib-courbariaux_binarized_2016a>Matthieu
      Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv<localize|, and
      >Yoshua Bengio. <newblock>Binarized Neural Networks: Training Deep
      Neural Networks with Weights and Activations Constrained to +1 or -1.
      <newblock><with|font-shape|italic|ArXiv:1602.02830 [cs]>, feb
      2016.<newblock>

      <bibitem*|4><label|bib-dettmers_8bit_2015a>Tim Dettmers.
      <newblock>8-Bit Approximations for Parallelism in Deep Learning.
      <newblock><with|font-shape|italic|ArXiv:1511.04561 [cs]>, nov
      2015.<newblock>

      <bibitem*|5><label|bib-ioffe_batch_2015>Sergey Ioffe<localize| and
      >Christian Szegedy. <newblock>Batch Normalization: Accelerating Deep
      Network Training by Reducing Internal Covariate Shift.
      <newblock><with|font-shape|italic|ArXiv:1502.03167 [cs]>,
      <localize|page >11, feb 2015. <newblock>Citecount: 00000.<newblock>

      <bibitem*|6><label|bib-zhang_shufflenet_2017>Xiangyu Zhang, Xinyu Zhou,
      Mengxiao Lin<localize|, and >Jian Sun. <newblock>ShuffleNet: An
      Extremely Efficient Convolutional Neural Network for Mobile Devices.
      <newblock><with|font-shape|italic|ArXiv:1707.01083 [cs]>, jul
      2017.<newblock>
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|preamble|false>
    <associate|save-aux|true>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|2|?>>
    <associate|auto-4|<tuple|2|?>>
    <associate|auto-5|<tuple|3|?>>
    <associate|auto-6|<tuple|3|?>>
    <associate|auto-7|<tuple|3|?>>
    <associate|bib-anderson_highdimensional_2017|<tuple|1|?>>
    <associate|bib-courbariaux_binarized_2016a|<tuple|3|?>>
    <associate|bib-courbariaux_training_2014a|<tuple|2|?>>
    <associate|bib-dettmers_8bit_2015a|<tuple|4|?>>
    <associate|bib-ioffe_batch_2015|<tuple|5|?>>
    <associate|bib-zhang_shufflenet_2017|<tuple|6|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnote-3|<tuple|3|?>>
    <associate|footnote-4|<tuple|4|?>>
    <associate|footnote-5|<tuple|5|?>>
    <associate|footnote-6|<tuple|6|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
    <associate|footnr-3|<tuple|3|?>>
    <associate|footnr-4|<tuple|4|?>>
    <associate|footnr-5|<tuple|5|?>>
    <associate|footnr-6|<tuple|6|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      zhang_shufflenet_2017

      dettmers_8bit_2015a

      courbariaux_binarized_2016a

      anderson_highdimensional_2017

      courbariaux_training_2014a

      ioffe_batch_2015
    </associate>
    <\associate|figure>
      <tuple|normal|<surround|<hidden|<tuple>>||Dynamic range of tensor
      entries of a ResNet on CIFAR10.>|<pageref|auto-1>>

      <tuple|normal|<surround|<hidden|<tuple>>||Autoflex. >|<pageref|auto-3>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Some
      basics> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>