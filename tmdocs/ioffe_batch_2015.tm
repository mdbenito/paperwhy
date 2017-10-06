<TeXmacs|1.99.5>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Batch normalization: accelerating deep network
  training by reducing internal covariate
  shift>|<doc-author|<author-data|<author-name|Ioffe,
  Sergey>>>|<doc-author|<author-data|<author-name|Szegedy,
  Christian>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|deep-learning|regularization|normalization|optimization>

  <strong|tl;dr:> Normalization to zero mean and unit variance of layer
  outputs in a deep model vastly improves learning rates and yields
  improvements in generalization performance. Approximating the full sample
  statistics by mini-batch ones is effective and computationally manageable.
  You should be doing it too.

  <hrule>

  <subsection*|Covariate shift and whitening>

  For any procedure learning a function <math|f> from random data
  <math|X\<sim\>\<bbb-P\><rsub|X>> it is essential that the distribution
  itself does not vary along the learning process.<\footnote>
    Think e.g. of any PAC generalization bounds: even in these worst-case
    estimates, the sampling distribution, albeit arbitrary, has to be fixed.
  </footnote> When it does, one says that there is <dfn|covariate shift>, a
  phenomenon which one clearly wishes to avoid or mitigate.<\footnote>
    <cite|shimodaira_improving_2000> introduces the term covariate shift for
    the difference between the distribution of the training data and the test
    data, the first being typically heavily conditioned by the sampling
    method and the second being the \Ptrue\Q population distribution. The
    authors of the current paper extend its meaning to a continuous
    \Pshifting under the feet\Q of the training distribution.
  </footnote> One possibility is to \Pfix\Q the first two moments of
  <math|\<bbb-P\><rsub|X>> by <strong|whitening>: the transformation on the
  full sample data

  <\equation*>
    \<b-x\>\<mapsto\><wide|\<Sigma\>|^><rsup|-1/2>*<around*|(|\<b-x\>-<wide|\<b-mu\>|^>|)>
  </equation*>

  subtracting the sample average and multiplying by the inverse covariance
  matrix centers the data around 0 and decorrelates features to have unit
  variance and vanishing covariance (assuming positive definiteness). This is
  long known to yield faster convergence rates.<\footnote>
    <cite|lecun_efficient_1998> already discussed using mean normalization in
    neural networks, as well as many of its properties, together with
    whitening.
  </footnote>

  Consider now a general <dfn|feed forward neural network>

  <\equation*>
    f=f<rsub|L>\<circ\>\<cdots\>\<circ\>f<rsub|1>
  </equation*>

  with arbitrary (non linear) layers <math|f<rsub|l>=f<rsub|l><around*|(|\<cdot\>;\<theta\><rsub|l>|)>>.
  Updates to the layer parameters <math|\<theta\><rsub|1>,\<ldots\>,\<theta\><rsub|L>>
  during training will alter the mean, variance and other statistics of the
  <em|activations> of each layer <math|f<rsub|l>> acting as input for layer
  <math|f<rsub|l+1>>, that is: the distributions
  <math|f<rsub|l><around*|(|<wide|X|~>;\<theta\><rsub|l>|)>> will shift, or,
  in other words, the <em|subnetwork> experiences covariate shift. One says
  that the (full) network suffers <dfn|internal covariate shift>.

  So even if we do the usual normalization of the training data (the input to
  <math|f<rsub|1>>) and initialise all the weights to have 0 mean and unit
  variance, the distributions <math|f<rsub|1><around*|(|X|)>,<around*|(|f<rsub|2>\<circ\>f<rsub|1>|)><around*|(|X|)>,\<ldots\>>
  will shift as training progresses. This is bad enough for learning itself,
  but it will have further negative impact in networks using saturating
  activations like sigmoids, since it will tend to move them into saturating
  regimes where learning stops.

  In today's paper, the authors propose a method of approximately whitening
  each layer with reduced computational cost.

  <subsection*|Batch normalization>

  In order to improve training one would like to whiten all activations by
  interspersing <math|L-1> additional layers

  <\equation*>
    g<rsub|l><around*|(|x,\<theta\><rsub|l>|)>=<wide|\<Sigma\>|^><rsub|l><rsup|-1/2>*<around*|(|x-<wide|\<mu\>|^><rsub|l>|)>,
  </equation*>

  where <math|<wide|\<mu\>|^><rsub|l>> and <math|<wide|\<Sigma\>|^><rsub|l>>
  are the full sample mean and covariance, <em|taking into account the
  network parameters <math|\<theta\><rsub|l>> <around*|(|up to layer
  <math|l>|)> distorting the training data>. In the case of a linear network,
  the transformation <math|f<rsub|l><around*|(|x|)>=W*x+b> maps the random
  input <math|X<rsub|l>> to <math|X<rsub|l+1>> by shifting its mean by
  <math|b> and scaling the covariance <math|C<rsub|X<rsub|l>>> to
  <math|C<rsub|X<rsub|l+1>>=W*C<rsub|X<rsub|l>>*W<rsup|\<top\>>>. This
  transformation only affects the first two moments, so that it can be undone
  by whitening. When one adds nonlinear effects, it is hoped that this first
  order approximation will be enough to keep the distribution of
  <math|X<rsub|l+1>> under control.

  It is clear that computing these quantities is utterly impractical: they
  change for each layer after each parameter update and depend on the full
  training data. Note however that the \Pobvious\Q simplification of ignoring
  the effect of <math|\<theta\><rsub|l>> and taking statistics only over the
  training data, instead of over the intermediate activations can lead to
  layers not updating their parameters even for nonzero gradients.<\footnote>
    \ Details in Ÿ2 of the paper.
  </footnote> For this reason the authors propose two simplifications:

  <\enumerate>
    <item>Normalize each component <math|x<rsup|<around*|(|k|)>>> of an
    activation independently

    <\equation*>
      Norm<around*|(|x|)><rsup|<around*|(|k|)>>=<frac|x<rsup|<around*|(|k|)>>-<wide|\<mu\>|^><rsup|<around*|(|k|)>>|<wide|\<sigma\>|^><rsup|<around*|(|k|)>>>.
    </equation*>

    This avoids computing covariance matrices and still improves convergence
    even if there are cross-correlations among features.

    <item>Compute statistics <math|\<mu\><rsub|l,B>> and
    <math|\<sigma\><rsub|l,B>> at each layer <math|l> <em|for SGD
    mini-batches> <math|B=<around*|{|x<rsub|B<rsub|1>>,\<ldots\>,x<rsub|B<rsub|n>>|}>>
    instead of over the full sample (these are rough approximations to the
    \Ptrue\Q statistics <math|\<mu\><rsub|l>,\<sigma\><rsub|l>> at a layer
    with <em|fixed> parameters).
  </enumerate>

  In order to have a functioning method, there is an important addition to
  make: because simply normalizing activations changes the regime in which
  the next layer operates,<\footnote>
    E.g. by limiting the input to a sigmoid to be
    <math|\<cal-N\><around*|(|0,1|)>> it will roughly operate in its linear
    regime around 0.
  </footnote> two parameters <math|\<gamma\><rsub|l>,\<beta\><rsub|l>\<in\>\<bbb-R\><rsup|d>>
  are added to allow for linear scaling of the normalized activations, in
  principle enabling the undoing of the normalization.<\footnote>
    <math|\<gamma\><rsub|l>> and <math|\<beta\><rsub|l>> will be learnt along
    all other parameters <math|\<theta\>> whereas the precise batch mean
    <math|\<mu\><rsub|l,B>> and variance <math|\<sigma\><rsub|l,B>> vary with
    each training step since they depend on the parameters <math|\<theta\>>
    of each layer and the minibatch. If they were good estimators of some
    \Ptrue\Q population moments <math|\<mu\><rsub|l>,\<sigma\><rsub|l>> then
    we could say that the batch normalization layer could become an identity
    if the optimization required it, but it is not clear what this true
    distribution would be since it changes at each point in parameter space.
    Also, even if we only consider <math|\<mu\><rsub|l>,\<sigma\><rsub|l>> at
    local minima for the energy, where the network is supposed to converge,
    there can be many of them...
  </footnote>

  The final <dfn|batch normalization layer> looks like

  <\equation*>
    BN<rsub|l><around*|(|x<rsup|<around*|(|k|)>>|)>=<frac|x<rsup|<around*|(|k|)>>-<wide|\<mu\>|^><rsub|l,B><rsup|<around*|(|k|)>>|<sqrt|<wide|\<sigma\>|^><rsub|l,B><rsup|<around*|(|k|)>>+\<varepsilon\>>>*\<gamma\><rsub|l><rsup|<around*|(|k|)>>+\<beta\><rsub|l><rsup|<around*|(|k|)>>,
  </equation*>

  where <math|<wide|\<mu\>|^><rsup|<around*|(|k|)>><rsub|l,B>> is the sample
  mean of the <math|k>-th component of the <em|activations> in minibatch
  <math|B>, <math|<wide|\<sigma\>|^><rsub|l,B><rsup|<around*|(|k|)>>> is the
  sample variance of the same component, and <math|\<varepsilon\>\<gtr\>0>
  avoids divisions by too small numbers. It is important to stress the fact
  that we are not computing statistics over the training data but <em|over
  the activations computed for a given minibatch>, which includes the effect
  of all relevant network parameters.

  <subsection*|Test time>

  At each training step <math|t> we have normalized each layer using
  \Plocal\Q batch mean and variances, which again, depended on the current
  parameters <math|\<theta\><rsub|l><rsup|t>> of the network. In the limit
  <math|\<theta\><rsup|t><rsub|l>\<rightarrow\>\<theta\><rsub|l><rsup|\<star\>>>
  for some (locally) optimal <math|\<theta\><rsup|\<star\>><rsub|l>>, we have
  some fixed population mean and variance of the activations at this layer,
  <math|X<rsub|l>=f<rsub|l><around*|(|X<rsub|l-1>;\<theta\><rsup|\<star\>><rsub|l>|)>>
  which, intuitively, we should use at test time. To estimate these
  quantities, we use can use the pertinent <strong|full-sample statistics,
  rather than mini-batch,> <math|<wide|\<mu\>|^>,<wide|\<sigma\>|^>> for
  layer activations:<\footnote>
    Note that we rewrite the operation as <math|a*x+b> to point out that all
    quantities but <math|x> are constant at test time.
  </footnote>

  <\equation*>
    BN<around*|(|x<rsup|<around*|(|k|)>>|)>=<frac|\<gamma\><rsup|<around*|(|k|)>>|<sqrt|<wide|\<sigma\>|^><rsup|<around*|(|k|)>>+\<varepsilon\>>>*x<rsup|<around*|(|k|)>>+<around*|(|\<beta\><rsup|<around*|(|k|)>>-<frac|\<gamma\><rsup|<around*|(|k|)>>*<wide|\<mu\>|^><rsup|<around*|(|k|)>>|<sqrt|<wide|\<sigma\>|^><rsup|<around*|(|k|)>>+\<varepsilon\>>>|)>.
  </equation*>

  However computing full covariance matrices is typically out of the question
  so we approximate these by averaging different
  <math|<wide|\<mu\>|^><rsub|B>,<wide|\<sigma\>|^><rsub|B>> over multiple
  minibatches, and in practice this is usually done with a moving average
  during training. It is clear that all this requires some rigorous
  developments in order to be fully satisfactory...

  <subsection*|Application, learning rates and regularization>

  The first application is to convolutional nets, in particular a modified
  Inception<\footnote>
    <cite|szegedy_going_2015>.
  </footnote> network for <hlink|ImageNet|http://www.image-net.org/>
  classification. Here normalization is performed <em|before> the
  nonlinearity, because as explained above, the linear layer only alters the
  first order moments of its input so normalization of first moments makes
  more sense there. This has the additional benefit of dispensing with the
  bias parameters since they are subsumed into <math|\<beta\>>. There are
  further details to be taken into account for convnets, see the paper for
  details.

  As already mentioned, BN is conjectured to be advantageous for optimization
  because

  <\quotation>
    it prevents small changes to the parameters from amplifying into larger
    and suboptimal changes in activations in gradients; for instance, it
    prevents the training from getting stuck in the saturated regimes of
    nonlinearities.
  </quotation>

  A further conjecture, based on a heuristic argument assuming Gaussianity,
  is that it

  <\quotation>
    may lead the layer Jacobians to have singular values close to 1, which is
    known to be beneficial for training.<\footnote>
      <cite|saxe_exact_2013>.
    </footnote>
  </quotation>

  Finally, it was experimentally observed that some sort of regularization is
  performed by BN, to the point that Dropout<\footnote>
    <cite|hinton_improving_2012>.
  </footnote> could be entirely omitted in some examples. Again, some
  rigorous work is required here.

  <\subsection*>
    In practice
  </subsection*>

  It is noted that BN alone does not fully exploit its potential. One needs
  to adapt the architecture and optimization by (details in the paper):

  <\itemize>
    <item>Increasing the learning rate and its rate of decay.

    <item>Removing dropout (or reducing the dropout probability) and reducing
    the <math|L<rsup|2>> weight regularization.

    <item>Removing Local Response Normalization.
  </itemize>

  Furthermore, improving the quality shuffling of training examples for
  minibatches (by preventing samples to repeatedly be chosen together) and
  decreasing the intensity of transformations in augmented data proved
  beneficial. The overall results are impressive (best performance at
  ImageNet at the time):

  <\quotation>
    Merely adding Batch Normalization to a state-of-the-art image
    classification model yields a substantial speedup in training. [With the
    modifications mentioned] we reach the previous state of the art with only
    a small fraction of training steps \U and then beat the state of the art
    in single-network image classification. Furthermore, by combining
    multiple models trained with Batch Normalization, we perform better than
    the best known system on ImageNet, by a significant margin.
  </quotation>

  <big-figure|<image|../static/img/ioffe_batch_2015-fig2.png|719px|||>|Single
  crop validation accuracy of Inception and its batch-normalized variants,
  vs. the number of training steps.>

  <subsection*|Some recent developments>

  Since the introduction of BN, several related techniques have been
  developed. Two prominent ones are:

  <\itemize>
    <item><dfn|Layer normalization>: normalize the output of each unit in
    layer <math|l> by the mean and variance of <em|all> other outputs given
    just <em|one> example.<\footnote>
      <cite|ba_layer_2016>.
    </footnote>

    <item><dfn|Weight normalization>: activations are normalized by the norm
    of the weights.<\footnote>
      <cite|salimans_weight_2016>.
    </footnote> Faster but still performant.
  </itemize>

  Twists on BN include:

  <\itemize>
    <item>Diminishing Batch Normalization: <hlink|Convergence Analysis of
    Batch Normalization for Deep Neural Nets|http://arxiv.org/abs/1705.08011v1>,
    Yintai Ma, Diego Klabjan

    <item><hlink|Recurrent Batch Normalization|http://arxiv.org/abs/1603.09025v5>,Tim
    Cooijmans, Nicolas Ballas, César Laurent, Ça§lar Gülçehre, Aaron
    Courville.

    <item>To be updated...
  </itemize>

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|7>
      <bibitem*|1><label|bib-ba_layer_2016>Jimmy<nbsp>Lei Ba, Jamie<nbsp>Ryan
      Kiros<localize|, and >Geoffrey<nbsp>E.<nbsp>Hinton.<newblock> Layer
      Normalization.<newblock> <with|font-shape|italic|ArXiv:1607.06450 [cs,
      stat]>, jul 2016.<newblock> Citecount: 00080 arXiv:
      1607.06450.<newblock>

      <bibitem*|2><label|bib-hinton_improving_2012>Geoffrey<nbsp>E.<nbsp>Hinton,
      Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever<localize|, and
      >Ruslan<nbsp>R.<nbsp>Salakhutdinov.<newblock> Improving neural networks
      by preventing co-adaptation of feature detectors.<newblock>
      <with|font-shape|italic|ArXiv:1207.0580 [cs]>, <localize|page >18, jul
      2012.<newblock> Citecount: 01870 arXiv: 1207.0580.<newblock>

      <bibitem*|3><label|bib-lecun_efficient_1998>Yann LeCun, Léon Bottou,
      Genevieve<nbsp>B.<nbsp>Orr<localize|, and >Klaus-Robert
      Müller.<newblock> Efficient BackProp.<newblock> <localize|In
      ><with|font-shape|italic|Neural networks: tricks of the trade>,
      <localize|number> 1524<localize| in >Lecture Notes in Computer Science,
      <localize|page >0. Springer, 1998.<newblock> Citecount: 01464 DOI:
      10.1007/3-540-49430-8.<newblock>

      <bibitem*|4><label|bib-salimans_weight_2016>Tim Salimans<localize| and
      >Diederik<nbsp>P Kingma.<newblock> Weight Normalization: A Simple
      Reparameterization to Accelerate Training of Deep Neural
      Networks.<newblock> <localize|In >D.<nbsp>D.<nbsp>Lee,
      M.<nbsp>Sugiyama, U.<nbsp>V.<nbsp>Luxburg, I.<nbsp>Guyon<localize|, and
      >R.<nbsp>Garnett<localize|, editors>, <with|font-shape|italic|Advances
      in Neural Information Processing Systems 29>, <localize|pages
      >901\U909. Curran Associates, Inc., 2016.<newblock> Citecount:
      00060.<newblock>

      <bibitem*|5><label|bib-saxe_exact_2013>Andrew<nbsp>M.<nbsp>Saxe,
      James<nbsp>L.<nbsp>McClelland<localize|, and >Surya Ganguli.<newblock>
      Exact solutions to the nonlinear dynamics of learning in deep linear
      neural networks.<newblock> <with|font-shape|italic|ArXiv:1312.6120
      [cond-mat, q-bio, stat]>, dec 2013.<newblock> Citecount: 00174 arXiv:
      1312.6120.<newblock>

      <bibitem*|6><label|bib-shimodaira_improving_2000>Hidetoshi
      Shimodaira.<newblock> Improving predictive inference under covariate
      shift by weighting the log-likelihood function.<newblock>
      <with|font-shape|italic|Journal of statistical planning and inference>,
      90(2):227\U244, 2000.<newblock> Citecount: 00623.<newblock>

      <bibitem*|7><label|bib-szegedy_going_2015>C.<nbsp>Szegedy, Wei Liu,
      Yangqing Jia, P.<nbsp>Sermanet, S.<nbsp>Reed, D.<nbsp>Anguelov,
      D.<nbsp>Erhan, V.<nbsp>Vanhoucke<localize|, and
      >A.<nbsp>Rabinovich.<newblock> Going deeper with
      convolutions.<newblock> <localize|In ><with|font-shape|italic|2015 IEEE
      Conference on Computer Vision and Pattern Recognition (CVPR)>,
      <localize|page >9. Jun 2015.<newblock> Citecount: 03482.<newblock>
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
    <associate|auto-1|<tuple|?|?>>
    <associate|auto-2|<tuple|3|?>>
    <associate|auto-3|<tuple|6|?>>
    <associate|auto-4|<tuple|7|?>>
    <associate|auto-5|<tuple|10|?>>
    <associate|auto-6|<tuple|1|?>>
    <associate|auto-7|<tuple|1|?>>
    <associate|auto-8|<tuple|<with|mode|<quote|math>|\<bullet\>>|?>>
    <associate|bib-ba_layer_2016|<tuple|1|?>>
    <associate|bib-hinton_improving_2012|<tuple|2|?>>
    <associate|bib-lecun_efficient_1998|<tuple|3|?>>
    <associate|bib-salimans_weight_2016|<tuple|4|?>>
    <associate|bib-saxe_exact_2013|<tuple|5|?>>
    <associate|bib-shimodaira_improving_2000|<tuple|6|?>>
    <associate|bib-szegedy_going_2015|<tuple|7|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-10|<tuple|10|?>>
    <associate|footnote-11|<tuple|11|?>>
    <associate|footnote-12|<tuple|12|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnote-3|<tuple|3|?>>
    <associate|footnote-4|<tuple|4|?>>
    <associate|footnote-5|<tuple|5|?>>
    <associate|footnote-6|<tuple|6|?>>
    <associate|footnote-7|<tuple|7|?>>
    <associate|footnote-8|<tuple|8|?>>
    <associate|footnote-9|<tuple|9|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-10|<tuple|10|?>>
    <associate|footnr-11|<tuple|11|?>>
    <associate|footnr-12|<tuple|12|?>>
    <associate|footnr-2|<tuple|2|?>>
    <associate|footnr-3|<tuple|3|?>>
    <associate|footnr-4|<tuple|4|?>>
    <associate|footnr-5|<tuple|5|?>>
    <associate|footnr-6|<tuple|6|?>>
    <associate|footnr-7|<tuple|7|?>>
    <associate|footnr-8|<tuple|8|?>>
    <associate|footnr-9|<tuple|9|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      shimodaira_improving_2000

      lecun_efficient_1998

      szegedy_going_2015

      saxe_exact_2013

      hinton_improving_2012

      ba_layer_2016

      salimans_weight_2016
    </associate>
    <\associate|figure>
      <tuple|normal|Single crop validation accuracy of Inception and its
      batch-normalized variants, vs. the number of training
      steps.|<pageref|auto-6>>
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|Covariate shift and whitening
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|1tab>|Batch normalization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|Test time
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|Application, learning rates and
      regularization <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|In practice
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|Some recent developments
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>