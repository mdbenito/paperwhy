<TeXmacs|1.99.4>

<style|<tuple|generic|paperwhy>>

<\body>
  <doc-data|<doc-title|Identity matters in Deep
  Learning>|<doc-author|<author-data|<author-name|Hardt,
  Moritz>>>|<doc-author|<author-data|<author-name|Ma,
  Tengyu>>>|<doc-running-author|Miguel de Benito Delgado>>

  <tags|deep-learning|representation>

  <strong|tl;dr:> vanilla residual networks are very good approximators of
  functions which can be represented as linear perturbations of the identity.
  In the linear setting, optimization is aided by a benevolent landscape
  having only minima in certain (interesting) regions. Finally, very simple
  ResNets can completely learn datasets with <math|\<cal-O\><around*|(|n*log
  n+\<ldots\>|)>> parameters. All this seems to indicate that deep and simple
  architectures might be enough to achieve great performance.

  <hrule>

  In general, it is hard for classical deep nets to \Ppreserve features which
  are good\Q: initialization with zero mean and small gradients make it hard
  to learn the identity at any given layer. Even though
  <strong|batchnorm><\footnote>
    See <cite|ioffe_batch_2015>.
  </footnote> seeks to alleviate this issue, it has been <strong|residual
  networks> which have most improved upon it.<\footnote>
    See <cite|he_deep_2016>.
  </footnote> In a residual net

  <\quotation>
    (...) each residual layer has the form <math|x+h(x)>, rather than
    <math|h(x)>. This simple reparameterization allows for much deeper
    architectures largely avoiding the problem of vanishing (or exploding)
    gradients.
  </quotation>

  <subsection|Identity parametrizations improve optimization>

  The authors work first in the linear setting, i.e. they consider only
  networks which are compositions of linear perturbations of the
  identity:<\footnote>
    For extensions to the non-linear setting see
    <cite|bartlett_representational_2017>. Note that even though in principle
    the network can be flattened to only one linear map by taking the product
    of all <math|A<rsub|i>> with no loss in its representational capacity, a
    great cost in efficiency can be incurred in doing so, see
    <cite-detail|lin_why_2016|ŸG>. The dynamics of optimization also are
    affected by the stacking of purely linear layers, see e.g.
    <cite|saxe_exact_2013>.
  </footnote>

  <\equation*>
    h<around*|(|x|)>=<around*|(|I+A<rsub|l>|)>**\<cdots\>*<around*|(|I+A<rsub|2>|)>*<around*|(|I+A<rsub|1>|)>*x
  </equation*>

  The objective function to mimize is the <dfn|population risk> with
  quadratic loss:

  <\equation*>
    f<around*|(|A|)>=f<around*|(|A<rsub|1>,\<ldots\>,A<rsub|l>|)>\<assign\>\<bbb-E\><rsub|X,Y><around*|\||Y-<around*|(|I+A<rsub|l>|)>*\<cdots\>*<around*|(|I+A<rsub|1>|)>X|\|><rsup|2>.
  </equation*>

  Labels are assigned with noise, that is <math|Y=R*X+\<xi\>>, with
  <math|\<xi\>\<sim\>\<cal-N\><around*|(|0,I<rsub|d>|)>>. Note that the
  problem over the variables <math|A=<around*|(|A<rsub|1>,\<ldots\>,A<rsub|l>|)>>
  is non-convex.

  The first result of the paper states that deep networks with many layers
  have minima of proportionally low (spectral) norm.<\footnote>
    Essentially, the theorem states that there exists a constant
    <math|\<gamma\>> depending on the largest and smallest singular values of
    <math|R> such that there exists a global minimum <math|A<rsup|\<star\>>>
    of the population risk fulfilling <math|<below|max|1\<leqslant\>i\<leqslant\>l>
    <around*|\<\|\|\>|A<rsup|\<star\>><rsub|i>|\<\|\|\>>\<leqslant\><frac|1|l>*<around*|(|4*\<mathpi\>+3*\<gamma\>|)>>
    whenever the number of layers <math|l\<geqslant\>3\<gamma\>>.
  </footnote> Because of this, it makes sense to study critical points with
  small norm, and it turns out that there are only minima. The main result of
  this section is the following one, where one can think of <math|\<tau\>> as
  being <math|\<cal-O\><around*|(|1/l|)>>:

  <\quotation>
    <\theorem>
      For every <math|\<tau\>\<less\>1>, every critical point <math|A> of the
      objective function <math|f> with

      <\equation*>
        <below|max|1\<leqslant\>i\<leqslant\>l>
        <around*|\<\|\|\>|A<rsub|i>|\<\|\|\>>\<leqslant\>\<tau\>
      </equation*>

      is a global minimum.
    </theorem>
  </quotation>

  This is good news since, under the assumption that the model is correct, a
  \Pgood\Q (in some sense) optimization algorithm will converge to
  it.<\footnote>
    Equation (2.3) of the paper provides a lower bound on the gradient which
    does guarantee convergence under the assumption that iterates don't jump
    out of the domain of interest. See the reference in the paper.
  </footnote> The proof is relatively straightforward too: rewrite the risk
  as the norm of a product by the covariance:
  <math|f<around*|(|A|)>=<around*|\<\|\|\>|E<around*|(|A|)>*\<Sigma\><rsup|1/2>|\<\|\|\>><rsup|2>+C>,
  then, using that <math|<around*|\<\|\|\>|A<rsub|i>|\<\|\|\>>> are small,
  show that if <math|\<nabla\>f<around*|(|A|)>=0> this can only be if
  <math|E<around*|(|A|)>=0>, where <math|E> precisely encodes the condition
  of being at an optimum: <math|E<around*|(|A|)>=<around*|(|I+A<rsub|l>|)>*\<cdots\>*<around*|(|I+A<rsub|1>|)>-R>.

  <subsection|Identity parametrizations improve representation>

  The authors consider next non-linear simplified residual networks with each
  layer of the form

  <\equation>
    <label|eq:residual-unit>h<rsub|j><around*|(|x|)>=x+V<rsub|j>*\<sigma\><around*|(|U<rsub|j>*x+b<rsub|j>|)>
  </equation>

  where <math|V<rsub|j>,U<rsub|j>\<in\>\<bbb-R\><rsup|k\<times\>k>> are
  weight matrices, <math|b<rsub|j>\<in\>\<bbb-R\><rsup|k>> is a bias vector
  and <math|\<sigma\><around*|(|z|)>=max<around*|(|0,z|)>> is a ReLU
  activation.<\footnote>
    This is a deliberately simpler setup than the original ResNets with two
    ReLU activations and two instances of batch normalization. See
    <cite|he_identity_2016>.
  </footnote> Note that <em|the layer size is constant>. No batchnorm is
  applied. The problem is <math|r>-class classification. Assuming (the
  admittedly natural condition) that all the training data are uniformly
  separated by a minimal distance <math|\<rho\>\<gtr\>0>, they prove that
  perfectly learning <math|n> data points is possible with <math|n log n>
  parameters:

  <\quotation>
    <\theorem>
      There exists a residual network with <math|\<cal-O\>(n*log
      n+r<rsup|2>)> parameters that perfectly fits the training data.
    </theorem>
  </quotation>

  By choosing the hyperparameter <math|k\<in\>\<cal-O\><around*|(|log n|)>>
  and <math|l=<around*|\<lceil\>|n/k|\<rceil\>>> the complexity stated is
  obtained with a bit of arithmetic. The proof consists then of a somewhat
  explicit construction of the network. Very roughly, the weight matrices of
  the hidden layers are chosen as to assign each data point to one of
  <math|r> <dfn|surrogate label vectors> <math|q<rsub|1>,\<ldots\>,q<rsub|r>\<in\>\<bbb-R\><rsup|k>>,
  then the last layer converts these to 1-hot label vectors for output. The
  main ingredient is therefore the proof that it is possible to map the
  inputs <math|x<rsub|i>> to the surrogate vectors in a way such that the
  final layer has almost no work left to do.<\footnote>
    An interesting point is the use of the <strong|Johsonn-Lindestrauss
    lemma> to ensure that an initial random projection of input data onto
    <math|\<bbb-R\><rsup|k>> by the first layer does not violate the
    condition that it remains separated, with high probability.
  </footnote>

  This is achieved by showing that:

  <\quotation>
    (...) for an (almost) arbitrary sequence of vectors
    <math|x<rsub|1>,\<ldots\>,x<rsub|n>> there exist [weights <math|U,V,b>]
    such that operation <eqref|eq:residual-unit> transforms <math|k> [of
    them] to an arbitrary set of other <math|k> vectors that we can freely
    choose, and maintains the value of the remaining <math|n-k> vectors.
  </quotation>

  <subsection|Experiments>

  <big-figure|<image|../static/img/hardt_identity_2016-fig1.jpg|1par|||>|Convergence
  plots of best model for CIFAR10 (left) and CIFAR (100) right.>

  Working on CIFAR10 and CIFAR100, the authors tweaked a <hlink|standard
  ResNet architecture|https://github.com/tensorflow/models/tree/master/resnet>
  in Tensorflow to have constant size <math|c> of convolution, no batch norm
  and smaller weight initialization.<\footnote>
    Gaussian with <math|\<sigma\>\<sim\><around*|(|k<rsup|-2>*c<rsup|-1>|)>>
    instead of <math|\<sigma\>\<sim\><around*|(|k<rsup|-1>*c<rsup|-1/2>|)>>.
    For more on proper initialisation see e.g.
    <cite|sutskever_importance_2013>.
  </footnote> Several features stand out:

  <\itemize-dot>
    <item>The last layer is a fixed random projection. Therefore all
    parameters are in the convolutioms.

    <item>Lack of batchnorm or other regularizers seemed not to lead to
    serious overfitting, even though the model had
    <math|\<sim\>13.6\<times\>10<rsup|6>> parameters.

    <item>Both problems where tackled with the same networks and the
    convolutions where of constant size.
  </itemize-dot>

  Finally, results on <strong|ImageNet> were not as bright, though the
  authors seem confident that this is due to lack of tuning of
  hyperparameters and learning rate. It would be interesting to find out how
  true this is and <strong|how much harder this tuning becomes by having
  discarded regularization techniques> like dropout or additional data
  processing.

  <subsection|Extensions>

  For an extension of these results to the non-linear case (which as of this
  writing is reported to be work in progress), be sure to check out
  Bartlett's talk: <cite|bartlett_representational_2017>.

  <\bibliography|bib|tm-ieeetr|paperwhy.bib>
    <\bib-list|7>
      <bibitem*|1><label|bib-ioffe_batch_2015>S.<nbsp>Ioffe<localize| and
      >C.<nbsp>Szegedy, ``Batch Normalization: Accelerating Deep Network
      Training by Reducing Internal Covariate Shift,''
      <with|font-shape|italic|arXiv:1502.03167 [cs]>, p.<nbsp>11, feb
      2015.<newblock> Citecount: 01618 arXiv: 1502.03167.<newblock>

      <bibitem*|2><label|bib-he_deep_2016>K.<nbsp>He, X.<nbsp>Zhang,
      S.<nbsp>Ren<localize|, and >J.<nbsp>Sun, ``Deep Residual Learning for
      Image Recognition,'' (), pp.<nbsp>770\U778, 2016.<newblock> Citecount:
      02156 Extended version with appendix from the Arxiv.<newblock>

      <bibitem*|3><label|bib-bartlett_representational_2017>P.<nbsp>Bartlett,
      ``Representational and optimization properties of Deep Residual
      Networks,'' may 2017.<newblock> Citecount: 00000 2681
      seconds.<newblock>

      <bibitem*|4><label|bib-lin_why_2016>H.<nbsp>W.<nbsp>Lin<localize| and
      >M.<nbsp>Tegmark, ``Why does deep and cheap learning work so well?,''
      <with|font-shape|italic|arXiv:1608.08225 [cond-mat, stat]>, p.<nbsp>17,
      aug 2016.<newblock> Citecount: 00019 arXiv: 1608.08225.<newblock>

      <bibitem*|5><label|bib-saxe_exact_2013>A.<nbsp>M.<nbsp>Saxe,
      J.<nbsp>L.<nbsp>McClelland<localize|, and >S.<nbsp>Ganguli, ``Exact
      solutions to the nonlinear dynamics of learning in deep linear neural
      networks,'' <with|font-shape|italic|arXiv:1312.6120 [cond-mat, q-bio,
      stat]>, dec 2013.<newblock> Citecount: 00174 arXiv:
      1312.6120.<newblock>

      <bibitem*|6><label|bib-he_identity_2016>K.<nbsp>He, X.<nbsp>Zhang,
      S.<nbsp>Ren<localize|, and >J.<nbsp>Sun, ``Identity Mappings in Deep
      Residual Networks,'' <with|font-shape|italic|arXiv:1603.05027 [cs]>,
      p.<nbsp>15, mar 2016.<newblock> Citecount: 00247 arXiv:
      1603.05027.<newblock>

      <bibitem*|7><label|bib-sutskever_importance_2013>I.<nbsp>Sutskever,
      J.<nbsp>Martens, G.<nbsp>Dahl<localize|, and >G.<nbsp>E.<nbsp>Hinton,
      ``On the importance of initialization and momentum in deep learning,''
      <localize|in ><with|font-shape|italic|PMLR>, pp.<nbsp>1139\U1147, feb
      2013.<newblock> Citecount: 00593.<newblock>
    </bib-list>
  </bibliography>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|3|?>>
    <associate|auto-4|<tuple|1|?>>
    <associate|auto-5|<tuple|4|?>>
    <associate|auto-6|<tuple|4|?>>
    <associate|bib-bartlett_representational_2017|<tuple|3|?>>
    <associate|bib-he_deep_2016|<tuple|2|?>>
    <associate|bib-he_identity_2016|<tuple|6|?>>
    <associate|bib-ioffe_batch_2015|<tuple|1|?>>
    <associate|bib-lin_why_2016|<tuple|4|?>>
    <associate|bib-saxe_exact_2013|<tuple|5|?>>
    <associate|bib-sutskever_importance_2013|<tuple|7|?>>
    <associate|eq:residual-unit|<tuple|1|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnote-3|<tuple|3|?>>
    <associate|footnote-4|<tuple|4|?>>
    <associate|footnote-5|<tuple|5|?>>
    <associate|footnote-6|<tuple|6|?>>
    <associate|footnote-7|<tuple|7|?>>
    <associate|footnote-8|<tuple|8|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
    <associate|footnr-3|<tuple|3|?>>
    <associate|footnr-4|<tuple|4|?>>
    <associate|footnr-5|<tuple|5|?>>
    <associate|footnr-6|<tuple|6|?>>
    <associate|footnr-7|<tuple|7|?>>
    <associate|footnr-8|<tuple|8|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      ioffe_batch_2015

      he_deep_2016

      bartlett_representational_2017

      lin_why_2016

      saxe_exact_2013

      he_identity_2016

      sutskever_importance_2013
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Identity parametrizations
      improve optimization <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|1tab>|2<space|2spc>Identity parametrizations
      improve representation <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|3<space|2spc>Experiments
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>