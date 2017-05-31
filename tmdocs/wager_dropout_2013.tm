<TeXmacs|1.99.4>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;
  </hide-preamble>

  <doc-data|<doc-title|Dropout training as adaptive
  regularization>|<doc-author|<author-data|<author-name|Miguel de Benito
  Delgado>>>>

  <strong|tl;dr:> dropout (of features) for GLMs is a noising procedure
  equivalent to Tykhonov regularization. A first order approximation of the
  regularizer actually scales the parameters with the Fisher information
  matrix, adapting the objective function to the dataset, independently of
  the labels. This makes dropout useful in the context of semi-supervised
  learning: regularizers can be adapted to the unlabeled data yielding better
  generalization. For logistic regression the adaption amounts to favoring
  features on which the estimator is confident.

  <hrule>

  For shallow architectures, there were already some (stated) results on the
  averaging properties of dropout.<\footnote>
    See <cite|hinton_improving_2012>.
  </footnote> This was later extended to multiple layers with sigmoid units:
  simple weighting of outputs in the forward pass computes an approximation
  of the expectation of the ensemble.<\footnote>
    See <cite|baldi_understanding_2013|baldi_dropout_2014>.
  </footnote> Today's paper predates this work and focuses still on shallow
  networks, albeit within the wider scope of Generalized Linear Models.

  Since these are shallow models, dropout is performed on the inputs and it
  can be compared to other methods of input perturbation like additive
  Gaussian noise.

  There are <strong|3 main contributions> in this paper:

  <\enumerate>
    <item>A dropout regularizer for a GLM is (up to first order)
    <strong|equivalent to a classical Tykhonov regularizer> with
    <math|L<rsup|2>>-norm, with a specific scaling. Crucially, this scaling
    <em|depends on the data but not on the labels> and makes the
    regularization adaptive.

    <item>Incorporating this regularization into a rewriting of SGD as the
    repeated solution of regularized linear problems leads to an
    <strong|update similar to AdaGrad>.<\footnote>
      See <cite|duchi_adaptive_2011>.
    </footnote> A connection between the goals of both is established.

    <item>In the case of logistic regression the dropout regularizer is shown
    to <strong|favour confident predictions, regardless of the label> (in the
    sense that it penalizes less those weights corresponding to features on
    which the predicted probability is far from <math|1/2>). Therefore it
    makes sense to <strong|apply it to semi-supervised problems> computing an
    extra term over unlabeled data.
  </enumerate>

  <subsection|(Feature-) Dropout is weighted <math|L<rsup|2>>-regularization>

  Consider any <dfn|Generalized Linear Model> with parameters
  <math|\<beta\>>, inputs <math|x\<in\>\<bbb-R\><rsup|d>> and outputs
  <math|y\<in\>Y>, i.e.<\footnote>
    Recall that in a GLM one uses a so-called <strong|link function> <math|h>
    to relate a linear predictor <math|x*\<beta\>> with the posterior
    <math|p<around*|(|y\|x|)>> by means of the relationship
    <math|\<bbb-E\><around*|[|y\|x|]>=h<rsup|-1><around*|(|x*\<beta\>|)>>. In
    our notation, <math|h=A<rprime|'>>. To fix ideas think of logistic
    regression, where <math|p<around*|(|y\|x|)>=<around*|(|1+\<mathe\><rsup|-x*\<beta\>>|)><rsup|-1>>.
    In this case we assume <math|y\<in\><around*|{|0,1|}>>, the log
    likelihood is <math|p<around*|(|\<b-y\>\|\<b-x\>|)>=<big|prod><rsub|i>p<rsub|i><rsup|y<rsub|i>>*<around*|(|1-p<rsub|i>|)><rsup|1-y<rsub|i>>>,
    with <math|p<rsub|i>\<assign\><around*|(|1+\<mathe\><rsup|-x<rsub|i>*\<beta\>>|)><rsup|-1>>
    and the negative log likelihood is the <strong|cross entropy loss>:
    <math|log p<around*|(|y\|x|)>=-<big|sum><rsub|i>y<rsub|i>*log
    p<rsub|i>+<around*|(|1-y<rsub|i>|)>*log<around*|(|1-p<rsub|i>|)>>.
  </footnote>

  <\equation*>
    p<around*|(|y\|x,\<beta\>|)>=h<around*|(|y|)>*\<mathe\><rsup|y*x*\<beta\>-A<around*|(|x*\<beta\>|)>>,
  </equation*>

  and <strong|negative log likelihood> as the loss. For one training sample
  <math|<around*|(|x,y|)>>: <math|l<around*|(|\<beta\>;x,y|)>=-log
  p<around*|(|y\|x,\<beta\>|)>>. Now choose some noise vector
  <math|\<xi\>\<in\>\<bbb-R\><rsup|d>> with i.i.d. entries with zero mean and
  replace <math|x\<mapsto\><wide|x|~>> where <math|<wide|x|~>> has been
  noised with <math|\<xi\>> in some way we specify later. A couple of
  computations show that the empirical loss on the (full) noised data
  <math|<wide|\<b-x\>|~>=<around*|(|<wide|x|~><rsub|1>,\<ldots\>,<wide|x|~><rsub|n>|)>,\<b-y\>=<around*|(|y<rsub|1>,\<ldots\>,y<rsub|n>|)>>
  is the loss on the original data plus a new term:

  <\equation*>
    <wide|L|^><around*|(|<wide|\<b-x\>|~>,\<b-y\>,\<beta\>|)>=<wide|L|^><around*|(|\<b-x\>,\<b-y\>,\<beta\>|)>+R<around*|(|\<beta\>|)>,
  </equation*>

  which is the <dfn|noising regularizer>

  <\equation>
    <label|eq:noising-regularizer>R<around*|(|\<beta\>|)>\<assign\><big|sum><rsub|i=1><rsup|n>\<bbb-E\><rsub|\<xi\>><around*|[|A<around*|(|<wide|x|~><rsub|i>*\<beta\>|)>|]>-A<around*|(|x<rsub|i>*\<beta\>|)>.
  </equation>

  <math|R<around*|(|\<beta\>|)>> has two key features:

  <\itemize-dot>
    <item><strong|It does not depend on the labels>: this will allow for its
    use in unsupervised setting.

    <item><strong|It is adapted to the training data>.
  </itemize-dot>

  But how exactly \Padapted\Q? Definition <eqref|eq:noising-regularizer> is
  quite impenetrable as is, even after plugging in a specific <math|A>.
  Assuming we can do a Taylor expansion of <math|A> around <math|x*\<beta\>>
  one obtains<\footnote>
    Indeed <math|A<around*|(|<wide|x|~>*\<beta\>|)>-A<around*|(|x*\<beta\>|)>=A<rprime|'><around*|(|x*\<beta\>|)>*<around*|(|<wide|x|~>*\<beta\>-x*\<beta\>|)>+<frac|1|2>*A<rprime|''><around*|(|x*\<beta\>|)>*<around*|(|<wide|x|~>*\<beta\>-x*\<beta\>|)><rsup|2>+<text|h.o.t.>>
    and taking expectations: <math|\<bbb-E\><rsub|\<xi\>><around*|[|A<around*|(|<wide|x|~>*\<beta\>|)>|]>-A<around*|(|x*\<beta\>|)>=A<rprime|'><around*|(|x*\<beta\>|)>*<around*|(|\<bbb-E\><around*|[|<wide|x|~>*\<beta\>|]>-x*\<beta\>|)>+<frac|1|2>*A<rprime|''><around*|(|x*\<beta\>|)>*\<bbb-E\><around*|(|<wide|x|~>*\<beta\>-x*\<beta\>|)><rsup|2>+<text|h.o.t>>.
  </footnote>

  <\equation*>
    \<bbb-E\><rsub|\<xi\>><around*|[|A<around*|(|<wide|x|~>*\<beta\>|)>|]>-A<around*|(|x*\<beta\>|)>\<approx\><tfrac|1|2>*A<rprime|''><around*|(|x*\<beta\>|)>*Var<rsub|\<xi\>>
    <around*|[|<wide|x|~>*\<beta\>|]>
  </equation*>

  and substituting into <eqref|eq:noising-regularizer> the (approximate)
  <dfn|quadratic noising regularizer>:

  <\equation>
    <label|eq:quadratic-noising-regularizer>R<rsup|q><around*|(|\<beta\>|)>\<assign\><big|sum><rsub|i=1><rsup|n><tfrac|1|2>*A<rprime|''><around*|(|x*\<beta\>|)>*Var<rsub|\<xi\>>
    <around*|[|<wide|x|~>*\<beta\>|]>.
  </equation>

  Note that <eqref|eq:quadratic-noising-regularizer> is in general
  non-convex. When questioned by a reviewer about this fact, the authors
  respond

  <\quotation>
    Although our objective is not formally convex, we have not encountered
    any major difficulties in fitting it for datasets where n is reasonably
    large (say on the order of hundreds). When working with LBFGS, multiple
    restarts with random parameter values give almost identical results. The
    fact that we have never really had to struggle with local minimas
    suggests that there is something interesting going on here in terms of
    convexity.
  </quotation>

  We can now fix the noising method and look at its variance to gain insight
  into what <math|R<rsup|q>> does, and hopefully, by extension
  <math|R>.<\footnote>
    There is a handwavy discussion in the paper on the error
    <math|<around*|\||R-R<rsup|q>|\|>> which is not worth discussing here.
    Suffice to say: it works \Pwell\Q in practice.
  </footnote> The authors consider:

  <\itemize-dot>
    <item><strong|Additive gaussian noise>: <math|<wide|x|~>=x+\<xi\>> with
    <math|\<xi\><rsub|i>> i.i.d. spherical Gaussians
    <math|\<cal-N\><around*|(|0,\<sigma\><rsup|2>*Id<rsub|d>|)>>.

    <item><strong|Dropout noise>: fix <math|\<delta\>\<in\><around*|(|0,1|)>>
    to build a (scaled) binary mask <math|\<xi\>> with i.i.d entries
    <math|Bernoulli<around*|(|1-\<delta\>|)>> and set
    <math|<wide|x|~>=x\<odot\>\<xi\>/<around*|(|1-\<delta\>|)>> to cancel
    some of the inputs with probability <math|\<delta\>>.<\footnote>
      Here <math|\<odot\>> stands for the entrywise or <hlink|Hadamard
      product|https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>.
    </footnote>
  </itemize-dot>

  Notice that in both cases <math|\<bbb-E\><rsub|\<xi\>><around*|[|<wide|x|~>|]>=x>
  and the expectation of the Taylor expansion of <math|A> yields
  <eqref|eq:quadratic-noising-regularizer> (that's the reason for the scaling
  factor <math|\<delta\>>). After performing the necessary computations, and
  assuming the design matrix has been normalized to
  <math|\<Sigma\><rsub|i\<nocomma\>j>x<rsup|2><rsub|i\<nocomma\>j>=1>, the
  authors obtain the following neat table:

  <big-table|<tabular|<tformat|<cwith|1|-1|1|1|cell-halign|r>|<cwith|1|1|2|-1|cell-halign|c>|<cwith|1|-1|2|2|cell-halign|c>|<cwith|1|-1|3|3|cell-halign|c>|<cwith|1|-1|4|4|cell-halign|c>|<cwith|1|1|2|-1|cell-background|pastel
  grey>|<cwith|2|-1|1|1|cell-background|pastel
  grey>|<cwith|3|3|3|4|cell-tborder|1ln>|<cwith|2|2|3|4|cell-bborder|1ln>|<cwith|4|4|3|4|cell-bborder|1ln>|<cwith|3|4|3|3|cell-lborder|1ln>|<cwith|3|4|2|2|cell-rborder|1ln>|<cwith|3|4|4|4|cell-rborder|1ln>|<table|<row|<cell|>|<cell|Linear
  regression>|<cell|Logistic regression>|<cell|GLM>>|<row|<cell|<math|L<rsup|2>>-penalty>|<cell|<math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsup|2><rsub|2>>>|<cell|<math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsup|2><rsub|2>>>|<cell|<math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsup|2><rsub|2>>>>|<row|<cell|Additive
  noise>|<cell|<math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsup|2><rsub|2>>>|<cell|<math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsup|2><rsub|2>*<big|sum><rsub|i>*p<rsub|i>*<around*|(|1-p<rsub|i>|)>>>|<cell|<math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsup|2><rsub|2>*tr
  <around*|(|V<around*|(|\<beta\>|)>|)>>>>|<row|<cell|Dropout
  noise>|<cell|<math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsup|2><rsub|2>>>|<cell|<math|<big|sum><rsub|i,j>*p<rsub|i>*<around*|(|1-p<rsub|i>|)>*x<rsub|i\<nocomma\>j><rsup|2>*\<beta\><rsub|j><rsup|2>>>|<cell|<math|\<beta\><rsup|\<top\>>*diag<around*|(|X<rsup|\<top\>>*V<around*|(|\<beta\>|)>*X|)>*\<beta\>>>>>>>|<label|tab:summary-regs><math|R<rsup|q>>
  (up to constants) for different models and noising methods. See below for
  the definition of <math|V<around*|(|\<beta\>|)>>.>

  The first row holds by definition. The first column recovers known
  results<\footnote>
    See <cite|bishop_training_1995> for more on additive noise leading to
    ridge regression.
  </footnote> and adds the fact that dropout (after scaling) on linear
  regression is ridge regression. It's the box who tells a more interesting
  story. First we note that the key matrix
  <math|V<around*|(|\<beta\>|)>\<in\>\<bbb-R\><rsup|n\<times\>n>> is diagonal
  with entries <math|V<around*|(|\<beta\>|)><rsub|i\<nocomma\>i>=A<rprime|''><around*|(|x<rsub|i>*\<beta\>|)>>.

  Additive noising for logistic regression penalizes more strongly uncertain
  predictions (<math|p<rsub|i>\<approx\>0.5>). For arbitrary GLMs,
  <math|R<rsup|q>> is just multiplied by a constant.

  Dropout in logistic regression has the same feature as additive noise
  <em|plus selective exclusion of features:> given a training sample
  <math|x<rsub|i>>, <math|\<beta\><rsub|j>> is not penalized if
  <math|x<rsub|i\<nocomma\>j>=0>. In particular
  <math|p<rsub|i>*<around*|(|1-p<rsub|i>|)>> and <math|\<beta\><rsub|j>> may
  both be large if the <em|cross-term> <math|x<rsub|i\<nocomma\>j><rsup|2>>
  is small. This means that

  <\quotation>
    (...) dropout regularization should be better than
    <math|L<rsup|2>>-regularization for learning weights for features that
    are rare (i.e., often 0) but highly discriminative, because dropout
    effectively does not penalize \ <math|j> over observations for which
    <math|x<rsub|i\<nocomma\>j> = 0>.
  </quotation>

  And

  <\quotation>
    dropout rewards those features that are rare and positively co-adapted
    with other features in a way that enables the model to make confident
    predictions whenever the feature of interest is active.
  </quotation>

  In the more general case the insight comes from the fact that

  <\equation*>
    <tfrac|1|n>*X<rsup|\<top\>>*V<around*|(|\<beta\><rsup|\<star\>>|)>*X=<tfrac|1|n><big|sum><rsub|i=1><rsup|n>\<nabla\><rsup|2>l<around*|(|\<beta\><rsup|\<star\>>;x<rsub|i>,y<rsub|i>|)>
  </equation*>

  is an estimator of the Fisher information matrix <math|\<cal-I\>>.
  Therefore if we write <math|\<beta\><rsup|\<top\>>*diag<around*|(|X<rsup|\<top\>>*V<around*|(|\<beta\>|)>*X|)>*\<beta\>=\<beta\><rsup|\<top\>>*D*\<beta\>=<around*|\<\|\|\>|D<rsup|1/2>*\<beta\>|\<\|\|\>><rsub|2><rsup|2>=<around*|\<\|\|\>|<wide|\<beta\>|~>|\<\|\|\>><rsub|2><rsup|2>>
  we see that dropout is applying an <math|L<rsup|2>> penalty after
  normalizing with an approximation of <math|diag<around*|(|\<cal-I\>|)><rsup|-1/2>>.

  <\quotation>
    The Fisher information is linked to the shape of the level surfaces of
    <math|l(\<beta\>)> around <math|\<beta\><rsup|\<star\>>>. If
    <math|\<cal-I\>> were a multiple of the identity matrix, then these level
    surfaces would be perfectly spherical around
    <math|\<beta\><rsup|\<star\>>>.
  </quotation>

  By normalizing, the feature space is deformed into a shape where \P<em|the
  features have been balanced out>\Q.<\footnote>
    Notice that we could use any quadratic form to redefine the norm in which
    weights are measured. There are surely many other interesting
    possibilities!
  </footnote> The authors provide a very nice picture for intuition:

  <render-big-figure||Figure A.2|<image|../static/img/wager_dropout_2013-figA2.jpg|1par|||>|(page
  11 in the Appendix) <em|Comparison of two <math|L<rsup|2>> regularizers. In
  both cases, the black solid ellipses are level surfaces of the likelihood
  and the blue dashed curves are level surfaces of the regularizer; the
  optimum of the regularized objective is denoted by OPT. The left panel
  shows a classic spherical <math|L<rsup|2>> regular izer
  <math|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsub|2><rsup|2>>, whereas the
  right panel has an <math|L<rsup|2>> regularizer
  <math|\<beta\><rsup|\<top\>>*diag(\<cal-I\>)*\<beta\>> that has been
  adapted to the shape of the likelihood (<math|\<cal-I\>> is the Fisher
  information matrix). The second regularizer is still aligned with the axes,
  but the relative importance of each axis is now scaled using the curvature
  of the likelihood function. As argued [above], dropout training is
  comparable to the setup depicted in the right panel.>>

  <subsection|Relation to AdaGrad>

  By rewriting standard SGD into an iterative solution of linear
  <math|L<rsup|2>>-penalized problems

  <\equation*>
    <wide|\<beta\>|^><rsub|t+1>=argmin<rsub|\<beta\>>
    <around*|{|l<around*|(|<wide|\<beta\>|^><rsub|t>;x<rsub|t>,y<rsub|t>|)>+\<nabla\>l<around*|(|<wide|\<beta\>|^><rsub|t>|)>*<around*|(|\<beta\>-<wide|\<beta\>|^><rsub|t>|)>+<frac|1|2*\<eta\><rsub|t>>*<around*|\<\|\|\>|\<beta\>-<wide|\<beta\>|^><rsub|t>|\<\|\|\>><rsub|2><rsup|2>|}>
  </equation*>

  and substituting the dropout penalty for the penalty in this formulation,
  one obtains the update rule

  <\equation*>
    <wide|\<beta\>|^><rsub|t+1>=argmin<rsub|\<beta\>>
    <around*|{|l<around*|(|<wide|\<beta\>|^><rsub|t>;x<rsub|t>,y<rsub|t>|)>+g<rsub|t>*<around*|(|\<beta\>-<wide|\<beta\>|^><rsub|t>|)>+R<rsup|q><around*|(|\<beta\>-<wide|\<beta\>|^><rsub|t>;<wide|\<beta\>|^><rsub|t>|)>|}>
  </equation*>

  with the <dfn|centered quadratic dropout penalty>, similarly to the entry
  in Table <reference|tab:summary-regs>:

  <\equation*>
    R<rsup|q><around*|(|\<beta\>-<wide|\<beta\>|^><rsub|t>;<wide|\<beta\>|^><rsub|t>|)>=<around*|(|\<beta\>-<wide|\<beta\>|^><rsub|t>|)><rsup|\<top\>>*diag<around*|(|X<rsup|\<top\>>*V<around*|(|<wide|\<beta\>|^><rsub|t>|)>*X|)>*<around*|(|\<beta\>-<wide|\<beta\>|^><rsub|t>|)>.
  </equation*>

  This is effectively solving the problem of SGD has learning weights for
  \Prare but highly discriminative features\Q, by using the update

  <\equation*>
    <wide|\<beta\>|^><rsub|t+1>=<wide|\<beta\>|^><rsub|t>-\<eta\><rsub|t>*A<rsub|t><rsup|-1>\<nabla\>l<around*|(|<wide|\<beta\>|^><rsub|t>|)>.
  </equation*>

  AdaGrad<\footnote>
    See <cite|duchi_adaptive_2011>.
  </footnote> uses <math|A<rsub|t>=diag<around*|(|\<nabla\><rsup|\<top\>>l<around*|(|<wide|\<beta\>|^><rsub|t>|)>*\<nabla\>l<around*|(|<wide|\<beta\>|^><rsub|t>|)>|)><rsup|-1/2>>,
  warping the gradient by some sort of intrinsic metric, whereas dropout uses
  its estimate of the Fisher information.<\footnote>
    This looks like a nice connection to second order methods: warp the
    update step with information on the target function or warp feature space
    with information on the data to \Pimprove\Q it. <em|Very> handwavily...
  </footnote> However, in the limit <math|<wide|\<beta\>|^><rsub|t>\<rightarrow\>\<beta\><rsup|\<star\>>>
  for GLMs the expectations of both matrices are equal to <math|\<cal-I\>>,
  meaning that the SGD updates when using feature dropout in GLMs are
  \Pconverging\Q in some sense to AdaGrad updates.

  <subsection|Semi-supervised tasks>

  As we said above, the dropout regularizer is shown to change the loss
  function with the Fisher information matrix in a way that focuses on
  weights relevant for discriminative features, <em|without recourse to the
  labels <math|y<rsub|i>>>. Therefore in a semi-supervised context, we can
  use unlabeled data to improve the regularizer:

  <\equation*>
    R<rsub|\<ast\>><around*|(|\<beta\>|)>\<assign\><frac|n|n+\<alpha\>*m>*<around*|(|R<around*|(|\<beta\>|)>+\<alpha\>*R<rsub|<text|unlabeled>><around*|(|\<beta\>|)>|)>,
  </equation*>

  where <math|n> is the size of the labeled dataset, <math|m> that of the
  unlabeled one and <math|\<alpha\>> a \Pdiscount factor\Q for the latter
  which is a hyperparameter. Unlike other semi-supervised approaches relying
  on generative models, the authors' approach

  <\quotation>
    is based on a different intuition: we'd like to set weights to make
    confident predictions on unlabeled data as well as the labeled data, an
    intuition shared by entropy regularization [24] and transductive SVMs
    [25].
  </quotation>

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|5>
      <bibitem*|1><label|bib-baldi_dropout_2014>Pierre Baldi<localize| and
      >Peter Sadowski.<newblock> The Dropout Learning Algorithm.<newblock>
      <with|font-shape|italic|Artificial intelligence>, 210:78\U122, may
      2014.<newblock> Citecount: 00068.<newblock>

      <bibitem*|2><label|bib-baldi_understanding_2013>Pierre Baldi<localize|
      and >Peter<nbsp>J Sadowski.<newblock> Understanding Dropout.<newblock>
      <localize|In >C.<nbsp>J.<nbsp>C.<nbsp>Burges, L.<nbsp>Bottou,
      M.<nbsp>Welling, Z.<nbsp>Ghahramani<localize|, and
      >K.<nbsp>Q.<nbsp>Weinberger<localize|, editors>,
      <with|font-shape|italic|Advances in Neural Information Processing
      Systems 26>, <localize|pages >2814\U2822. 2013.<newblock> Citecount:
      00066.<newblock>

      <bibitem*|3><label|bib-bishop_training_1995>Chris<nbsp>M
      Bishop.<newblock> Training with noise is equivalent to Tikhonov
      regularization.<newblock> <with|font-shape|italic|Neural computation>,
      7(1):108\U116, 1995.<newblock> Citecount: 00545.<newblock>

      <bibitem*|4><label|bib-duchi_adaptive_2011>John Duchi, Elad
      Hazan<localize|, and >Yoram Singer.<newblock> Adaptive Subgradient
      Methods for Online Learning and Stochastic Optimization.<newblock>
      <with|font-shape|italic|Journal of Machine Learning Research>,
      12(Jul):2121\U2159, 2011.<newblock> Citecount: 01743.<newblock>

      <bibitem*|5><label|bib-hinton_improving_2012>Geoffrey<nbsp>E.<nbsp>Hinton,
      Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever<localize|, and
      >Ruslan<nbsp>R.<nbsp>Salakhutdinov.<newblock> Improving neural networks
      by preventing co-adaptation of feature detectors.<newblock>
      <with|font-shape|italic|ArXiv:1207.0580 [cs]>, <localize|page >18, jul
      2012.<newblock> Citecount: 01870 arXiv: 1207.0580.<newblock>
    </bib-list>
  </bibliography>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|9|?>>
    <associate|auto-4|<tuple|2|?>>
    <associate|auto-5|<tuple|3|?>>
    <associate|auto-6|<tuple|3|?>>
    <associate|bib-baldi_dropout_2014|<tuple|1|?>>
    <associate|bib-baldi_understanding_2013|<tuple|2|?>>
    <associate|bib-bishop_training_1995|<tuple|3|?>>
    <associate|bib-duchi_adaptive_2011|<tuple|4|?>>
    <associate|bib-hinton_improving_2012|<tuple|5|?>>
    <associate|eq:noising-regularizer|<tuple|1|?>>
    <associate|eq:quadratic-noising-regularizer|<tuple|2|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-10|<tuple|10|?>>
    <associate|footnote-11|<tuple|11|?>>
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
    <associate|footnr-2|<tuple|2|?>>
    <associate|footnr-3|<tuple|3|?>>
    <associate|footnr-4|<tuple|4|?>>
    <associate|footnr-5|<tuple|5|?>>
    <associate|footnr-6|<tuple|6|?>>
    <associate|footnr-7|<tuple|7|?>>
    <associate|footnr-8|<tuple|8|?>>
    <associate|footnr-9|<tuple|9|?>>
    <associate|tab:summary-regs|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|>
      <tuple|normal|(page 11 in the Appendix)
      <with|font-shape|<quote|italic>|Comparison of two
      <with|mode|<quote|math>|L<rsup|2>> regularizers. In both cases, the
      black solid ellipses are level surfaces of the likelihood and the blue
      dashed curves are level surfaces of the regularizer; the optimum of the
      regularized objective is denoted by OPT. The left panel shows a classic
      spherical <with|mode|<quote|math>|L<rsup|2>> regular izer
      <with|mode|<quote|math>|<around*|\<\|\|\>|\<beta\>|\<\|\|\>><rsub|2><rsup|2>>,
      whereas the right panel has an <with|mode|<quote|math>|L<rsup|2>>
      regularizer <with|mode|<quote|math>|\<beta\><rsup|\<top\>>*diag(\<cal-I\>)*\<beta\>>
      that has been adapted to the shape of the likelihood
      (<with|mode|<quote|math>|\<cal-I\>> is the Fisher information matrix).
      The second regularizer is still aligned with the axes, but the relative
      importance of each axis is now scaled using the curvature of the
      likelihood function. As argued [above], dropout training is comparable
      to the setup depicted in the right panel.>|<pageref|auto-3>>
    </associate>
    <\associate|bib>
      hinton_improving_2012

      baldi_understanding_2013

      baldi_dropout_2014

      duchi_adaptive_2011

      bishop_training_1995

      duchi_adaptive_2011
    </associate>
    <\associate|table>
      <tuple|normal|<with|mode|<quote|math>|R<rsup|q>> (up to constants) for
      different models and noising methods. See below for the definition of
      <with|mode|<quote|math>|V<around*|(|\<beta\>|)>>.|<pageref|auto-2>>
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>(Feature-) Dropout is weighted
      <with|mode|<quote|math>|L<rsup|2>>-regularization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|1tab>|2<space|2spc>Relation to AdaGrad
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|3<space|2spc>Semi-supervised tasks
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>