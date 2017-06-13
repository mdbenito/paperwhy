<TeXmacs|1.99.4>

<style|<tuple|generic|paperwhy>>

<\body>
  <doc-data|<doc-title|Training with noise is equivalent to Tikhonov
  regularization>|<doc-author|<author-data|<author-name|Bishop, Chris
  M.>>>|<doc-running-author|Miguel de Benito Delgado>>

  <tags|regularization|input noise>

  <strong|tl;dr:> Adding noise to training inputs changes the risk function.
  A Taylor expansion shows that up to a term quadratic in the noise
  amplitude, the empirical risk is the same as without noise but with an
  additional term involving 1st derivatives of the estimator.

  <hrule>

  In our quest to understand all things regularization, today we review an
  old piece by Christopher Bishop no less!

  <subsection|The bias-variance tradeoff>

  We begin with a classical observation: for any statistical model we develop
  (i.e. for any choice of estimator <math|T=T<around*|(|X<rsub|1>,\<ldots\>,X<rsub|N>|)>>
  as a function of the data), we will always face the proverbial tradeoff
  between bias and variance of the statistic wrt. different datasets.
  Roughly: low bias <math|\<bbb-E\><rsub|\<b-up-X\>> T> implies high variance
  <math|V<rsub|\<b-up-X\>> T> and viceversa. One way to see it is fhe
  following: We desire low model complexity (fewer parameters for the
  description of <math|T>) but this typically results in high bias and low
  variance, so we can increase the complexity for a lower bias but higher
  variance. There are many ways to tackle this connundrum: a few examples are
  <strong|structural stabilization> to reduce the bias, <strong|ensembling>
  of poor (<em|weak>) estimators to reduce the variance, or
  <strong|regularization> of the objective function to achieve the same goal.
  The latter consists in adding a penalty term to the risk which
  <em|regularizes> the estimator in the sense that the problem of computing a
  good estimator from the data becomes well-posed, i.e. it depends smoothly
  on the training set. This leads to classical <hlink|Tikhonov
  regularization|en.wikipedia.org/tikhonov_regularization>.

  <\subsection>
    Adding noise to the input
  </subsection>

  The focus of the paper is in the related technique of <strong|adding noise
  to the training samples> and how (up to second order) it can be regarded as
  just adding a penalty term to the risk function. This noise could for
  instance be simply Gaussian, or <em|salt and pepper> noise (a binary mask),
  common in image recognition tasks.

  To fix ideas, suppose that we are training an estimator
  <math|x\<mapsto\><wide|y|^><around*|(|x|)>> via <dfn|empirical risk
  minimization>.<\footnote>
    \ That is: we want <math|<wide|y|^>=argmin<rsub|y>
    <wide|R|^><around*|(|y|)>>, where <math|<wide|R|^>> is an approximation
    to the <dfn|population risk> <math|R<around*|(|y|)>>, a magnitude
    encoding the expected failure of <math|y> in predicting well from
    <math|X>. There are many good introductions to statistical learning
    theory available: for a very brief one, see
    <cite-detail|bousquet_advanced_2004|p. 169>.
  </footnote> We have training data <math|<around*|(|x<rsub|i>,y<rsub|i>|)>>
  being realizations of the random variable <math|<around*|(|X,Y|)>> and we
  use quadratic loss and add some noise <math|\<xi\>> to the input <math|X>.
  This changes the risk function from <math|\<bbb-E\><rsub|X,Y><around*|[|<around*|\||<wide|y|^><around*|(|X|)>-Y|\|><rsup|2>|]>>
  to <math|\<bbb-E\><rsub|X,Y,\<xi\>><around*|[|<around*|\||<wide|y|^><around*|(|X+\<xi\>|)>-Y|\|><rsup|2>|]>>,
  so that the <dfn|empirical risk> to minize is

  <\equation>
    <label|eq:empirical-risk-noise><wide|R|^><rsub|\<xi\>><around*|(|<wide|y|^>|)>=<frac|1|N>*<big|sum><rsub|i=1><rsup|N><around*|\||<wide|y|^><around*|(|x<rsub|i>+\<xi\><rsub|i>|)>-y<rsub|i>|\|><rsup|2>.
  </equation>

  Assuming that the amplitude of the noise <math|<around*|\||\<xi\>|\|>> is
  small, a second order Taylor expansion of
  <math|<wide|y|^><around*|(|X+\<xi\>|)>> around <math|X> yields, after some
  computations on the population risk and bringing them back to the empirical
  one, a new <eqref|eq:empirical-risk-noise>:

  <\equation*>
    <wide|R|^><rsub|\<xi\>><around*|(|<wide|y|^>|)>=<wide|R|^><around*|(|<wide|y|^>|)>+\<eta\><rsup|2>*\<rho\><around*|(|<wide|y|^>|)>+<text|h.o.t.>
  </equation*>

  where the term <math|\<rho\><around*|(|<wide|y|^>|)>> is the
  <dfn|(empirical) regularizer> and <math|\<eta\><rsup|2>=Var<around*|(|\<xi\>|)>>.<\footnote>
    Note that we are omitting all population quantities here for brevity, but
    the regularizer is computed for the true expected error (population
    risk), then approximated using the sample data. The paper is actually a
    bit confusing in this respect since it tries to gather both population
    and sample quantities under one notation.
  </footnote> The expression that pops out for <math|\<rho\>> (which we don't
  reproduce here) has the big disadvantage of being <strong|not bounded from
  below> so that <math|<wide|R|^><rsub|\<xi\>>> is a rather poor choice for
  an objective function to minimise.

  <subsection|A quadratic approximate regularizer>

  However, one can rewrite the equations in term of the conditional
  expectations <math|\<bbb-E\><around*|[|Y\|X|]>> and
  <math|\<bbb-E\><around*|[|Y<rsup|2>\|X|]>> to obtain equivalent ones where
  it becomes apparent that, for small variances <math|\<eta\><rsup|2>>, the
  (empirical) regularizer can be approximated by

  <\equation*>
    <wide|\<rho\>|~><around*|(|<wide|y|^>|)>=<frac|1|2*N>*<big|sum><rsub|i=1><rsup|N><around*|\<\|\|\>|\<nabla\><wide|y|^><around*|(|x<rsub|i>|)>|\<\|\|\>><rsup|2>.
  </equation*>

  This is now much better: being quadratic and bounded below by 0 it is a
  \Pgood\Q term for the objective. The derivations in the paper show that it
  leads to the same minima (up to <math|\<cal-O\><around*|(|\<eta\><rsup|2>|)>>)
  as the \Ptrue\Q regularizer <math|\<rho\>>.

  The computations are next repeated for the <strong|cross-entropy error> to
  obtain a similar approximate regularizer, this time with an additional
  factor breaking its nice quadratic and Tikhonov-like form. There are
  efficient ways to compute the derivatives involved as part of
  backpropagation.

  The conclusion is then that, at least in these settings one can simply plug
  these regularizers in instead of adding noise to the input. This might not
  be that interesting computationally, but it provides a deeper understanding
  of what it is that we are doing when we perturb inputs, a technique very
  common nowadays e.g. in object classification with convnets.

  Finally, the paper concludes with a specific computation of the updates for
  weights in a neural network using the quadratic regularizer. The problem is
  that the Hessian of the error wrt. the weights is required, making the
  method unattractive for modern applications with millions of parameters.

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|1>
      <bibitem*|1><label|bib-bousquet_advanced_2004>Olivier Bousquet, Ulrike
      von<nbsp>Luxburg<localize|, and >Gunnar Rätsch<localize|,
      editors>.<newblock> <with|font-shape|italic|Advanced Lectures on
      Machine Learning>, <localize|volume> 3176<localize| of
      ><with|font-shape|italic|Lecture Notes in Computer Science>.<newblock>
      Springer Berlin Heidelberg, Berlin, Heidelberg, 2004.<newblock>
      Citecount: 00036.<newblock>
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
    <associate|auto-4|<tuple|3|?>>
    <associate|bib-bousquet_advanced_2004|<tuple|1|?>>
    <associate|eq:empirical-risk-noise|<tuple|1|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      bousquet_advanced_2004
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>The bias-variance tradeoff
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|1tab>|2<space|2spc>Adding noise to the input
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|3<space|2spc>A quadratic approximate
      regularizer <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>