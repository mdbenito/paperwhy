<TeXmacs|1.99.5>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Local minima in training of neural
  networks>|<doc-author|<author-data|<author-name|Swirszcz,
  Grzegorz>>>|<doc-author|<author-data|<author-name|Marian,
  Wojciech>>>|<doc-author|<author-data|<author-name|Pascanu,
  Razvan>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|deep-learning|optimization>

  <strong|tl;dr:> The goal is to construct elementary examples of datasets
  such that some neural network architectures get stuck in very bad local
  minima. The purpose is to better understand why NNs seem to work so well
  for many problems and what it is that makes them fail when they do. The
  authors conjecture that their examples can be generalized to higher
  dimensional problems and therefore that <strong|the good learning
  properties of deep networks rely heavily on the structure of the
  data>.<\footnote>
    <cite|lin_why_2016>
  </footnote>

  <hrule>

  <strong|Literature review:> Besides the examples themselves, the authors
  provide a valuable review, citing, among others:

  <\quotation>
    In particular Fyodorov & Williams (2007); Bray & Dean (2007), for random
    Gaussian error functions (...) all points with a low index [number of
    negative eigenvalues of the Hessian] (note that every minimum has this
    index equal to 0) have roughly the same performance, while critical
    points of high error implicitly have a large number of negative
    eigenvalue which means they are saddle points.
  </quotation>

  <\quotation>
    The claim of Dauphin et al. (2013)<\footnote>
      <cite|dauphin_identifying_2014>
    </footnote> is that the same structure holds for neural networks as well,
    when they become large enough.
  </quotation>

  <\quotation>
    Goodfellow et al. (2016)<\footnote>
      <cite|goodfellow_qualitatively_2014>
    </footnote> argues and provides some empirical evidence that while moving
    from the original initialization of the model along a straight line to
    the solution (found via gradient descent) the loss seems to be only
    monotonically decreasing, which speaks towards the apparent convexity of
    the problem. Soudry & Carmon (2016); Safran & Shamir (2015) also look at
    the error surface of the neural network, providing theoretical arguments
    for the error surface becoming well-behaved in the case of
    overparametrized models.
  </quotation>

  <\quotation>
    A different view, presented in Lin & Tegmark (2016)<\footnote>
      <cite|lin_why_2016>
    </footnote>; Shamir (2016), is that the underlying easiness of optimizing
    deep networks does not simply rest just in the emerging structures due to
    high-dimensional spaces, but is rather tightly connected to the intrinsic
    characteristics of the data these models are run on.
  </quotation>

  <strong|Contributions:> In Theorem 1 they construct what they conjecture to
  be the smallest dataset (10 points) such that a 2-2-1 fully connected NN
  with sigmoid activations is \Pdeadlocked\Q into a local minimum with an
  accuracy significantly below the optimum (50% that of another point they
  explictly show).

  In Section 3.2 they provide 3 examples for a single layer network with
  ReLUs for regression. Note that the region in input space where each unit
  saturates (the \Pblind spot\Q where the gradient vanishes) is the whole
  <math|\<bbb-R\><rsup|->>. However they are able to devise examples showing
  that:

  <\quotation>
    (...) blind spots are not the only reason a model can be stuck in a
    suboptimal solution. Even more surprisingly, (...) blind spots can be
    completely absent in the local optima, while at the same time being
    present in the global solution.
  </quotation>

  Basically the construction is based on the idea that

  <\quotation>
    (...) if, due to initial conditions, the model partitions the input space
    in a suboptimal way, it might become impossible to find the optimal
    partitioning using gradient descent.
  </quotation>

  Crucially, they conjecture that this idea can be non-trivially
  <strong|generalized to more interesting higher dimensional problems>.

  Finally, in Section 4 they construct a dataset for regression with a bad
  local minimum, based on the observation that, since the dataset is
  necessarily finite, it is possible to

  <\quotation>
    (...) compute conditions for the weights of any given layer of the model
    such that for any datapoint all the units of that layer are saturated
    [and learning stops]. Furthermore, we show that one can obtain a better
    solution than the one reached from such a state.
  </quotation>

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|3>
      <bibitem*|1><label|bib-dauphin_identifying_2014>Yann<nbsp>N Dauphin,
      Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya
      Ganguli<localize|, and >Yoshua Bengio. <newblock>Identifying and
      attacking the saddle point problem in high-dimensional non-convex
      optimization. <newblock><localize|In >Z.<nbsp>Ghahramani,
      M.<nbsp>Welling, C.<nbsp>Cortes, N.<nbsp>D.<nbsp>Lawrence<localize|,
      and >K.<nbsp>Q.<nbsp>Weinberger<localize|, editors>,
      <with|font-shape|italic|Advances in Neural Information Processing
      Systems 27>, <localize|pages >2933\U2941. Curran Associates, Inc.,
      2014. <newblock>Citecount: 00221 arXiv: 1405.4604.<newblock>

      <bibitem*|2><label|bib-goodfellow_qualitatively_2014>Ian<nbsp>J.<nbsp>Goodfellow,
      Oriol Vinyals<localize|, and >Andrew<nbsp>M.<nbsp>Saxe.
      <newblock>Qualitatively characterizing neural network optimization
      problems. <newblock><with|font-shape|italic|ArXiv:1412.6544 [cs,
      stat]>, <localize|page >11, dec 2014. <newblock>Citecount: 00032 arXiv:
      1412.6544.<newblock>

      <bibitem*|3><label|bib-lin_why_2016>Henry<nbsp>W.<nbsp>Lin<localize|
      and >Max Tegmark. <newblock>Why does deep and cheap learning work so
      well? <newblock><with|font-shape|italic|ArXiv:1608.08225 [cond-mat,
      stat]>, <localize|page >17, aug 2016. <newblock>Citecount: 00022 arXiv:
      1608.08225.<newblock>
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
    <associate|auto-1|<tuple|4|?>>
    <associate|bib-dauphin_identifying_2014|<tuple|1|?>>
    <associate|bib-goodfellow_qualitatively_2014|<tuple|2|?>>
    <associate|bib-lin_why_2016|<tuple|3|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnote-3|<tuple|3|?>>
    <associate|footnote-4|<tuple|4|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
    <associate|footnr-3|<tuple|3|?>>
    <associate|footnr-4|<tuple|4|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      lin_why_2016

      dauphin_identifying_2014

      goodfellow_qualitatively_2014

      lin_why_2016
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>