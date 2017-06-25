<TeXmacs|1.99.4>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|On the number of linear regions of<new-line>deep
  neural networks>|<doc-author|<author-data|<author-name|Montúfar,
  Guido>>>|<doc-author|<author-data|<author-name|Pascanu,
  Razvan>>>|<doc-author|<author-data|<author-name|Cho,
  Kyunghyun>>>|<doc-author|<author-data|<author-name|Bengio,
  Yoshua>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|complexity|neural-networks|representation>

  <strong|tl;dr:> Adding layers to build a deep model is exponentially better
  than just increasing the number of parameters in a shallow one in order to
  increase the complexity of the piecewise linear functions computed by
  feedforward neural networks with rectifier or maxout networks.

  <hrule>

  Consider a <dfn|feed forward neural network> with linear layers
  <math|f<rsub|l><around*|(|x|)>=W<rsup|l>*x+b<rsup|l>> followed by ReLUs
  <math|g<rsub|l><around*|(|z|)>=max<around*|{|0,z|}>>:

  <\equation*>
    F=f<rsub|out>\<circ\>g<rsub|L>\<circ\>f<rsub|L>\<circ\>\<ldots\>g<rsub|1>\<circ\>f<rsub|1>.
  </equation*>

  Each unit <math|i> of layer <math|l> is linear at each side of the
  hyperplane <math|H<rsub|i>=<around*|{|W<rsup|l><rsub|i:>x+b<rsup|l>=0|}>>,
  where <math|g> changes from 0 to the identity. The collection of al
  <math|H<rsub|i>> therefore splits the space of inputs to this layer into
  open, connected (and convex) sets. These are called <dfn|linear regions>
  for <math|g<rsub|l>\<circ\>f<rsub|l>>. More generally:

  <\quotation>
    <\definition>
      A linear region of a piecewise linear function
      <math|F:\<bbb-R\><rsup|n<rsub|0>>\<rightarrow\>\<bbb-R\><rsup|m>> is a
      maximal connected subset of the domain <math|\<bbb-R\><rsup|n<rsub|0>>>
      where <math|F> is linear.
    </definition>
  </quotation>

  The reason why these regions are important is that they measure how rich a
  piecewise linear function is, so the more of these (per number of layers or
  of parameters) a network can exhibit, the richer the set of functions it
  can approximate. Note that by adding more units to a single layer network,
  one can achieve any given number of linear regions; what matters is that
  adding layers while keeping fixed the number of parameters exponentially
  increases this number.

  <big-figure|<image|../static/img/montufar_number_2014.jpg|1par|||>|Decision
  boundaries of 1 and 2 layer models with the same number of hidden units>

  We will be discussing <strong|lower bounds> on the number of such linear
  regions for the full network <math|F> as a function of the number of layers
  <math|L> and of the number of parameters. In addition to rectifier
  activations, maxout is studied.<\footnote>
    Maxout activations take the maximum over several units. See
    <cite|goodfellow_maxout_2013>.
  </footnote>

  It was already known that deep networks with ReLUs split their input space
  into exponentially more linear regions than shallow networks, more
  specifically:

  <\theorem*>
    <dueto|Pascanu et al. 2013><\footnote>
      <cite|pascanu_number_2013>
    </footnote> A rectifier neural network with <math|n<rsub|0>> inputs and
    <math|L> hidden layers of width <math|n\<geqslant\>n<rsub|0>> can compute
    functions that have <math|\<Omega\><around*|(|<around*|(|n/n<rsub|0>|)><rsup|L-1>*n<rsup|n<rsub|0>>|)>>
    linear regions.
  </theorem*>

  The first contribution of the current paper is an improvement over this
  result with a bound which is also exponential in <math|n<rsub|0>>:

  <\quotation>
    <\theorem>
      A rectifier neural network with <math|n<rsub|0>> inputs and <math|L>
      hidden layers of width <math|n\<geqslant\>n<rsub|0>> can compute
      functions that have <math|\<Omega\><around*|(|<around*|(|n/n<rsub|0>|)><rsup|<around*|(|L-1|)>*n<rsub|0>>*n<rsup|n<rsub|0>>|)>>
      linear regions.
    </theorem>
  </quotation>

  This seems a small improvement at first glance, but it implies that even
  for <math|L> and <math|n> small, deep models are able to compute functions
  with a significantly greater amount of linear regions than shallow models
  can,<\footnote>
    Note that Pascanu et al. already mentions a similar fact. One wonders...
  </footnote> which is in tune with experimental evidence.

  The second contribution of the paper is the application to maxout networks,
  which again shows a growth of the number of linear regions which is
  exponential in <math|L>.

  <\quotation>
    <\theorem>
      A maxout network with <math|L> layers of constant width
      <math|n<rsub|0>> and rank <math|k> can compute functions with at least
      <math|k<rsup|L-1>*k<rsup|n<rsub|0>>> linear regions.
    </theorem>
  </quotation>

  By translating this theorem into a dependency on the number of parameters
  <math|K> it is possible to see that the growth in linear regions is
  exponential in <math|K> for deep models whereas it is only polynomial for
  shallow ones.

  To conclude the authors note:<\footnote>
    It seems to me that the condition <math|n<rsub|i>\<geqslant\>n<rsub|0>>
    would be violated in convnets. Why is this statement valid?
  </footnote>

  <\quotation>
    This framework is applicable to any neural network that has a piecewise
    linear activation function. For example, if we consider a convolutional
    network with rectifier units, as the one used in (Krizhevsky et al.
    2012), we can see that the convolution followed by max pooling at each
    layer identifies all patches of the input within a pooling region. This
    will let such a deep convolutional neural network recursively identify
    patches of the images of lower layers, resulting in exponentially many
    linear regions of the input space.
  </quotation>

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|2>
      <bibitem*|1><label|bib-goodfellow_maxout_2013>Ian<nbsp>J.<nbsp>Goodfellow,
      David Warde-Farley, Mehdi Mirza, Aaron Courville<localize|, and >Yoshua
      Bengio.<newblock> Maxout Networks.<newblock>
      <with|font-shape|italic|ArXiv:1302.4389 [cs, stat]>, <localize|page >9,
      feb 2013.<newblock> Citecount: 00756 code:
      http://www-etud.iro.umontreal.ca/\Cgoodfeli/maxout.html arxiv:
      1302.4389.<newblock>

      <bibitem*|2><label|bib-pascanu_number_2013>Razvan Pascanu, Guido
      Montufar<localize|, and >Yoshua Bengio.<newblock> On the number of
      response regions of deep feed forward networks with piece-wise linear
      activations.<newblock> <with|font-shape|italic|ArXiv:1312.6098 [cs]>,
      dec 2013.<newblock> Citecount: 00032 arXiv: 1312.6098.<newblock>
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|preamble|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|4|?>>
    <associate|bib-goodfellow_maxout_2013|<tuple|1|?>>
    <associate|bib-pascanu_number_2013|<tuple|2|?>>
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
      goodfellow_maxout_2013

      pascanu_number_2013

      pascanu_number_2013
    </associate>
    <\associate|figure>
      <tuple|normal|Decision boundaries of 1 and 2 layer models with the same
      number of hidden units|<pageref|auto-1>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>