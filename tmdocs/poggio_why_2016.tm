<TeXmacs|1.99.4>

<style|<tuple|generic|british|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|dfn|<macro|body|<strong|<with|font-shape|italic|<arg|body>>>>>
  </hide-preamble>

  <doc-data|<doc-title|Why and when can deep \U but not shallow \U networks
  avoid the curse of dimensionality: a review>>

  <strong|tl;dr:><\footnote>
    This paper is packed with results, comments and conjectures! I had to
    omit many details to keep this post at a reasonable length.
  </footnote> deep convnets avoid the curse of dimensionality for the
  approximation of certain classes of functions (hierarchical compositions):
  complexity bounds (for the number of units) are polynomial instead of
  exponential in the dimension of the input as is the case for shallow
  networks. This is true for smooth and non-smooth activations like ReLUs.
  For the latter insight into how they approximate (hierarchical) Lipschitz
  functions is provided . It is conjectured that many target functions
  relevant to current machine learning problems are in these classes due
  either to physical grounds (<cite|lin_why_2016>) or biological ones.

  Guarantees of the approximation properties of NNs are typically proven for
  general classes of functions, say <math|C<rsup|m><around*|(|X|)>,X\<subseteq\>\<bbb-R\><rsup|n>>
  or assuming weaker regularity, like some Sobolev space
  <math|W<rsup|m,p><around*|(|X|)>>. Without additional restrictions, not
  only can the approximation performance not be guaranteed to be good, but it
  can be <em|guaranteed to be bad>.<\footnote>
    For approximation to continuous functions by shallow networks,
    <cite-detail|pinkus_approximation_1999|Ÿ6>, cites results by Maiorov on
    the upper and lower bounds on the approximation quality which are
    exponential on the number of inputs and the \Pbad functions\Q guilty of
    the lower bound actually form a set of \Plarge measure\Q so this is not
    just a worst-case scenario. As a matter of fact:

    <\quotation>
      examples of specific functions that cannot be represented efficiently
      by shallow networks have been given very recently by Telgarsky [25] and
      by Shamir [26]. [The authors] provide in theorem 5 another example
      (...) for which there is a gap between shallow and deep networks.
    </quotation>
  </footnote>

  An old and powerful insight into why convnets work so well is the fact that
  they are <dfn|hierarchical compositions of local functions>. They look
  like:

  <\equation*>
    f<around*|(|x<rsub|1>,\<ldots\>,x<rsub|8>|)>=h<rsub|3><around*|(|h<rsub|21><around*|(|h<rsub|11><around*|(|x<rsub|1>,x<rsub|3>|)>,h<rsub|12><around*|(|x<rsub|2>,x<rsub|4>|)>|)>,h<rsub|22><around*|(|h<rsub|13><around*|(|x<rsub|5>,x<rsub|6>|)>,h<rsub|14><around*|(|x<rsub|7>,x<rsub|8>*|)>|)>|)>.
  </equation*>

  The hierarchical structure is obvious; we can represent the compositions as
  a <dfn|<math|d>-tree>: every non-leaf node has <math|d> inputs and one
  output, with <math|d=2> (i.e. a binary tree) and <math|8> leafs at the
  bottom. We say that <dfn|<math|f> has <math|d>-tree structure>. The term
  <dfn|locality> refers to the lower dimension <math|d\<ll\>n> of the domain
  of each constituent function <math|h<rsub|i\<nocomma\>j>>. Since convnets
  are exactly of this type, it seems sensible to restrict oneself to general
  classes of functions which keep this structure to investigate why deep
  convnets excel where shallow nets fail. The idea is to prove:

  <\enumerate-alpha>
    <item>Deep convnets approximate hierarchical, compositional functions
    arbitrarily well, with polynomial cost in complexity.

    <item>Shallow nets might incur a huge complexity penalty in approximating
    these functions (upper bounds) and they actually do in many cases (lower
    bounds).\ 
  </enumerate-alpha>

  <big-figure|<image|../static/img/poggio_why_2016-fig1.jpg||500px||>|A
  function with binary tree structure and an ideal network approximating it.>

  Of course the obvious question is why this kind of functions are relevant
  in practice! Here are two hypotheses to this respect:

  <\itemize-dot>
    <item>Lin and Tegmark<\footnote>
      See <cite|lin_why_2016>.
    </footnote> recently proposed that physical processes relevant to ML
    tasks are described by simple, polynomial, Hamiltonians (at different
    scales). These are very easily approximated by hierarchical neural
    networks, since multiplication is.

    <item>The authors of the review propose a sort of <hlink|(weak) anthropic
    principle|https://en.wikipedia.org/wiki/Anthropic_principle>: ML focuses
    on many problems which are well solved by the brain, and the brain is
    wired in a deep, hierarchical way because it was evolutionarily
    advantageous. Therefore it is reasonable that hierarchical deep networks
    perform well at the same tasks.<\footnote>
      Did that make any sense?
    </footnote>
  </itemize-dot>

  But let us get to (some of) the results. We can formalize the previous
  notions as follows. Note that we must change their notation a bit because
  it collides with otherwise common <strike-through|sense> notation.

  <subsection*|Some definitions>

  <\definition>
    Let <math|V<rsub|N>> be the set of all networks with total number of
    units (<dfn|complexity>) <math|N> and <math|f\<in\>W>, for some function
    set <math|W>. The <dfn|degree of approximation of <math|V<rsub|N>> to
    <math|f>> is

    <\equation*>
      dist<around*|(|f,V<rsub|N>|)>=<below|inf|P\<in\>V<rsub|N>>
      <around*|\<\|\|\>|f-P|\<\|\|\>><rsub|\<infty\>>.
    </equation*>
  </definition>

  Then, if <math|dist(f,V<rsub|N>) = \<cal-O\>(N<rsup|-\<gamma\>>)> for some
  <math|\<gamma\>\<gtr\>0>, then for any <math|\<varepsilon\>\<gtr\>0> there
  exists a network with complexity <math|N=\<cal-O\><around*|(|\<varepsilon\><rsup|<frac*|-1|\<gamma\>>>|)>>
  which approximates <math|f> with accuracy at least <math|\<varepsilon\>>.

  The restricted function spaces to consider will be characterised by their
  smoothness and \Pdegree of compositionality\Q <math|d>. The paper first
  handles the case <math|d=2>, then goes on to arbitrary but fixed <math|d>
  (Theorem <reference|thm:main-result>), then to variable <math|d> across
  units (Theorem <reference|thm:dag-functions>).

  <\definition>
    Let <math|S<rsub|N><rsup|n>\<subset\>V<rsub|N>> be the class of all
    shallow networks with <math|n> inputs and <math|N> hidden units of the
    form

    <\equation*>
      x\<mapsto\><big|sum><rsub|k=1><rsup|N>a<rsub|k>*\<sigma\><around*|(|<around*|\<langle\>|w<rsub|k>,x|\<rangle\>>+b<rsub|k>|)>,<space|1em>w<rsub|k>\<in\>\<bbb-R\><rsup|n>,b<rsub|k>\<in\>\<bbb-R\>,
    </equation*>

    with <math|\<sigma\>> a smooth non-polynomial non-linearity.

    Let <math|D<rsub|N,d><rsup|n>\<subset\>V<rsub|N>> be the class of all
    deep networks (i.e. having more than one hidden layer) with a
    <math|d>-tree architecture whose nodes are all in
    <math|S<rsub|M><rsup|d>>, where <math|M=N/<around*|\||V|\|>> and <math|V>
    is the set of non-leaf vertices of the tree.
  </definition>

  For a convnet, <math|d> is the size of the kernel, which for now we
  consider fixed. Each hidden unit of a network in <math|D<rsub|N,d><rsup|n>>
  has <math|d> inputs, and <math|M> sets of coefficients
  <math|a<rsub|k>,w<rsub|k>,b<rsub|k>>.

  Note that the smoothness assumption on <math|\<sigma\>> can be easily
  overcome for most non-linear activations since they can be approximated in
  the <math|<around*|\<\|\|\>|\<cdot\>|\<\|\|\>><rsub|\<infty\>>> norm by
  smooth ones, e.g. the ReLU can be trivially approximated by piecing
  together two linear functions and a polynomial, or more generally with
  splines. See more on this below.

  <subsection*|Smooth activation functions>

  Equipped with these definitions we come to the main result (which is not in
  this exact form in the paper)

  <\theorem>
    <label|thm:main-result>Let <math|f\<in\>C<rsup|m><around*|(|X<rsub|n>|)>>
    with <math|n=dim X<rsub|n>>. Then a shallow network in some
    <math|S<rsub|N><rsup|n>> approximating <math|f> with accuracy
    <math|\<varepsilon\>\<gtr\>0> has complexity

    <\equation*>
      N=\<cal-O\><around*|(|\<varepsilon\><rsup|-n/m>|)>.
    </equation*>

    and this is the best possible <math|N>.

    If we assume that <math|f> has <math|d>-tree structure, then a deep
    network in some <math|D<rsub|N,d><rsup|n>> which approximates <math|f>
    with accuracy <math|\<varepsilon\>\<gtr\>0> has complexity

    <\equation*>
      N=\<cal-O\><around*|(|<around*|(|n-1|)>*\<varepsilon\><rsup|-d/m>|)>.
    </equation*>
  </theorem>

  The proof of the second part of the theorem is quite straightforward, given
  the first (which is known at least since <cite|pinkus_approximation_1999>)
  and starting from a network which mimics the compositional structure of
  <math|f>. Noting that each of the units of the deep network is a a shallow
  one with <math|n=d> inputs, which approximates the corresponding
  compositional unit of <math|f> within <math|\<varepsilon\>>, one need only
  apply the triangle inequality <math|n-1> times and a Lipschitz bound to
  conclude.<\footnote>
    Sloppily: Assume <math|P,P<rsub|i>> approximate
    <math|h,h<rsub|i\<nocomma\>>> within <math|\<varepsilon\>>, then compute
    <math|<around*|\<\|\|\>|h<around*|(|h<rsub|i>,h<rsub|j>|)>-P<around*|(|P<rsub|i>,P<rsub|j>|)>|\<\|\|\>><rsub|\<infty\>>\<leqslant\><around*|\<\|\|\>|h<around*|(|h<rsub|i>,h<rsub|j>|)>-h<around*|(|P<rsub|i>,P<rsub|j>|)>|\<\|\|\>><rsub|\<infty\>>+<around*|\<\|\|\>|h<around*|(|P<rsub|i>,P<rsub|j>|)>-P<around*|(|P<rsub|i>,P<rsub|j>|)>|\<\|\|\>><rsub|\<infty\>>\<leqslant\>L*<around*|\<\|\|\>|<around*|(|h<rsub|i>,h<rsub|j>|)>-<around*|(|P<rsub|i>,P<rsub|j>|)>|\<\|\|\>>+c*\<varepsilon\>\<lesssim\>\<varepsilon\>>.
  </footnote> Note that

  <\quotation>
    the deep network does not need to have exactly the same compositional
    architecture as the compositional function to be approximated. It is
    sufficient that the acyclic graph representing the structure of the
    function is a subgraph of the graph representing the structure of the
    deep network.
  </quotation>

  Let us state that again: when we assume that we want to approximate
  (smooth) functions with <math|d>-tree structure and use deep networks with
  analogous structure we only pay a polynomial price in complexity for an
  increase in input dimension.

  This leads to the following

  <\quotation>
    <\definition>
      The effective dimension of a class <math|W> of functions (for a given
      norm) is said to be <math|d> if for every
      <math|\<varepsilon\>\<gtr\>0>, any function in <math|W> can be
      recovered within an accuracy of <math|\<varepsilon\>> (as measured by
      the norm) using an appropriate network (either shallow or deep) with
      <math|C*\<varepsilon\><rsup|-d>> parameters with
      <math|C=\<cal-O\><around*|(|N|)>>.<\footnote>
        I added this last bit with <math|C=\<cal-O\><around*|(|N|)>> for
        better consistency with the main result. It is the exponent what
        matters in the definition.
      </footnote>
    </definition>
  </quotation>

  For example the effective dimension of <math|m>-times continuously
  differentiable functions of <math|n> variables is <math|n/m> and that of
  those which in addition have <math|d>-tree structure is <math|d/m>. Also
  the effective dimension of <math|Lip<around*|(|\<bbb-R\>|)>> is 1 and that
  of <math|Lip<around*|(|\<bbb-R\><rsup|2>|)>> is 2. As an example of the
  reduction of effective dimension by compositionality, consider the function
  <math|x,y\<mapsto\><around*|\||x<rsup|2>-y<rsup|2>|\|>>. It is Lipschitz of
  two variables, but if we see it as the composition of a polynomial in
  <math|P<rsub|2><rsup|2>> and the norm function, which is in
  <math|Lip<around*|(|\<bbb-R\>|)>>, one can show that a bi-layer network can
  approximate it with <math|\<cal-O\><around*|(|\<varepsilon\><rsup|-1>|)>>.

  We come now to a more general version of Theorem
  <reference|thm:main-result>, where the degree of compositionality is
  allowed to vary across nodes in the network as is typically the case in
  convnets, whose kernels are of varying sizes. Compositional functions are
  thus generalized to having the structure of any
  <hlink|DAG|https://en.wikipedia.org/wiki/Directed_acyclic_graph>
  <math|\<cal-G\>> and are called <strong|<math|\<cal-G\>>-functions>.

  <\theorem>
    <label|thm:dag-functions>Assume <math|f\<in\>C<rsup|m>> is a
    <math|\<cal-G\>>-function for some DAG <math|\<cal-G\>>. Let <math|V> be
    the set of non-input nodes of <math|\<cal-G\>>, <math|m<rsub|v>> the
    smoothness of the constituent function for <math|f> at node <math|v> and
    <math|d<rsub|v>> the in-degree of <math|v>. Fix an accuracy
    <math|\<varepsilon\>\<gtr\>0>. Then for any shallow network approximating
    <math|f> with <math|\<varepsilon\>> accuracy, its complexity is

    <\equation*>
      N<rsub|s>=\<cal-O\><around*|(|\<varepsilon\><rsup|-n/m>|)>,
    </equation*>

    where <math|m=min<rsub|v\<in\>V> m<rsub|v>>. However, for a deep network:

    <\equation*>
      N<rsub|d>=\<cal-O\><around*|(|<big|sum><rsub|v\<in\>V>\<varepsilon\><rsup|-d<rsub|v>/m<rsub|v>>|)>.
    </equation*>
  </theorem>

  <subsection*|Using ReLUs>

  The following extension of Theorem <reference|thm:main-result> should
  follow by a simple approximation argument. Note that the statement in the
  paper omits the condition that <math|f> be <math|d>-compositional, but it
  should be there!

  <\theorem>
    Let <math|f> be an <math|L>-Lipschitz function of <math|n> variables with
    <math|d>-tree structure. Then, the complexity of a shallow network which
    is a linear combination of ReLUs providing an approximation with accuracy
    at least <math|\<varepsilon\>\<gtr\>0> is

    <\equation*>
      N<rsub|s>=\<cal-O\><around*|(|<around*|(|<tfrac|\<varepsilon\>|L>|)><rsup|-n>|)>
    </equation*>

    whereas that of a deep (binary) compositional architecture is

    <\equation*>
      N<rsub|d>=\<cal-O\><around*|(|<around*|(|n-1|)><around*|(|<tfrac|\<varepsilon\>|L>|)><rsup|-d>|)>.
    </equation*>
  </theorem>

  Furthermore Appendix 4 of the paper provides an explicit construction for
  the piecewise approximation of Lipschitz functions. There it is shown that
  multilayer ReLU networks can perform piecewise constant approximation of
  (hierarchical compositions of) Lipschitz functions. The authors

  <\quotation>
    conjecture that the construction (...) that performs piecewise constant
    approximation is qualitatively similar to what deep networks may
    represent after training
  </quotation>

  because of the greedy way in which supervised training proceeds and how it
  relates to their construction.

  <subsection*|Gaps>

  We have seen that shallow networks will perform badly for general
  continuous functions and that deep hierarchical nets will perform very well
  (polynomial complexity) for hierarchical targets. But what are specific
  examples of functions which reveal the <em|gap in performance>? There are
  many such examples and the authors cite several and provide their own, see
  Section Ÿ4.2 in the paper.

  One can also ask the question of when functions generated by deep
  architectures cannot be efficiently (with a comparable number of units)
  generated by shallower networks. See <cite|lin_why_2016> for concrete
  examples of such <dfn|no-flattening theorems>: the fact that certain deep
  architectures cannot be made shallower without incurring an exponential
  penalty in the number of inputs. In particular, Lin and Tegmark prove that
  this is the case for <em|multiplication>, which is \Pthe prototypical
  compositional function\Q.

  <subsection*|Discussion on compositionality>

  Section Ÿ6 of the paper is devoted to a long list of observations on
  compositionality, approximation, sparsity, multi-class classification, DNNs
  as memories and general considerations from the perspective of the theory
  of computation. Just read it.

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|2>
      <bibitem*|1><label|bib-lin_why_2016>Henry<nbsp>W.<nbsp>Lin<localize|
      and >Max Tegmark.<newblock> Why does deep and cheap learning work so
      well?<newblock> <with|font-shape|italic|ArXiv:1608.08225 [cond-mat,
      stat]>, <localize|page >17, aug 2016.<newblock> ArXiv:
      1608.08225.<newblock>

      <bibitem*|2><label|bib-pinkus_approximation_1999>Allan
      Pinkus.<newblock> Approximation theory of the MLP model in neural
      networks.<newblock> <with|font-shape|italic|Acta Numerica>, 8:143\U195,
      1999.<newblock>
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|info-flag|detailed>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|4|?>>
    <associate|auto-3|<tuple|2|?>>
    <associate|auto-4|<tuple|5|?>>
    <associate|auto-5|<tuple|6|?>>
    <associate|auto-6|<tuple|6|?>>
    <associate|auto-7|<tuple|6|?>>
    <associate|bib-lin_why_2016|<tuple|1|?>>
    <associate|bib-pinkus_approximation_1999|<tuple|2|?>>
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
    <associate|thm:dag-functions|<tuple|5|?>>
    <associate|thm:main-result|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      lin_why_2016

      pinkus_approximation_1999

      lin_why_2016

      pinkus_approximation_1999

      lin_why_2016
    </associate>
    <\associate|figure>
      <tuple|normal|A function with binary tree structure and an ideal
      network approximating it.|<pageref|auto-1>>
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|Some definitions
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|Smooth activation functions
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|Using ReLUs
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|Gaps <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|Discussion on compositionality
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>