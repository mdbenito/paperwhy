<TeXmacs|1.99.5>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Extrapolation and learning
  equations>|<doc-author|<author-data|<author-name|Georg,
  Martius>>>|<doc-author|<author-data|<author-name|Christoph H.,
  Lamport>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|intuitive physics|neural networks|model selection|sparsity>

  <strong|tl;dr:> Starting from the intuition that many physical dynamical
  systems are typically well modeled by first order systems of ODE with
  governing equations expressed in terms of a few elementary functions, the
  authors propose a fully connected architecture with multiple
  non-linearities with the purpose of learning <with|font-shape|italic|the
  formulae> for these systems of equations. The network effectively performs
  a kind of hierarchical, non-linear regression with the given nonlinearities
  as basis functions and is able to learn the governing equations for several
  examples like a compound pendulum or the forward kinematics of a robotic
  arm. Crucially, this approach provides good
  <with|font-shape|italic|extrapolation> performance to unexplored input
  regimes. For model selection (i.e. hyperparameter choice), competing
  solutions are scored based both on validation performance and crucially,
  computed model complexity, measured in number of terms in the equations.

  <hrule>

  Consider the task of designing an adequate model for e.g. the forward
  kinematics of a robotic arm. For the engineer, the goal is to find some
  \Psimple\Q set of equations with which to compute the state of all joints
  in phase space given any initial conditions. Designing such a model,
  potentially exhibiting complex couplings and non-linearities can be a
  challenging task so one alternative might be to try to learn a statistical
  model for it. However, standard regression will in many cases prove
  inadequate because the crucial assumption that the training data
  sufficiently represent the whole distribution may be violated, e.g. if
  measurements have not been taken in regimes beyond the normal operating
  regime of the robot.

  The authors propose <with|font-shape|italic|learning instead the equations
  for the physical model>, i.e. the equations themselves as algebraic
  expressions composed of coefficients and standard elementary operations
  like the identity, sums, sines, cosines and binary products.<\footnote>
    Division, square roots and logarithms are explicitly left for later work
    since their domains of definition are strict subsets of <math|\<bbb-R\>>,
    thus requiring special handling, e.g. via cut-offs. More on this in the
    last section.
  </footnote> By choosing the basis functions to have their support over all
  of <math|\<bbb-R\>> (with non negligible mass everywhere) the hope is that
  extrapolation to conditions not seen during training will be possible.

  In par goes an ad-hoc model selection technique: cross-validation is not
  adequate because it again hinges on the assumption that the whole data
  distribution is significantly captured with the training data.

  <section*|The model>

  <big-figure|<image|../static/img/martius_extrapolation_2016-fig1.jpg|0.8par|||>|The
  network architecture>

  Assume that some dynamics <math|y=\<phi\><around*|(|x|)>+\<varepsilon\>>
  are given by an unknown <math|\<phi\>:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\><rsup|m>>
  and <math|\<varepsilon\>> an <math|m>-dimensional R.V. with zero mean. The
  function <math|\<phi\>> is assumed to lie in a class <math|\<cal-C\>>
  consisting of compositions of algebraic expressions of <strong|elementary
  functions> (sum, product, sine, cosine, ...). As usual, the goal is to
  compute an estimator <math|<wide|\<phi\>|^>> in an adequate hypothesis
  space <math|\<cal-H\>>, such that risk wrt. the squared loss
  <math|R<around*|(|<wide|\<phi\>|^>|)>=\<bbb-E\><rsub|X,Y><around*|[|<around*|(|<wide|\<phi\>|^><around*|(|X|)>-Y|)><rsup|2>|]>>
  is minimized. Computation of <math|<wide|\<phi\>|^>> is done by means of
  minimization of the proxy <strong|empirical risk>

  <\equation*>
    <wide|R|^><around*|(|<wide|\<phi\>|^>|)>\<assign\><frac|1|N>*<big|sum><rsub|i=1><rsup|N><around*|\<\|\|\>|<wide|\<phi\>|^><around*|(|x<rsub|i>|)>-y<rsub|i>|\<\|\|\>><rsup|2>.
  </equation*>

  The proposed estimator (<em|<strong|EQ>uation <strong|L>earner>) is similar
  to hierarchical, non-linear regression with basis functions, in the form of
  a standard, fully connected, feed forward neural network

  <\equation*>
    <wide|\<phi\>|^><rsub|N><around*|(|x|)>=y<rsup|<around*|(|L|)>>\<circ\>y<rsup|<around*|(|L-1|)>>\<cdots\>\<circ\>y<rsup|<around*|(|1|)>><around*|(|x|)>
  </equation*>

  where the layers <math|1,\<ldots\>,L-1> are the standard composition of a
  linear mapping <math|z<rsup|<around*|(|l|)>>=W<rsup|<around*|(|l|)>>*x<rsup|<around*|(|l-1|)>>+b<rsup|*l>>
  with a non linearity. And there lies the key contribution:

  <\equation*>
    y<rsup|<around*|(|l|)>><around*|(|z|)>=<around*|(|f<rsub|1><around*|(|z<rsub|1><rsup|<around*|(|l|)>>|)>,\<ldots\>,f<rsub|u><around*|(|z<rsub|u><rsup|<around*|(|l|)>>|)>,g<rsub|1><rsup|<around*|(|l|)>><around*|(|z<rsub|u+1>,z<rsub|u+2>|)>,\<ldots\>,g<rsub|v><rsup|<around*|(|l|)>><around*|(|z<rsub|u+2*v-1>,z<rsub|u+2*v>|)>|)>.
  </equation*>

  Here <math|f<rsub|i>> are <em|unary> maps (identity, \ <math|sin>,
  <math|cos>, <math|sigm>) and <math|g<rsub|j>> are <em|binary> units,
  currently only multiplication of their inputs, but see below. Note that it
  is essential to be able to efficiently model multiplication but full
  multiplication of all entries might lead to polynomials of very high
  degree, which are uncommon in physical models (but shouldn't this be sorted
  out by the optimization / model selection?).

  <section*|Training and model selection>

  The objective function is complemented by an <math|L<rsub|1>> penalty to
  induce sparsity (as in the Lasso):

  <\equation*>
    \<cal-L\><around*|(|<wide|\<phi\>|^><rsub|N>|)>=<wide|R|^><around*|(|<wide|\<phi\>|^><rsub|N>|)>+\<lambda\>*<big|sum><rsub|j=1><rsup|L><around*|\<\|\|\>|W<rsup|<around*|(|l|)>>|\<\|\|\>><rsub|1>
  </equation*>

  This is actually done via three-stage optimization: for the first
  <math|t<rsub|1>> steps no penalty is used, then, until some later step
  <math|t<rsub|2>> the lasso term is activated and thereafter deactivated but
  small weights are clamped to zero and forced to remain there. The goal of
  this procedure is to let coefficients adjust during the first epochs
  without being subject to the driving force of the penalty term (which
  artificially pushes them to lower values), then ensure that lower weights
  stay there without driving others further down. [Given the ad-hoc nature of
  this method, it might be interesting to see what happens with some form of
  <strong|best subset selection>, like e.g. <with|font-series|bold|greedy
  forward regression>.<\footnote>
    However, see <cite|hastie_extended_2017>.
  </footnote>]

  Model selection (choosing how many layers how wide and with how many
  non-linearities of which kind) proves to be tricky: standard
  cross-validation requires that sampling from training data be
  representative of the full data distribution, but precisely one of the
  desired abilities of this model is that it extrapolate (generalize) beyond
  its input to data ranges not represented in the training data. Therefore
  the authors propose a two-goal objective to choose the best architecture
  among a set <math|<around*|{|\<phi\><rsub|k>|}><rsub|k=1><rsup|K>>:adequate

  <\equation*>
    <below|argmin|k=1,\<ldots\>,K> <around*|[|<around*|(|r<rsup|v><around*|(|\<phi\><rsub|k>|)>|)><rsup|2>+<around*|(|r<rsup|s><around*|(|\<phi\><rsub|k>|)>|)><rsup|2>|]>,
  </equation*>

  where <math|r<rsup|v>,r<rsup|s>:<around*|{|\<phi\><rsub|k>|}><rsub|k=1>\<rightarrow\><around*|{|1,\<ldots\>,K|}>>
  sort (rank) all <math|K> models respectively by validation accuracy and and
  complexity (measured as the number of units with activation above a given
  threshold). This is a way of embedding both measures into a common space
  (<math|<around*|{|1,\<ldots\>,K|}>>) for joint optimization.

  Because these values might correspond to (possibly poor) local optima
  subject to the initial values of the parameters, multiple runs are used to
  \Pestimate error bars\Q.

  <subsection*|Some ideas>

  <strong|Note:> this section does not report results of the paper.

  The last fact rises the point of whether one could define population
  quantities <math|\<rho\><rsup|v>,\<rho\><rsup|s>> over all possible
  hypothesis spaces <math|\<cal-H\>>. <math|\<rho\><rsup|v>> might encode
  e.g. the minimal risk (i.e. expected loss)
  <math|<below|min|<wide|\<phi\>|^>\<in\>\<cal-H\>>
  R<around*|(|<wide|\<phi\>|^>|)>>, which would be <math|0> iff
  <math|\<cal-H\>\<cap\>\<cal-C\>\<neq\>\<emptyset\>>, and
  <math|\<rho\><rsup|s>> might encode e.g. the capacity of <math|\<cal-H\>>
  or some measure of its complexity. Estimates as to the accuracy of some
  sample-approximation to these quantities would then be necessary.

  An alternative idea to explore could be Bayesian model selection. Basically
  one postulates some prior over a set of hypothesis spaces
  <math|<around*|{|\<cal-H\><rsub|k>|}><rsub|k=1><rsup|K>>, then computes the
  posterior of the data given some hypothesis by marginalizing over parameter
  space <math|\<Omega\>>:

  <\equation*>
    p<around*|(|\<b-t\>\|\<cal-H\><rsub|k>|)>=<big|int><rsub|\<Omega\>>p<around*|(|\<b-t\>\|W,\<cal-H\><rsub|k>|)>*p<around*|(|W\|\<cal-H\><rsub|k>|)>*\<mathd\>W.
  </equation*>

  This integral will probably not be tractable so it will have to be
  approximated using MCMC or some other method.

  <section*|Experiments>

  It is just easier if you check them in the paper itself. Suffice to say
  that both pendulum and double pendulum equations were easily learned but
  also that the hypothesis space <math|\<cal-H\>> needs to intersect
  <math|\<cal-C\>> or performance can be quite poor (e.g. if trying to
  integrate functions without an antiderivative or in an example with a
  rolling cart attached to a wall through a spring).

  <big-figure|<image|../static/img/martius_extrapolation_2016-fig3.jpg|1par|||>|Double
  pendulum data and extrapolation results for Multi Layer Perceptron, Support
  Vector Regression and Equation Learner.>

  Recent work improves on the \Pbad\Q examples.

  <section*|Recent extensions>

  In a recent (yesterday!) talk at <hlink|Munich's Deep Learning
  Meetup|https://www.meetup.com/deep-learning-meetup-munich/events/243910570/>,
  Martius presented some recent developments around this model, most notably
  the introduction of <strong|division units>. Because of the pole at 0, they
  decided to restrict the domain to positive reals bounded away from zero
  (<math|z\<geqslant\>\<theta\>\<gtr\>0>), while adding a penalty on negative
  inputs to the unit. The following slide displays their regularized division
  unit.

  <big-figure|<image|../static/img/martius_deep_2017-slide1.jpg|1par|||>|The
  speaker tries to answer a peculiar question.>

  Results of course vastly improve in examples involving quotients. This
  paves the road for further inclusions, like arbitrary exponentiation or
  logarithms.

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|1>
      <bibitem*|1><label|bib-hastie_extended_2017>Trevor Hastie, Robert
      Tibshirani<localize|, and >Ryan<nbsp>J.<nbsp>Tibshirani.<newblock>
      Extended Comparisons of Best Subset Selection, Forward Stepwise
      Selection, and the Lasso.<newblock>
      <with|font-shape|italic|ArXiv:1707.08692 [stat]>, jul 2017.<newblock>
      ArXiv: 1707.08692.<newblock>
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|preamble|false>
    <associate|save-aux|true>
  </collection>
</initial>

<\attachments>
  <\collection>
    <\associate|bib-bibliography>
      <\db-entry|+JuKABBlPIYxuHN|article|hastie_extended_2017>
        <db-field|contributor|miguel>

        <db-field|modus|imported>

        <db-field|date|1508596120>
      <|db-entry>
        <db-field|author|Trevor <name|Hastie><name-sep>Robert
        <name|Tibshirani><name-sep>Ryan J. <name|Tibshirani>>

        <db-field|title|Extended Comparisons of Best Subset Selection,
        Forward Stepwise Selection, and the Lasso>

        <db-field|journal|arXiv:1707.08692 [stat]>

        <db-field|year|2017>

        <db-field|month|jul>

        <db-field|note|arXiv: 1707.08692>

        <db-field|url|<slink|http://arxiv.org/abs/1707.08692>>

        <db-field|abstract|In exciting new work, Bertsimas et al. (2016)
        showed that the classical best subset selection problem in regression
        modeling can be formulated as a mixed integer optimization (MIO)
        problem. Using recent advances in MIO algorithms, they demonstrated
        that best subset selection can now be solved at much larger problem
        sizes that what was thought possible in the statistics community.
        They presented empirical comparisons of best subset selection with
        other popular variable selection procedures, in particular, the lasso
        and forward stepwise selection. Surprisingly (to us), their
        simulations suggested that best subset selection consistently
        outperformed both methods in terms of prediction accuracy. Here we
        present an expanded set of simulations to shed more light on these
        comparisons. The summary is roughly as follows: (a) neither best
        subset selection nor the lasso uniformly dominate the other, with
        best subset selection generally performing better in high
        signal-to-noise (SNR) ratio regimes, and the lasso better in low SNR
        regimes; (b) best subset selection and forward stepwise perform quite
        similarly throughout; (c) the relaxed lasso (actually, a simplified
        version of the original relaxed estimator defined in Meinshausen,
        2007) is the overall winner, performing just about as well as the
        lasso in low SNR scenarios, and as well as best subset selection in
        high SNR scenarios.>

        <db-field|file|Hastie et al. - 2017 - Extended Comparisons of Best
        Subset Selection, For.pdf:/Users/miguel/Library/Application
        Support/Zotero/Profiles/1fz1fwch.default/zotero/storage/RAJUFDVX/Hastie
        et al. - 2017 - Extended Comparisons of Best Subset Selection,
        For.pdf:application/pdf>
      </db-entry>
    </associate>
  </collection>
</attachments>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?|template.tm>>
    <associate|auto-2|<tuple|1|?|template.tm>>
    <associate|auto-3|<tuple|1|?|template.tm>>
    <associate|auto-4|<tuple|2|?|template.tm>>
    <associate|auto-5|<tuple|2|?|template.tm>>
    <associate|auto-6|<tuple|2|?|template.tm>>
    <associate|auto-7|<tuple|2|?|template.tm>>
    <associate|auto-8|<tuple|3|?|template.tm>>
    <associate|auto-9|<tuple|3|?|template.tm>>
    <associate|bib-hastie_extended_2017|<tuple|1|?|template.tm>>
    <associate|footnote-1|<tuple|1|?|template.tm>>
    <associate|footnote-2|<tuple|2|?|template.tm>>
    <associate|footnr-1|<tuple|1|?|template.tm>>
    <associate|footnr-2|<tuple|2|?|template.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      hastie_extended_2017
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|The
      model> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>