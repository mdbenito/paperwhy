<TeXmacs|1.99.4>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|a talk <localize|by>>>
  </hide-preamble>

  <\doc-data|<doc-title|On gradient-based optimization:
  <new-line>accelerated, stochastic, asynchronous,
  distributed>|<doc-author|<author-data|<author-name|Jordan, Michael
  I.>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|optimization|accelerated-gradient-descent|talk>

  Today's post is about another great talk given at the <name|Simons
  Institute for the Theory of Computing> in the context of their currently
  ongoing series <hlink|Computational Challenges in Machine
  Learning|https://simons.berkeley.edu/workshops/machinelearning2017-3>.

  <subsection*|Part 1: Variational, Hamiltonian and Symplectic Perspectives
  on Acceleration>

  For convex functions, Nesterov accelerated gradient descent method attains
  the optimal rate of <math|\<cal-O\><around*|(|1/k<rsup|2>|)>>.<\footnote>
    Since we are in a convex setting, there is a global minimum: if you know
    it, then you attain it in one step. Besides the trivial case, if one has
    higher order derivatives, then higher order methods provide faster
    convergence rates and so on. For this reason a definition of optimality
    in the sense of an oracle <todo|was introduced>: the oracle is only
    allowed to look at gradients under some constraint, in particular it has
    no access to the gradient at every point. It is under this restriction
    that Nesterov's gradient descent achieves optimality.
  </footnote><\footnote>
    See <cite|nesterov_introductory_2004>.
  </footnote>

  <\equation>
    <label|eq:nesterov><choice|<tformat|<table|<row|<cell|y<rsub|k+1>>|<cell|=>|<cell|x<rsub|k>-\<beta\>*\<nabla\>f<around*|(|x<rsub|k>|)>>>|<row|<cell|x<rsub|k+1>>|<cell|=>|<cell|<around*|(|1-\<lambda\><rsub|k>|)>*y<rsub|k+1>+\<lambda\><rsub|k>*y<rsub|k>.>>>>>
  </equation>

  Note that this is not actually gradient descent since the momentum will
  make the trajectory deviate from the \Psteepest slope\Q at some point.

  This reminds of <hlink|leap-frog integration|https://en.wikipedia.org/wiki/Leapfrog_integration>.
  Gradient descent is a discretization of gradient flow:

  <\equation*>
    <wide|X|\<dot\>><rsub|t>=-\<nabla\>f<around*|(|X<rsub|t>|)>.
  </equation*>

  Nesterov's method is the discretisation of the ODE<\footnote>
    <cite|su_differential_2016> write a finite differences equation for
    <eqref|eq:nesterov>, take limit as the stepsize goes to zero and find the
    continuous equation.\ 
  </footnote>

  <\equation>
    <label|eq:su-boyd-candes-ode><wide|X|\<ddot\>><rsub|t>+<frac|3|t>*<wide|X|\<dot\>><rsub|t>+\<nabla\>f<around*|(|X<rsub|t>|)>=0.
  </equation>

  <\quotation>
    <\question>
      These ODEs are obtained by taking continuous time limits. Is there a
      deeper generative mechanism?
    </question>
  </quotation>

  <subsubsection*|The Lagrangian point of view>

  For a target function <math|f> to optimize, define the <dfn|Bregman
  Lagrangian>

  <\equation>
    <label|eq:lagrangian>\<cal-L\><around*|(|x,<wide|x|\<dot\>>,t|)>=\<mathe\><rsup|\<gamma\><rsub|t>+\<alpha\><rsub|t>>*<around*|(|D<rsub|h><around*|(|x+\<mathe\><rsup|-\<alpha\><rsub|t>>*<wide|x|\<dot\>>,x|)>-\<mathe\><rsup|\<beta\><rsub|t>>*f<around*|(|x|)>|)>,
  </equation>

  where the exponentials and parameters provide degrees of freedom for later
  fine-tuning, <math|D<rsub|h>> is the <hlink|<dfn|Bregman
  divergence>|https://en.wikipedia.org/wiki/Bregman_divergence>

  <\equation*>
    D<rsub|h><around*|(|y,x|)>=h<around*|(|y|)>-h<around*|(|x|)>-<around*|\<langle\>|\<nabla\>h<around*|(|x|)>,y-x|\<rangle\>>
  </equation*>

  taken between <math|x> and <math|x> plus some (scaled) velocity
  <math|<wide|x|\<dot\>>>, and <math|h> is the convex
  <dfn|distance-generating function> for <math|D<rsub|h>>. Note that if one
  takes <math|h<around*|(|x|)>=x<rsup|2>/2>, then
  <math|D<rsub|h><around*|(|\<ldots\>|)>=<frac|1|2>*<around*|\<\|\|\>|<wide|x|\<dot\>>|\<\|\|\>><rsup|2>>
  is the kinetic energy so we always interpret this term as such and the
  second one <math|-\<mathe\><rsup|\<beta\><rsub|t>>*f<around*|(|x|)>> as the
  potential energy whose well we are going down. The choice of <math|h> will
  depend on the geometry of the problem, i.e. on the space where minimization
  happens. <todo|(more...?)>

  The scaling functions <math|\<alpha\><rsub|t>,\<beta\><rsub|t>,\<gamma\><rsub|t>>
  are in fact fixed by certain <dfn|ideal scaling conditions> reducing them
  to <strong|one> effective degree of freedom. This constraint has been
  designed to obtain the desired rates below, but the whole parameter space
  has not been explored.

  The Euler-Lagrange equation for the minimization over paths with starting
  point <math|X<rsub|0>>

  <\equation>
    <label|eq:minimization-paths>min<rsub|X>
    <big|int>\<cal-L\><around*|(|X<rsub|t>,<wide|X|\<dot\>><rsub|t>,t|)>*\<mathd\>t
  </equation>

  is called the non-homogenous <dfn|master ODE>:<\footnote>
    Note that this has roughly the form of a damped oscillator with the
    additional \Pgeometric term\Q involving the Hessian of the distance
    generating function, evaluated at \P<math|X> plus velocity\Q (yieldieng
    the acceleration).
  </footnote>

  <\equation>
    <label|eq:master-ode><wide|X|\<ddot\>><rsub|t>+<around*|(|\<mathe\><rsup|\<alpha\><rsub|t>>-<wide|\<alpha\>|\<dot\>><rsub|t>|)>*<wide|X|\<dot\>><rsub|t>+\<mathe\><rsup|2*\<alpha\><rsub|t>+\<beta\><rsub|t>>*<around*|[|\<nabla\><rsup|2>h<around*|(|X<rsub|t>+\<mathe\><rsup|-\<alpha\><rsub|t>>*<wide|X|\<dot\>><rsub|t>|)>|]><rsup|-1>*\<nabla\>f<around*|(|X<rsub|t>|)>=0.
  </equation>

  The claim is that

  <\quotation>
    this is going to generate (essentially) all known accelerated gradient
    methods in continuous time.
  </quotation>

  The first result is a rate in continuous time:<\footnote>
    Proved in one line with an adequate Lyapunov function. Note however that
    reparametrizing the equation can change the rate, so this is not
    groundbreaking news for the continuous equation. It is the passage to the
    discrete setting and the conditions under which the rate can or cannot be
    achieved that matter. <todo|?>
  </footnote>

  <\quotation>
    <\theorem>
      Under ideal scaling, the E-L equation <eqref|eq:master-ode> has
      convergence rate

      <\equation*>
        f<around*|(|X<rsub|t>|)>-f<around*|(|x<rsup|\<star\>>|)>\<leqslant\>\<cal-O\><around*|(|\<mathe\><rsup|-\<beta\><rsub|t>>|)>
      </equation*>

      to the optimum <math|x<rsup|\<star\>>>.
    </theorem>
  </quotation>

  <strong|In discrete time>, for general smooth convex problems it is known
  that this rate cannot be attained, although for uniformly convex ones it
  can. So, what is going on?

  Suppose we had <math|\<beta\><rsub|t>=p*log t+log C>, then
  <math|\<alpha\><rsub|t>,\<gamma\><rsub|t>> are fixed by the ideal scaling
  relations and <eqref|eq:master-ode> has
  <math|\<cal-O\><around*|(|\<mathe\><rsup|-\<beta\><rsub|t>>|)>=\<cal-O\><around*|(|1/t<rsup|p>|)>>.
  The master ODE is now

  <\equation*>
    <wide|X|\<ddot\>><rsub|t>+<frac|p+1|t>*<wide|X|\<dot\>><rsub|t>+C*p<rsup|2>*t<rsup|p-2>*<around*|[|\<nabla\><rsup|2>h<around*|(|X<rsub|t>+<dfrac|t|p>*<wide|X|\<dot\>><rsub|t>|)>|]><rsup|-1>*\<nabla\>f<around*|(|X<rsub|t>|)>=0.
  </equation*>

  With <math|p=2> one obtains <eqref|eq:su-boyd-candes-ode>. Plugging
  different values of <math|p> yields other methods (like accelerated
  cubic-regularized Newton's method for <math|p=3>). The interesting point is
  that <strong|<math|\<cal-L\>> is a covariant operator>: a reparametrization
  of time (in particular a change in <math|p>) <em|does not change the
  solution path>.

  <\quotation>
    Under these assumptions we have an optimal way to optimze: there is a
    particular path and acceleration is just changing the speed at which you
    move along it.
  </quotation>

  Note that this is not a property of gradient flow: reparametrization
  changes the path. In general it will be different from the one obtained
  from <eqref|eq:minimization-paths>.

  <\question*>
    <dueto|audience, \PNahdi\Q?>Is it possible to introduce new parameters
    into the master ODE (or change the current ones) to interpolate in some
    way between Nesterov-like methods and gradient flow?
  </question*>

  The answer is that indeed, the whole range of
  <math|\<alpha\><rsub|t>,\<beta\><rsub|t>,\<gamma\><rsub|t>> has not been
  exhausted and \Pwe could recover other algorithms by exploring [it]\Q.

  <subsubsection*|Discretizing the E-L equation (1)>

  (While preserving stability and the convergence rate). As usual, reduce to
  1st order system and apply e.g. an Euler scheme to obtain an algorithm.
  Problem: it is not stable! (and it lost the rate)

  <big-figure|<image|../static/img/jordan_gradient_2017-slide29.jpg|1par|||>|Instability
  of conventional methods for the master ODE.>

  Try Runge-Kutta, whatever: they all lose stability and the rate.

  Two approaches: \Preverse-engineer Nesterov estimate sequence technique\Q
  interpreting them as a discretization method or symplectic integration (see
  below). For the first one it is possible to recover oracle rates by
  increasing the assumptions on <math|f>:

  <\quotation>
    <\theorem>
      Assume <math|h> is uniformly convex and introduce an auxiliary sequence
      <math|y<rsub|k>> into the \Pnaive\Q Euler discretization. Assuming a
      certain condition on <math|\<nabla\>f<around*|(|y<rsub|k>|)>>:

      <\equation*>
        f<around*|(|y<rsub|k>|)>-f<around*|(|x<rsup|\<star\>>|)>\<leqslant\>\<cal-O\><around*|(|<frac|1|\<varepsilon\>*k<rsup|p>>|)>.
      </equation*>
    </theorem>
  </quotation>

  <subsubsection*|Discretizing the E-L equation: symplectic integration>

  A way of performing integration in time which conserves quantities like
  energy, momentum, etc. by switching to a (time-dependent) Hamiltonian
  framework. Take the Legendre transform (a.k.a. Fenchel conjugate) of the
  velocity and time to obtain momentum and energy respectively. The
  Hamiltionian has the form <eqref|eq:lagrangian> modulo constants and signs.
  Solve Hamilton's equations in phase space. For the discretization, look at
  and conserve a certain local volume tensor / differential form along the
  path of integration. This achieves faster rates. <todo|elaborate / see
  Harrer et al. Geometric Functional...>

  <subsubsection*|Ongoing / future / related work>

  Non-convex setting: the framework described can be applied as well.
  Stochastic setting: there will probably also exist an \Poptimal way to
  diffuse\Q in SDEs derived from some Focker-Planck type equation.

  Symplectic intregrators are used in <em|hybrid Montecarlo>, where one
  writes a Hamiltonian, etc.

  <\bibliography|bib|tm-ieeetr|paperwhy.bib>
    <\bib-list|2>
      <bibitem*|1><label|bib-nesterov_introductory_2004>Y.<nbsp>Nesterov,
      <with|font-shape|italic|Introductory Lectures on Convex Optimization -
      A Basic Course>.<newblock> No.<nbsp>87<localize| in >Applied
      Optimization, Springer, 2004.<newblock> Citecount: 02412.<newblock>

      <bibitem*|2><label|bib-su_differential_2016>W.<nbsp>Su,
      S.<nbsp>Boyd<localize|, and >E.<nbsp>J.<nbsp>Cand�s, ``A differential
      equation for modeling Nesterov's accelerated gradient method: Theory
      and insights,'' <with|font-shape|italic|Journal of Machine Learning
      Research>, vol.<nbsp>17, no.<nbsp>153, pp.<nbsp>1\U43, 2016.<newblock>
      Citecount: 00083 arXiv: 1503.01243.<newblock>
    </bib-list>
  </bibliography>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|?|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|2|?>>
    <associate|auto-4|<tuple|1|?>>
    <associate|auto-5|<tuple|3|?>>
    <associate|auto-6|<tuple|3|?>>
    <associate|auto-7|<tuple|3|?>>
    <associate|bib-nesterov_introductory_2004|<tuple|1|?>>
    <associate|bib-su_differential_2016|<tuple|2|?>>
    <associate|eq:lagrangian|<tuple|3|?>>
    <associate|eq:master-ode|<tuple|5|?>>
    <associate|eq:minimization-paths|<tuple|4|?>>
    <associate|eq:nesterov|<tuple|1|?>>
    <associate|eq:su-boyd-candes-ode|<tuple|2|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnote-3|<tuple|3|?>>
    <associate|footnote-4|<tuple|4|?>>
    <associate|footnote-5|<tuple|5|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
    <associate|footnr-3|<tuple|3|?>>
    <associate|footnr-4|<tuple|4|?>>
    <associate|footnr-5|<tuple|5|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      nesterov_introductory_2004

      su_differential_2016
    </associate>
    <\associate|figure>
      <tuple|normal|Instability of conventional methods for the master
      ODE.|<pageref|auto-4>>
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|Part 1: Variational, Hamiltonian and
      Symplectic Perspectives on Acceleration
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|2tab>|The Lagrangian point of view
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|2tab>|Discretizing the E-L equation (1)
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|2tab>|Discretizing the E-L equation: symplectic
      integration <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|2tab>|Ongoing / future / related work
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>