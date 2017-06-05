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

  <hugo|youtube|VE2ITg_hGnI>

  <subsection*|Part 1: Variational, Hamiltonian and Symplectic Perspectives
  on Acceleration>

  For convex functions, Nesterov accelerated gradient descent method attains
  the optimal rate of <math|\<cal-O\><around*|(|1/k<rsup|2>|)>>.<\footnote>
    Since we are in a smooth convex setting, there is a global minimum: if
    you know it, then you trivially attain it in one step. Besides this
    useless case, if one has higher order derivatives, then higher order
    methods provide faster convergence rates and so on. For this reason one
    needs to restrict the definition of optimality in some sense. The concept
    of an oracle was introduced with this purpose: An oracle is an entity
    which \Panswers\Q questions of an optimization scheme about the function
    to be optimized (e.g. what is the value of <math|\<nabla\>f> at
    <math|x<rsub|k>>?), within the model of the problem (i.e. it represents
    the things known to the method). This leads to the definition of the
    class of <dfn|smooth first order methods> as those which only have access
    to gradient infromation and produce iterates of the form
    <math|x<rsub|k>\<in\>x<rsub|0>+span<around*|{|\<nabla\>f<around*|(|x<rsub|0>|)>,\<ldots\>,\<nabla\>f<around*|(|x<rsub|k-1>|)>|}>>.
    It is under this restriction that Nesterov's gradient descent achieves
    the optimal rate of <math|\<cal-O\><around*|(|1/k<rsup|2>|)>>. See e.g.
    <cite-detail|nesterov_introductory_2004|Ch. 1> for more on oracles and
    Ÿ2.1.2 for the previous statements, originally proved in
    <cite|nesterov_method_1983>.
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
  happens.

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
    The proof is a one-liner choosing and adequate Lyapunov function. Note
    however that (as explained later) by reparametrizing the equation one can
    change the rate, so this is not groundbreaking news for the continuous
    setting. It is the passage to the discrete setting and the conditions
    under which the rate can or cannot be achieved that matter.
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
  exhausted and \P<em|we could recover other algorithms by exploring [it]>\Q.

  <subsubsection*|Discretizing the E-L equation (1)>

  (While preserving stability and the convergence rate). The first thing one
  thinks of is to reduce the 2nd order equation to a 1st order system and
  apply e.g. an Euler scheme to obtain an algorithm. The problem is that the
  method is not stable! (and it lost the rate)

  <big-figure|<image|../static/img/jordan_gradient_2017-slide29.jpg|1par|||>|Instability
  of conventional methods for the master ODE.>

  Then one can try Runge-Kutta or whatever: they all lose stability and the
  rate. Jordan's group saw two approaches for solving this problem:
  \Preverse-engineer the Nesterov estimate sequence technique\Q interpreting
  it as a discretization method or use symplectic integration (see below).
  For the first one it is possible to recover oracle rates by increasing the
  assumptions on <math|f>:

  <\quotation>
    <\theorem>
      Assume <math|h> is uniformly convex and introduce an auxiliary sequence
      <math|y<rsub|k>> into the \Pnaive\Q Euler discretization. Assuming
      higher regularity on <math|f<around*|(|y<rsub|k>|)>>:

      <\equation*>
        f<around*|(|y<rsub|k>|)>-f<around*|(|x<rsup|\<star\>>|)>\<leqslant\>\<cal-O\><around*|(|<frac|1|\<varepsilon\>*k<rsup|p>>|)>.
      </equation*>
    </theorem>
  </quotation>

  But one typically does not want to have to assume these additional
  conditions on <math|\<nabla\>f> (Lipschitz <math|p-1> derivatives).

  <subsubsection*|Discretizing the E-L equation (2): symplectic integration>

  <hlink|Symplectic integration|https://en.wikipedia.org/wiki/Symplectic_integrator>
  is a numerical integration technique which conserves quantities like energy
  and momentum in a (time-dependent) Hamiltonian framework.<\footnote>
    Symplectic intregrators are also used in the <dfn|<hlink|hybrid
    Montecarlo method|https://en.wikipedia.org/wiki/Hybrid_Monte_Carlo>>, an
    MCMC technique where the transition between states is governed by
    Hamiltonian dynamics.
  </footnote> By taking the Legendre transform (a.k.a. Fenchel conjugate) of
  the velocity and time one obtains momentum and energy respectively and can
  write a Hamiltionian which basically mimics the Lagrangian we had (it has
  the form <eqref|eq:lagrangian> modulo constants and signs). Then one solves
  Hamilton's equations in phase space with a discretization which looks at
  and conserves a certain local volume tensor / differential form along the
  path of integration.

  Why is this interesting in our setting? The discretization seen for the
  Lagrangian formulation suffers from high sensitivity to step size and a
  momentum build-up which hurts performance near the optimum. The Hamiltonian
  perspective produces equivalent equations whose symplectic integration
  should alleviate these issues and achieve faster rates without further
  assumptions : conservation of momentum seems to be the key.

  <big-figure|<image|../static/img/jordan_gradient_2017-slide42.jpg|1par|||>|Comparing
  Lagrangian and Symplectic integrators.>

  <subsubsection*|Ongoing / future / related work>

  Two possible venues for exploration are:

  <\itemize-dot>
    <item>Non-convex setting: the framework described can be applied as well.

    <item>Stochastic equations: there will probably also exist an \Poptimal
    way to diffuse\Q in SDEs derived from some Focker-Planck type equation.
  </itemize-dot>

  <\bibliography|bib|tm-ieeetr|paperwhy.bib>
    <\bib-list|3>
      <bibitem*|1><label|bib-nesterov_introductory_2004>Y.<nbsp>Nesterov,
      <with|font-shape|italic|Introductory Lectures on Convex Optimization -
      A Basic Course>.<newblock> No.<nbsp>87<localize| in >Applied
      Optimization, Springer, 2004.<newblock>

      <bibitem*|2><label|bib-nesterov_method_1983>Y.<nbsp>Nesterov, ``A
      method of solving a convex programming problem with convergence rate
      <math|\<cal-O\>(1/k<rsup|2>)>,'' <with|font-shape|italic|Soviet
      Mathematics Doklady>, vol.<nbsp>27, pp.<nbsp>372\U376, 1983.

      <bibitem*|3><label|bib-su_differential_2016>W.<nbsp>Su,
      S.<nbsp>Boyd<localize|, and >E.<nbsp>J.<nbsp>Candès, ``A differential
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
    <associate|auto-6|<tuple|2|?>>
    <associate|auto-7|<tuple|2|?>>
    <associate|auto-8|<tuple|<with|mode|<quote|math>|\<bullet\>>|?>>
    <associate|bib-nesterov_introductory_2004|<tuple|1|?>>
    <associate|bib-nesterov_method_1983|<tuple|2|?>>
    <associate|bib-su_differential_2016|<tuple|3|?>>
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

      nesterov_method_1983

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

      <with|par-left|<quote|2tab>|Discretizing the E-L equation (2):
      symplectic integration <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
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