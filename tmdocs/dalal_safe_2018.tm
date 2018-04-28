<TeXmacs|1.99.6>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Safe exploration in continuous action spaces
  (WIP)>|<doc-author|<author-data|<author-name|Dalal, Gal et
  al...>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|reinforcement learning|constraints compliance>

  <strong|tl;dr:> Through a linearization trick, the task of perturbing an
  agent's policy to comply with a set of constraints is framed as an off-line
  training of a neural net based on log data and a simple quadratic program.

  <hrule>

  An agent in a Constrained MDP must follow a policy whose resulting actions
  <math|a> for states <math|s> do not violate a set of given <em|immediate
  constraints> <math|c<rsub|i>=c<rsub|i><around*|(|s,a|)>>. These are
  summarised (averaged in the case of non-deterministic policies) in
  per-state <em|safety signals> <math|<wide|c|\<bar\>><rsub|i>=<wide|c|\<bar\>><rsub|i><around*|(|s|)>>.
  These signals are approximated with a model linear in the actions with
  gradient determined by a neural network. Then they are used to solve a
  simple (thanks to the linearity) constrained optimization problem yielding
  a slightly perturbed policy which respects the constraints.

  Note that this holds during training too.

  Some insights:

  The policy is basically <math|l<rsup|2>> projected onto an affine space of
  admissible actions. The \Pslope\Q of this space is computed with a NN whose
  architecture is of little consequence. A shallow structure qith 10 hidden
  units is enough.

  A penalty method works worse due to bad condition numbers of the gradient:
  different scales in the constraints make the Lagrangian not well behaved.

  \;

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <bib-list|0|>
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
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>