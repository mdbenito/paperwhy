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
  In this manner policies with low likelihood of constraint violation are
  obtained in few iterations.

  <hrule>

  <subsection|The setting>

  In a <dfn|Constrained MDP>, an agent must follow a policy
  <math|\<mu\>:S\<rightarrow\>A> which does not violate a set of given
  constraints encoded by so-called <em|safety signals>
  <math|<wide|c|\<bar\>><rsub|i>:S\<rightarrow\>\<bbb-R\>>,
  <math|i\<in\><around*|[|K|]>>. The goal is to maximise the expected
  discounted reward, under the condition that
  <math|<wide|c|\<bar\>><rsub|i><around*|(|s<rsub|t>|)>\<leqslant\>C<rsub|i>>
  for all states <math|s<rsub|t>> at all timesteps <math|t> and
  <math|i\<in\><around*|[|K|]>>. The mappings <math|<wide|c|\<bar\>><rsub|i>>
  are after-the-fact measurements of physical quantities once the agent is in
  a given state and must thus be predicted if the constraints are to be
  respected. This is achieved with a linear model based on estimated
  <dfn|immediate constraints> <math|c<rsub|i>:S\<times\>A\<rightarrow\>\<bbb-R\>>.
  For a transition <math|<around*|(|s,a|)>\<mapsto\>s<rprime|'>>:

  <\equation*>
    <wide|c|\<bar\>><rsub|i><around*|(|s<rprime|'>|)>=c<rsub|i><around*|(|s,a|)>\<simeq\><wide|c|\<bar\>><rsub|i><around*|(|s|)>+g<around*|(|s;w<rsub|i>|)>\<cdot\>a,
  </equation*>

  where <math|g<around*|(|\<cdot\>;w<rsub|i>|)>:S\<rightarrow\>\<bbb-R\><rsup|d>>
  is a neural network for each signal <math|<wide|c|\<bar\>><rsub|i>> and
  <math|d=dim A>. This is a

  <\quotation>
    rst-order approximation to <math|c<rsub|i>(s,a)> with respect to
    <math|a>; i.e., an explicit representation of sensitivity of changes in
    the safety signal to the action using features of the state.
  </quotation>

  <subsection|A linear approximation to the safety signals>

  Based on recorded data of transitions <math|D=<around*|{|<around*|(|s<rsub|j>,a<rsub|j>,s<rprime|'><rsub|j>|)>:j=1,\<ldots\>,N|}>>
  and the corresponding measurements <math|<wide|c|\<bar\>><rsub|i><around*|(|s<rsub|j>|)>,<wide|c|\<bar\>><rsub|i><around*|(|s<rprime|'><rsub|j>|)>>
  the networks <math|g<rsub|i>=g<around*|(|\<cdot\>;w<rsub|i>|)>> are trained
  by solving the least squares problems

  <\equation*>
    w<rsub|i><rsup|\<star\>>\<in\><below|argmin|w<rsub|i>\<in\>\<bbb-R\><rsup|p>>
    <big|sum><rsub|j=1><rsup|N><around*|[|<wide|c|\<bar\>><rsub|i><around*|(|s<rsub|j><rprime|'>|)>-g<rsub|w<rsub|i>><around*|(|s<rsub|j>|)>\<cdot\>a<rsub|j>|]><rsup|2>.
  </equation*>

  Assume now that we have trained a deterministic policy
  <math|\<mu\><rsub|\<theta\>>> using DDGP. The idea is to slightly perturb
  the recommended action <math|\<mu\><rsub|\<theta\>><around*|(|s<rsub|t>|)>>
  at every step <math|t> and state <math|s<rsub|t>> to obtain a new action
  <math|a<rsub|t>> which fulfills the immediate constraints:
  <math|c<rsub|i><around*|(|s<rsub|t>,a<rsub|t>|)>\<leqslant\>C<rsub|i>>.
  This is a simple constrained convex optimisation problem

  <\equation*>
    a<rsub|t>=argmin <tfrac|1|2>*<around*|\<\|\|\>|a-\<mu\><rsub|\<theta\>><around*|(|s<rsub|t>|)>|\<\|\|\>><rsup|2><text|<space|1em>s.t.<space|1em>>c<rsub|i><around*|(|s<rsub|t>,a<rsub|t>|)>\<leqslant\>C<rsub|i><text|
    for all >i\<in\><around*|[|K|]>,
  </equation*>

  where we use the approximation

  <\equation*>
    c<rsub|i><around*|(|s<rsub|t>,a<rsub|t>|)>\<simeq\><wide|c|\<bar\>><rsub|i><around*|(|s<rsub|t>|)>+g<around*|(|s<rsub|t>;w<rsup|\<star\>><rsub|i>|)>\<cdot\>a<rsub|t>.
  </equation*>

  Because the constraints are affine and the problem convex, optimality is
  characterised by the Karush-Kuhn-Tucker conditions. Under the assumption
  that only one constraint is active at each timestep, it is possible to
  compute a closed analytical solution:

  <\equation*>
    a<rsup|\<star\>>=\<mu\><rsub|\<theta\>><around*|(|s|)>-\<lambda\><rsup|\<star\>><rsub|i<rsup|\<star\>>>*g<around*|(|s;w<rsup|\<star\>><rsub|i>|)>
  </equation*>

  where

  \;

  Note that this is done during training too.

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
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|2|?>>
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