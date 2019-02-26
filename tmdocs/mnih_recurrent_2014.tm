<TeXmacs|1.99.9>

<style|<tuple|generic|paperwhy|old-spacing>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Recurrent models of visual
  attention>|<doc-author|<author-data|<author-name|Mnih,
  Volodymir>>>|<doc-author|<author-data|<author-name|Hees,
  Nicolas>>>|<doc-author|<author-data|<author-name|Graves,
  Alex>>>|<doc-author|<author-data|<author-name|Kavukcuoglu,
  Koray>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|image classification|policy gradient|attention|reinforcement
  learning|POMDP|RNN>

  <strong|tl;dr:> Training a network to classify images (with a single label)
  is modeled as a sequential decision problem where actions are salient
  locations in the image and tentative labels. The state (full image) is
  partially observed through a fixed size subimage around each location. The
  policy takes the full history into account compressed into a hidden vector
  via an RNN. REINFORCE is used to compute the policy gradient.

  <hrule>

  Although the paper targets several applications, to fix ideas, say we want
  to classify images with one label. These can vary in size but the number of
  parameters of the model will not change. Taking inspiration from how humans
  process images, the proposed model iteratively selects points in the image
  and focuses on local patches around them at different resolutions. The
  problem of choosing the locations and related classification labels is cast
  as a reinforcement learning problem.

  <subsection|Attention as a sequential decision problem>

  One begins by <strong|fixing one image> <math|x> and selecting a number
  <math|T> of timesteps. At each timestep <math|t=1,\<ldots\>,T>:

  <\itemize-dot>
    <item>We are in some (observed) <dfn|state>
    <math|s<rsub|t>\<in\>\<cal-S\>>: it consists of a <dfn|location>
    <math|l<rsub|t>> in the image (a pair of coordinates), and a
    corresponding <dfn|glimpse> <math|x<rsub|t>> of <math|x>. This glimpse is
    a concatenation of multiple subimages of <math|x> taken at different
    resolutions, centered at location <math|l<rsub|t>>, then resampled to the
    same size. How many (<math|k>), at what resolutions
    (<math|\<rho\><rsub|1>,\<ldots\>,\<rho\><rsub|k>>) and to what fixed size
    (<math|w>) they are resampled are all hyperparameters.<\footnote>
      It would be nice to try letting a CNN capture the relevant features for
      us, instead of fixing the resolutions. I'm sure this has been tried
      since 2014.
    </footnote> The set of <em|all past states> is the <dfn|history>:
    <math|s<rsub|1:t-1>\<assign\><around*|{|s<rsub|1>,\<ldots\>,s<rsub|t-1>|}>>.

    <item>We take an <dfn|action> <math|a<rsub|t>=<around*|(|l<rsub|t>,y<rsub|t>|)>\<in\>\<cal-A\>>,
    with the new location <math|l<rsub|t>> in the (same) image <math|x> to
    look at and the current guess <math|y<rsub|t>> as to what the label for
    <math|x> is. In the typical way for image classification with neural
    networks, <math|y<rsub|t>> is a vector of \Pprobabilities\Q coming from a
    softmax layer. Analogously, the location <math|l<rsub|t>> is sampled from
    a distribution parametrized by the last layer of a network. The actions
    are taken according to a <dfn|policy>
    <math|\<pi\><rsup|t><rsub|\<theta\>>:\<cal-S\><rsup|t>\<rightarrow\>\<cal-P\><around*|(|\<cal-A\>|)>>,
    with <math|S<rsup|t>=S\<times\><above|\<cdots\>|t-1>\<times\>S>, and
    <math|\<cal-P\><around*|(|\<cal-A\>|)>> the set of all probabilty
    measures over <math|\<cal-A\>>. The policy is implemented as a neural
    network, where <math|\<theta\>> represents all internal parameters. The
    crucial point in the paper is that <em|the network takes the whole
    history as input>, compressed into a hidden state vector, i.e. the policy
    will be implemented with a recurrent network. Because parameters are
    shared across all timesteps, we drop the superindex <math|t> and denote
    its output at timestep <math|t> by <math|\<pi\><rsub|\<theta\>><around*|(|a<rsub|t>\|s<rsub|1:t>|)>>.

    <item>We obtain a scalar <dfn|reward>
    <math|r<rsub|t>\<in\><around*|{|0,1|}>>. Actually, the reward will be 0
    at all timesteps but the last <around*|(|<math|T>|)>, where it can be
    either 0 if <math|y<rsub|T>> predicts the wrong class or 1 if it is the
    right one.<\footnote>
      I wonder: wouldn't it make more sense to let
      <math|r<rsub|t>\<in\><around*|[|0,1|]>> instead using the cross entropy
      to the 1-hot vector encoding the correct class?
    </footnote>
  </itemize-dot>

  <big-figure|<image|../static/img/mnih_recurrent_2014-mine-1.jpg|0.6par|||>|The
  model at timestep <math|t>. >

  Note that the policy <math|\<pi\><rsub|\<theta\>><around*|(|a<rsub|t>\|s<rsub|1:t>|)>>
  has two \Pheads\Q, a labeling network <math|f<rsub|y>>, outputting a
  probability of the current glimpse belonging to each class and a location
  network <math|f<rsub|l>>. Only the output of the latter directly influences
  the next state. This is important when computing the distribution over
  trajectories <math|\<tau\>=<around*|(|s<rsub|1>,a<rsub|1>,\<ldots\>,s<rsub|T>,a<rsub|T>|)>>
  induced by the policy:

  <\equation*>
    p<rsub|\<theta\>><around|(|\<tau\>|)>\<assign\>p<around|(|s<rsub|1>|)>*<big|prod><rsub|t=1><rsup|T>\<pi\><rsub|\<theta\>><around|(|a<rsub|t>\<mid\>s<rsub|t>|)>*p<around|(|s<rsub|t+1>\<mid\>s<rsub|t>,a<rsub|t>|)>=p<around|(|s<rsub|1>|)>*<big|prod><rsub|t=1><rsup|T>p<around|(|l<rsub|t>\<mid\>f<rsub|l><around*|(|s<rsub|t>;\<theta\>|)>|)>*p<around|(|s<rsub|t+1>\<mid\>s<rsub|t>,l<rsub|t>|)>.
  </equation*>

  The goal is to maximise the total expected reward

  <\equation*>
    J<around*|(|\<theta\>|)>\<assign\>\<bbb-E\><rsub|\<tau\>\<sim\>p<rsub|\<theta\>><around*|(|\<tau\>|)>><rsub|><around|[|<with|math-display|false|<big|sum><rsub|t=1><rsup|T>>r<around*|(|s<rsub|t>,a<rsub|t>|)>|]>.
  </equation*>

  The algorithm used is basically the policy gradient method with the
  REINFORCE rule:<\footnote>
    See e.g. <cite|sutton_reinforcement_2018>, <math|>Ÿ13.3.
  </footnote>

  <\algorithm>
    <\enumerate>
      <item>Initialise <math|\<pi\><rsub|\<theta\>>> with some random set of
      parameters.

      <item>For <math|n=1\<ldots\>N>, pick some input image <math|x<rsub|n>>
      with label <math|y<rsub|n>>.

      <item>Sample some random initial location <math|l<rsub|0>>.

      <item>Run the policy (the recurrent network)
      <math|\<pi\><rsub|\<theta\>>> for <math|T> timesteps, creating new
      locations <math|l<rsub|t>> and labels <math|y<rsub|t>>. At the end
      collect the reward <math|r<rsub|T>\<in\><around*|{|0,1|}>>.

      <item>Compute the gradient of the reward
      <math|\<nabla\><rsub|\<theta\>> J<around*|(|\<theta\><rsub|n>|)>>.

      <item>Update <math|\<theta\><rsub|n+1>\<leftarrow\>\<theta\><rsub|n>+\<alpha\><rsub|n>*\<nabla\><rsub|\<theta\>>
      J<rsub|\<theta\>><around*|(|\<theta\><rsub|n>|)>>.
    </enumerate>
  </algorithm>

  The difficulty lies in step <math|5> because the reward is an expectation
  over trajectories whose gradient cannot be analitically computed. One
  solution is to rewrite the gradient of the expectation as another
  expectation using a simple but clever substitution:

  <\equation*>
    \<nabla\><rsub|\<theta\>> J<around*|(|\<theta\>|)>=<big|int>\<nabla\><rsub|\<theta\>>
    p<rsub|\<theta\>><around*|(|\<tau\>|)>*r<around*|(|\<tau\>|)>*\<mathd\>\<tau\>=<big|int>p<rsub|\<theta\>><around*|(|\<tau\>|)>*<frac|\<nabla\><rsub|\<theta\>>
    p<rsub|\<theta\>><around*|(|\<tau\>|)>|p<rsub|\<theta\>><around*|(|\<tau\>|)>>*r<around*|(|\<tau\>|)>*\<mathd\>\<tau\>=<big|int>p<rsub|\<theta\>><around*|(|\<tau\>|)>*\<nabla\><rsub|\<theta\>>
    log p<rsub|\<theta\>><around*|(|\<tau\>|)>*r<around*|(|\<tau\>|)>*\<mathd\>\<tau\>,
  </equation*>

  and this is

  <\equation*>
    \<nabla\><rsub|\<theta\>> J<around*|(|\<theta\>|)>=\<bbb-E\><rsub|\<tau\>\<sim\>p<rsub|\<theta\>><around*|(|\<tau\>|)>><rsub|><around|[|\<nabla\><rsub|\<theta\>>
    log p<rsub|\<theta\>><around*|(|\<tau\>|)>*r<around*|(|\<tau\>|)>|]>
  </equation*>

  In order to compute this integral we can now use Monte-Carlo sampling:

  <\equation*>
    \<nabla\><rsub|\<theta\>> J<around*|(|\<theta\>|)>\<approx\><frac|1|M>*<big|sum><rsub|m=1><rsup|M>\<nabla\><rsub|\<theta\>>
    log p<rsub|\<theta\>><around*|(|\<tau\>|)>*r<around*|(|\<tau\>|)>,
  </equation*>

  and after rewriting <math|log p<rsub|\<theta\>><around*|(|\<tau\>|)>> as a
  sum of logarithms and discarding the terms which do not depend on
  <math|\<theta\>> we obtain:

  <\equation*>
    \<nabla\><rsub|\<theta\>> J<around*|(|\<theta\>|)>\<approx\><frac|1|M>*<big|sum><rsub|m=1><rsup|M><big|sum><rsub|t=1><rsup|T>\<nabla\><rsub|\<theta\>>
    log \<pi\><rsub|\<theta\>><around*|(|a<rsup|m><rsub|t>\|s<rsup|m><rsub|1:t>|)>*r<rsup|m>,
  </equation*>

  where <math|r<rsup|m>=r<rsub|T><rsup|m>> is the final reward (recall that
  in this application <math|r<rsub|t>=0> for all <math|t\<less\>T>). In order
  to reduce the variance of this estimator it is standard to subtract a
  baseline estimate \ <math|b=\<bbb-E\><rsub|\<pi\><rsub|\<theta\>>><around*|[|r<rsub|T>|]>>
  of the expected reward, thus arriving at the expression

  <\equation*>
    \<nabla\><rsub|\<theta\>> J<around*|(|\<theta\>|)>\<approx\><frac|1|M>*<big|sum><rsub|m=1><rsup|M><big|sum><rsub|t=1><rsup|T>\<nabla\><rsub|\<theta\>>
    log \<pi\><rsub|\<theta\>><around*|(|a<rsup|m><rsub|t>\|s<rsup|m><rsub|1:t>|)>*<around*|(|r<rsup|m>-b|)>.
  </equation*>

  There is a vast literature on the Monte-Carlo approximation for policy
  gradients, as well as techniques for variance reduction.<\footnote>
    Again, see <cite|sutton_reinforcement_2018>.
  </footnote>

  <subsection|Hybrid learning>

  Because in classification problems the labels are known at training time,
  one can provide the network with a better signal than just the reward at
  the end of all the process. In this case the authors

  <\quotation>
    optimize the cross entropy loss to train the [labeling] network
    <math|f<rsub|y>> and backpropagate the gradients through the core and
    glimpse networks. The location network <math|f<rsub|l>> is always trained
    with REINFORCE.
  </quotation>

  <subsection|Results for image classification>

  An image is worth a thousand words:

  <\render-big-figure|||<image|../static/img/mnih_recurrent_2014-fig2.jpg|1par|||>>
    \;
  </render-big-figure>

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|1>
      <bibitem*|1><label|bib-sutton_reinforcement_2018>Richard<nbsp>S.<nbsp>Sutton<localize|
      and >Andrew<nbsp>G.<nbsp>Barto. <newblock><with|font-shape|italic|Reinforcement
      Learning: An Introduction>. <newblock>MIT Press, 2nd (in
      progress)<localize| edition>, jan 2018. <newblock>Citecount:
      28254.<newblock>
    </bib-list>
  </bibliography>
</body>

<\initial>
  <\collection>
    <associate|info-flag|detailed>
    <associate|preamble|false>
    <associate|save-aux|true>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|2|?>>
    <associate|auto-4|<tuple|3|?>>
    <associate|auto-5|<tuple|3|?>>
    <associate|auto-6|<tuple|3|?>>
    <associate|bib-sutton_reinforcement_2018|<tuple|1|?>>
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
      sutton_reinforcement_2018

      sutton_reinforcement_2018
    </associate>
    <\associate|figure>
      <tuple|normal|<surround|<hidden-binding|<tuple>|1>||The model at
      timestep <with|mode|<quote|math>|t>. >|<pageref|auto-2>>
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Attention as a sequential
      decision problem <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|1tab>|2<space|2spc>Hybrid learning
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|3<space|2spc>Results for image
      classification <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>