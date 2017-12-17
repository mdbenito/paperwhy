<TeXmacs|1.99.5>

<style|<tuple|generic|paperwhy>>

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

  <strong|tl;dr:> Training a network to classify images (single label) is
  seen as a reinforcement learning problem. Actions are salient locations in
  the image and tentative labels. The state (full image) is partially
  observed through a fixed size subimage around each location. The policy
  takes the full history into account compressed into a hidden vector via an
  RNN. Policy gradient is used in REINFORCE.

  <hrule>

  We want to classify images with one label. These can vary in size but the
  number of parameters of the model will not change. Taking inspiration from
  how we process images, the model iteratively focuses on local patches of
  image at different resolutions. The problem of choosing the locations and
  related labels is cast as a sequential decision problem, in particular a
  <abbr|POMPD>.<\footnote>
    Partially Observable Markov Decission Process. For some background on
    these <todo|we recommend ...>
  </footnote>

  One begins by <strong|fixing one image> <math|x> and selecting a number
  <math|T> of timesteps.

  At each timestep <math|t=1,\<ldots\>,T>:

  <\itemize-dot>
    <item>We have a <dfn|state>: a <dfn|location> <math|l<rsub|t>> in the
    image (a pair of coordinates), and a corresponding \P<dfn|glimpse>\Q
    <math|x<rsub|t>> of <math|x>. This glimpse is a concatenation of multiple
    subimages of <math|x> taken at different resolutions, centered at
    <math|l<rsub|t>>, then resampled to the same size. How many (<math|k>),
    at what resolutions (<math|\<rho\><rsub|1>,\<ldots\>,\<rho\><rsub|k>>)
    and to what fixed size (<math|w>) they are resampled are all
    hyperparameters.<\footnote>
      It would be nice to try letting a CNN capture the relevant features for
      us, instead of fixing the resolutions. I'm sure this has been tried
      since 2014.
    </footnote> The set of <em|all past states> is the <dfn|history>:
    <math|s<rsub|1:t-1>\<assign\><around*|{|s<rsub|1>,\<ldots\>,s<rsub|t-1>|}>>.

    <item>We take an <dfn|action>: a tuple
    <math|<around*|(|l<rsub|t>,a<rsub|t>|)>> with the new location in the
    (same) image <math|x> to look at, and the current guess <math|a<rsub|t>>
    as to what the label for <math|x> is. In the typical way for image
    classification with neural networks, <math|a<rsub|t>> is a vector of
    \Pprobabilities\Q coming from a softmax layer. Analogously, the location
    <math|l<rsub|t>> is sampled from a distribution parametrized by the last
    layer of a network. The actions are taken according to a <dfn|policy>: a
    neural network having as input the state. The crucial point in the paper
    is that <em|it will take the whole history as input>, compressed into a
    hidden state vector, i.e. the policy will be implemented with a recurrent
    network. We denote its output at timestep <math|t> by
    <math|\<pi\><rsub|\<theta\>><around*|(|l<rsub|t>,a<rsub|t>\|s<rsub|1:t>|)>>,
    where <math|\<theta\>> represents all internal parameters.

    <item>We obtain a <dfn|reward>: a scalar
    <math|r<rsub|t>\<in\><around*|{|0,1|}>>. Actually, the reward will be 0
    at all timesteps but the last <around*|(|<math|T>|)>, where it can be
    either 0 if <math|a<rsub|T>> predicts the wrong class or 1 if it is the
    right one.<\footnote>
      I wonder: wouldn't it make more sense to let
      <math|r<rsub|t>\<in\><around*|[|0,1|]>> instead using the cross entropy
      to the 1-hot vector encoding the correct class.
    </footnote>
  </itemize-dot>

  <big-figure|<image|../static/img/mnih_recurrent_2014-mine-1.jpg|0.6par|||>|The
  model at timestep <math|t>. >

  The goal is to maximise the total expected reward across all images.

  <\equation*>
    \<bbb-E\><around|[|<with|math-display|false|<big|sum><rsub|t=1><rsup|T>>r<around*|(|s<rsub|t>,a<rsub|t>|)>|]>
  </equation*>

  <todo|We detail later wrt. what the expectations are taken.> The algorithm
  is basically REINFORCE<\footnote>
    See Williams92, but also, Lecture 3/4? of CS224-112 @ UCB.
  </footnote>:

  <\algorithm>
    <\enumerate>
      <item>Initialise <math|\<pi\><rsub|\<theta\>>> with some random set of
      parameters.

      <item>For <math|n=1\<ldots\>N>, fix some input image <math|x<rsub|n>>
      with label <math|y<rsub|n>>.

      <item>Sample some random initial location <math|l<rsub|0>>.

      <item>Run the policy (the recurrent network)
      <math|\<pi\><rsub|\<theta\>>> for <math|T> timesteps, creating new
      locations <math|l<rsub|t>> and actions <math|a<rsub|t>>. At the end
      collect the reward <math|R>.

      <item>Compute the gradient of the loss
      <math|\<nabla\><rsub|\<theta\>>J<around*|(|\<theta\><rsub|n>|)>>.

      <item>Update <math|\<theta\><rsub|n+1>\<leftarrow\>\<theta\><rsub|n>+\<alpha\><rsub|n>*\<nabla\><rsub|\<theta\>>J<rsub|\<theta\>><around*|(|\<theta\><rsub|n>|)>>.
    </enumerate>
  </algorithm>

  Now, the problem is with step <math|5>. The actual loss is an expectation
  over all possible policies, and this is hard to approximate. One
  possibility is that of REINFORCE, \ which is based on a single sample among
  all possible policies:

  <\equation*>
    \<nabla\><rsub|\<theta\>>J<around*|(|\<theta\>|)>\<approx\><frac|1|M>*<big|sum><rsub|m=1><rsup|M><big|sum><rsub|t=1><rsup|T>\<nabla\><rsub|\<theta\>>
    log \<pi\><rsub|\<theta\>><around*|(|<around*|(|l<rsup|m><rsub|t>,a<rsup|m><rsub|t>\|s<rsup|m><rsub|1:t>|)>|)>*<around*|(|R<rsup|m>-b<rsup|m>|)>,
  </equation*>

  where <math|b> is an estimate of the expected reward whose purpose is to
  reduce the variance of this estimator for the gradient. Note that we make
  <math|M> runs of the policy also to reduce the variance. Unpacking this
  definition is actually quite some work and it is perhaps best to refer to
  some good sources.<\footnote>
    See...
  </footnote>

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
    <associate|auto-2|<tuple|5|?>>
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
    <\associate|figure>
      <tuple|normal|The model at timestep <with|mode|<quote|math>|t>.
      |<pageref|auto-1>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>