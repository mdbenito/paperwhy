<TeXmacs|1.99.5>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Deep Residual Learning for image
  recognition>|<doc-author|<author-data|<author-name|he,
  kaiming>>>|<doc-author|<author-data|<author-name|Zhang,
  Xiangyu>>>|<doc-author|<author-data|<author-name|Ren,
  Shaoqing>>>|<doc-author|<author-data|<author-name|Sun,
  Jian>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|deep learning|residual networks|image recognition>

  <strong|tl;dr:> Deeper models for visual tasks have been proven to greatly
  outperform shallow ones, but after some point simply adding more layers
  decreases performance even if the networks are in principle more
  expressive. Adding skip-connections overcomes these difficulties and vastly
  improves performance, while keeping the number of parameters under control.

  <hrule>

  This post is a prequel to previous ones where we went over work studiying
  the theoretical properties of <strong|Residual Networks>, introduced in the
  current paper. In <cite|lin_why_2016> we learnt that deeper networks are
  very good approximators of compositional functions at the expense of energy
  landscapes with poorer local optima. Later, in <cite|hardt_identity_2016>
  we saw that (nonlinear) perturbations of the identity as models are easy to
  optimize and are able to learn <math|r> classes with
  <math|\<cal-O\><around*|(|n*log n+r<rsup|2>|)>> parameters, whereas
  <cite|bartlett_representational_2017> discusses why Lipschitz functions can
  (in principle) be very well approximated by resnets. \ Changing the
  hypothesis space to perturbations of the identity for easier optimization
  yields vastly improved results. Be sure to check those papers later.

  <section*|Deeper is harder>

  Vanishing gradients used to be a huge issue with deeper networks, which has
  partly been addressed by <strong|normalized initialization> and
  <strong|batch normalization>.<\footnote>
    <cite|ioffe_batch_2015>.
  </footnote> However, even if they then converge to some optima, networks
  with lots of layers show degraded performance. Notably, the problem is not
  overfitting since they can exhibit <em|poorer training error>. But since
  just stacking more layers can only increase the expressiveness of the class
  of functions which can be computed, this points to an optimization issue.

  The authors suggest then the addition of <dfn|skip connections> among
  layers with the idea of letting the network preserve relevant features from
  across layers: in the case that an identity is optimal, it's just easier to
  use these connections than to learn weights through the nonlinearities.

  <section*|Nothing new under the sun>

  As is (almost) always the case, the idea of propagating residual
  information is present in many branches of mathematics. The authors mention
  applications in vector quantization, and more excitingly multigrid methods
  for PDEs, where each subproblem computes the residual between solutions at
  each scale. But shortcut connections where also present in the beginnings
  of neural networks or more recently with highway networks with gated
  shortcuts (i.e. with trainable additional weights able to shut them off
  entirely).

  <section*|Network architecture and implementations>

  <big-figure|<image|../static/img/he_deep_2015-fig2.jpg|0.5par|||>|The basic
  building block of a Residual Network.>

  Assuming that we augment data in one dimension to include biases into the
  network's weight matrices, we can compactly denote the building block of
  the figure as

  <\equation*>
    \<b-y\>=\<cal-F\><around*|(|\<b-x\>,<around*|{|W<rsub|i>|}>|)>+\<b-x\>,
  </equation*>

  where

  <\equation*>
    \<cal-F\><around*|(|\<b-x\>,<around*|{|W<rsub|i>|}>|)>=W<rsub|i+1>*\<sigma\><around*|(|W<rsub|i>*\<b-x\>|)>.
  </equation*>

  Note that the shortcut <math|\<cal-F\>+\<b-x\>> doesn't add any parameters
  to the model, which is important not only because of the obvious reason,
  but also when comparing performance to that of other networks without skip
  connections.<\footnote>
    One minor modification is required in case <math|dim \<b-x\>\<neq\>dim
    \<cal-F\>>, namely using some projection matrix to change the dimension:
    <math|\<b-y\>=\<cal-F\><around*|(|\<b-x\>,<around*|{|W<rsub|i>|}>|)>+W<rsub|s>*\<b-x\>>.
  </footnote> Note also that having at least two layers with one nonlinearity
  is essential for the skip connection to make sense, since otherwise the
  building block reduces to a linear mapping.

  The reasoning behind adding the identity was already mentioned above: the
  degrading performance of models which are actually more expressive means
  that they have trouble approximating the identity (since that would be a
  way of \Pdiscarding\Q unnecessary layers and falling back to the simpler
  model). It was hoped that by adding the identity this would be mitigated.
  In fact it effectively changes the hypothesis space to concatenated
  perturbations of the identity, which are empirically seen to be small
  because the weights <math|W<rsub|i>> are. And we now know thanks to later
  work that this hypothesis space has very good properties both in terms of
  approximation ability and optimization properties.

  The gist of all the claims made until now can be seen in the very first
  example of the paper, where the authors consider three models: first a
  VGG-19 network, second a plain (no residual connections) network of 34
  layers inspired by VGG-19's architecture, and thirdthe second model with
  skip connections. Recall that the latter maintains the number of parameters
  wrt. the second model.

  The first comparison between 18 and 34 layers display the aforementioned
  phenomenon of lower performance but no vanishing gradients.

  <big-figure|<image|../static/img/he_deep_2015-fig4a.jpg|0.7par|||>|Adding
  more layers makes optimization harder>

  The authors conjecture that

  <\quotation>
    this optimization difficulty is unlikely to be caused by vanishing
    gradients. These plain networks are trained with [Batch Normalization],
    which ensures forward propagated signals to have non-zero variances. We
    also verify that the backward propagated gradients exhibit healthy norms
    with BN. So neither forward nor backward signals vanish.
  </quotation>

  However, skip connections fix the issue and the interpretation already
  explained is put forth. Recall again that there is now theoretical work
  supporting some of the claims.

  <big-figure|<image|../static/img/he_deep_2015-fig4b.jpg|0.7par|||>|Adding
  skip connections vastly improves performance>

  There is also an interesting point with plain networks which are not as
  deep: adding skip connections to an 18 layer network doesn't increase
  performance but it does decrease the time to convergence. Again the
  optimization landscape is more benign in the new hypothesis space.

  Finally the authors report great results with CIFAR-10 and COCO detection
  and localization which I won't repeat here because the paper has \Pall\Q
  the details (modulo any actual implementation details ;-).

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|4>
      <bibitem*|1><label|bib-bartlett_representational_2017>Peter
      Bartlett.<newblock> Representational and optimization properties of
      Deep Residual Networks.<newblock> may 2017.<newblock> Citecount: 00000,
      2681 seconds.<newblock>

      <bibitem*|2><label|bib-hardt_identity_2016>Moritz Hardt<localize| and
      >Tengyu Ma.<newblock> Identity matters in Deep Learning.<newblock>
      <with|font-shape|italic|ArXiv:1611.04231 [cs, stat]>, nov
      2016.<newblock> Citecount: 00001, arXiv: 1611.04231.<newblock>

      <bibitem*|3><label|bib-ioffe_batch_2015>Sergey Ioffe<localize| and
      >Christian Szegedy.<newblock> Batch Normalization: Accelerating Deep
      Network Training by Reducing Internal Covariate Shift.<newblock>
      <with|font-shape|italic|ArXiv:1502.03167 [cs]>, <localize|page >11, feb
      2015.<newblock> Citecount: 01618 arXiv: 1502.03167.<newblock>

      <bibitem*|4><label|bib-lin_why_2016>Henry<nbsp>W.<nbsp>Lin<localize|
      and >Max Tegmark.<newblock> Why does deep and cheap learning work so
      well?<newblock> <with|font-shape|italic|ArXiv:1608.08225 [cond-mat,
      stat]>, <localize|page >17, aug 2016.<newblock> Citecount: 00019 arXiv:
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

<\attachments>
  <\collection>
    <\associate|bib-bibliography>
      <\db-entry|+HVwQQj5msr2GNU|article|lin_why_2016>
        <db-field|contributor|miguel>

        <db-field|modus|imported>

        <db-field|date|1505039342>
      <|db-entry>
        <db-field|author|Henry W. <name|Lin><name-sep>Max <name|Tegmark>>

        <db-field|title|Why does deep and cheap learning work so well?>

        <db-field|journal|arXiv:1608.08225 [cond-mat, stat]>

        <db-field|year|2016>

        <db-field|pages|17>

        <db-field|month|aug>

        <db-field|note|citecount: 00019 arXiv: 1608.08225>

        <db-field|url|<slink|http://arxiv.org/abs/1608.08225>>

        <db-field|abstract|We show how the success of deep learning depends
        not only on mathematics but also on physics: although well-known
        mathematical theorems guarantee that neural networks can approximate
        arbitrary functions well, the class of functions of practical
        interest can be approximated through "cheap learning" with
        exponentially fewer parameters than generic ones, because they have
        simplifying properties tracing back to the laws of physics. The
        exceptional simplicity of physics-based functions hinges on
        properties such as symmetry, locality, compositionality and
        polynomial log-probability, and we explore how these properties
        translate into exceptionally simple neural networks approximating
        both natural phenomena such as images and abstract representations
        thereof such as drawings. We further argue that when the statistical
        process generating the data is of a certain hierarchical form
        prevalent in physics and machine-learning, a deep neural network can
        be more efficient than a shallow one. We formalize these claims using
        information theory and discuss the relation to renormalization group
        procedures. We prove various "no-flattening theorems" showing when
        such efficient deep networks cannot be accurately approximated by
        shallow ones without efficiency loss: flattening even linear
        functions can be costly, and flattening polynomials is exponentially
        expensive; we use group theoretic techniques to show that n variables
        cannot be multiplied using fewer than 2\Bn neurons in a single hidden
        layer.>

        <db-field|language|en>

        <db-field|urldate|2016-10-26>

        <db-field|file|Lin and Tegmark - 2016 - Why does deep and cheap
        learning work so well.pdf:/Users/miguel/Library/Application
        Support/Zotero/Profiles/j358n6qi.default/zotero/storage/JD6HE5EB/Lin
        and Tegmark - 2016 - Why does deep and cheap learning work so
        well.pdf:application/pdf;tegmark-comments.tm:/Users/miguel/Library/Application
        Support/Zotero/Profiles/j358n6qi.default/zotero/storage/A8V6DEBS/tegmark-comments.tm:text/plain>
      </db-entry>

      <\db-entry|+HVwQQj5msr2GNv|article|hardt_identity_2016>
        <db-field|contributor|miguel>

        <db-field|modus|imported>

        <db-field|date|1505039343>
      <|db-entry>
        <db-field|author|Moritz <name|Hardt><name-sep>Tengyu <name|Ma>>

        <db-field|title|Identity matters in Deep Learning>

        <db-field|journal|arXiv:1611.04231 [cs, stat]>

        <db-field|year|2016>

        <db-field|month|nov>

        <db-field|note|citecount: 00001 arXiv: 1611.04231>

        <db-field|url|<slink|http://arxiv.org/abs/1611.04231>>

        <db-field|abstract|An emerging design principle in deep learning is
        that each layer of a deep artificial neural network should be able to
        easily express the identity transformation. This idea not only
        motivated various normalization techniques, such as \\emph{batch
        normalization}, but was also key to the immense success of
        \\emph{residual networks}. In this work, we put the principle of
        \\emph{identity parameterization} on a more solid theoretical footing
        alongside further empirical progress. We first give a strikingly
        simple proof that arbitrarily deep linear residual networks have no
        spurious local optima. The same result for linear feed-forward
        networks in their standard parameterization is substantially more
        delicate. Second, we show that residual networks with ReLu
        activations have universal finite-sample expressivity in the sense
        that the network can represent any function of its sample provided
        that the model has more parameters than the sample size. Directly
        inspired by our theory, we experiment with a radically simple
        residual architecture consisting of only residual convolutional
        layers and ReLu activations, but no batch normalization, dropout, or
        max pool. Our model improves significantly on previous
        all-convolutional networks on the CIFAR10, CIFAR100, and ImageNet
        classification benchmarks.>

        <db-field|file|Hardt and Ma - 2016 - Identity matters in Deep
        Learning.pdf:/Users/miguel/Library/Application
        Support/Zotero/Profiles/j358n6qi.default/zotero/storage/HIP545D2/Hardt
        and Ma - 2016 - Identity matters in Deep
        Learning.pdf:application/pdf>
      </db-entry>

      <\db-entry|+HVwQQj5msr2GNV|misc|bartlett_representational_2017>
        <db-field|contributor|miguel>

        <db-field|modus|imported>

        <db-field|date|1505039342>
      <|db-entry>
        <db-field|author|Peter <name|Bartlett>>

        <db-field|title|Representational and optimization properties of Deep
        Residual Networks>

        <db-field|month|may>

        <db-field|year|2017>

        <db-field|note|citecount: 00000 2681 seconds>

        <db-field|address|Simons Institute>

        <db-field|type|talk>

        <db-field|url|https://simons.berkeley.edu/talks/tba-1>

        <db-field|language|en>

        <db-field|urldate|2017-05-02>

        <db-field|file|Bartlett<rsub|2>017<rsub|R>epresentational and
        optimization properties of Deep Residual
        Networks.tm:/Users/miguel/Library/Application
        Support/Zotero/Profiles/j358n6qi.default/zotero/storage/URXCD347/Bartlett<rsub|2>017<rsub|R>epresentational
        and optimization properties of Deep Residual Networks.tm:text/plain>
      </db-entry>

      <\db-entry|+HVwQQj5msr2GNx|article|ioffe_batch_2015>
        <db-field|contributor|miguel>

        <db-field|modus|imported>

        <db-field|date|1505039343>
      <|db-entry>
        <db-field|author|Sergey <name|Ioffe><name-sep>Christian
        <name|Szegedy>>

        <db-field|title|Batch Normalization: Accelerating Deep Network
        Training by Reducing Internal Covariate Shift>

        <db-field|journal|arXiv:1502.03167 [cs]>

        <db-field|year|2015>

        <db-field|pages|11>

        <db-field|month|feb>

        <db-field|note|citecount: 01618 arXiv: 1502.03167>

        <db-field|shorttitle|Batch Normalization>

        <db-field|url|<slink|http://arxiv.org/abs/1502.03167>>

        <db-field|abstract|Training Deep Neural Networks is complicated by
        the fact that the distribution of each layer's inputs changes during
        training, as the parameters of the previous layers change. This slows
        down the training by requiring lower learning rates and careful
        parameter initialization, and makes it notoriously hard to train
        models with saturating nonlinearities. We refer to this phenomenon as
        internal covariate shift, and address the problem by normalizing
        layer inputs. Our method draws its strength from making normalization
        a part of the model architecture and performing the normalization for
        each training mini-batch. Batch Normalization allows us to use much
        higher learning rates and be less careful about initialization. It
        also acts as a regularizer, in some cases eliminating the need for
        Dropout. Applied to a state-of-the-art image classification model,
        Batch Normalization achieves the same accuracy with 14 times fewer
        training steps, and beats the original model by a significant margin.
        Using an ensemble of batch-normalized networks, we improve upon the
        best published result on ImageNet classification: reaching 4.9% top-5
        validation error (and 4.8% test error), exceeding the accuracy of
        human raters.>

        <db-field|urldate|2017-04-18>

        <db-field|file|Ioffe and Szegedy - 2015 - Batch Normalization
        Accelerating Deep Network Tra.pdf:/Users/miguel/Library/Application
        Support/Zotero/Profiles/j358n6qi.default/zotero/storage/CDCNFX6S/Ioffe
        and Szegedy - 2015 - Batch Normalization Accelerating Deep Network
        Tra.pdf:application/pdf>
      </db-entry>
    </associate>
  </collection>
</attachments>

<\references>
  <\collection>
    <associate|auto-1|<tuple|?|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|1|?>>
    <associate|auto-4|<tuple|1|?>>
    <associate|auto-5|<tuple|2|?>>
    <associate|auto-6|<tuple|3|?>>
    <associate|auto-7|<tuple|3|?>>
    <associate|bib-bartlett_representational_2017|<tuple|1|?>>
    <associate|bib-hardt_identity_2016|<tuple|2|?>>
    <associate|bib-ioffe_batch_2015|<tuple|3|?>>
    <associate|bib-lin_why_2016|<tuple|4|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnote-2|<tuple|2|?>>
    <associate|footnr-1|<tuple|1|?>>
    <associate|footnr-2|<tuple|2|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      lin_why_2016

      hardt_identity_2016

      bartlett_representational_2017

      ioffe_batch_2015
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>