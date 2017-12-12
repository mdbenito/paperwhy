<TeXmacs|1.99.5>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|What Uncertainties Do We Need in Bayesian Deep
  Learning for Computer Vision?>|<doc-author|<author-data|<author-name|Kendall,
  Alex>>>|<doc-author|<author-data|<author-name|Gal,
  Yarin>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|bayesian|deep learning|computer vision>

  <strong|tl;dr:>

  <hrule>

  <cite|kendall_multi-task_2017> builds upon this.
  <cite|kendall_geometric_2017> too.

  Possibly one of the greatest drawbacks for many applications of deep
  learning techniques is the lack of error bars. Typically, softmax outputs
  are used and although they can be interpreted as probability distributions,
  they are artificial and the lack of a true probabilistic model renders them
  useless for the estimation of error other than test error and uncertainty.

  Today's paper is a prequel to an application we already covered.<\footnote>
    <cite|kendall_geometric_2017>.
  </footnote>

  <\quotation>
    <dfn|Aleatoric uncertainty> captures noise inherent in the observations.
    This could be for example sensor noise or motion noise, resulting in
    uncertainty which cannot be reduced even if more data were to be
    collected. On the other hand, <dfn|epistemic uncertainty> accounts for
    uncertainty in the model parameters
  </quotation>

  <\quotation>
    We present a unified Bayesian deep learning framework which allows us to
    learn mappings from input data to aleatoric uncertainty and compose these
    together with epistemic uncertainty approximations.
  </quotation>

  Epistemic uncertainty is what is captured by classical Bayesian Neural
  Networks, in the classical Bayesian approach. Given data
  <math|<around*|(|X,Y|)>>, a random network output
  <math|f<rsub|W><around*|(|X|)>> with parameters <math|W> and a prior
  distribution <math|p<around*|(|W|)>> over them, the goal is to compute the
  posterior <math|p<around*|(|W\|X,Y|)>=p<around*|(|Y\|X,W|)>*p<around*|(|W|)>/p<around*|(|Y\|X|)>>

  \;

  <\quotation>
    <\enumerate>
      <item>We capture an accurate understanding of aleatoric and
      epistemicuncertainties,inparticular with a novel approach for
      classification,\ 

      <item>We improve model performance by 1 \<minus\> 3% over non-Bayesian
      baselines by reducing the effect of noisy data with the implied
      attenuation obtained from explicitly representing aleatoric
      uncertainty,\ 

      <item>We study the trade-offs between modeling aleatoric or epistemic
      uncertainty by character- izing the properties of each uncertainty and
      comparing model performance and inference time.\ 
    </enumerate>
  </quotation>

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
    <associate|auto-1|<tuple|3|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnr-1|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      kendall_multi-task_2017

      kendall_geometric_2017

      kendall_geometric_2017
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>