<TeXmacs|1.99.5>

<style|<tuple|generic|paperwhy>>

<\body>
  <\hide-preamble>
    \;

    <assign|by-text|<macro|>>
  </hide-preamble>

  <\doc-data|<doc-title|Geometric loss functions for camera pose regression
  with deep learning>|<doc-author|<author-data|<author-name|Kendall,
  Alex>>>|<doc-author|<author-data|<author-name|Cipolla,
  Roberto>>>|<doc-running-author|Miguel de Benito Delgado>>
    \;
  </doc-data|<doc-running-author|Miguel de Benito Delgado>>

  <tags|computer vision|camera pose|multi-task learning>

  <strong|tl;dr:> A Bayesian point of view allows simultaneously training for
  two different losses without hyperparameters. For camera pose estimation,
  geometric reprojection loss can be used to fine tune results.

  <hrule>

  Today's paper application is interesting enough by itself, but perhaps more
  so how it uses Bayesian ideas to train jointly for multiple losses,
  balancing them but without adding hyperparameters.<\footnote>
    The basic idea is explained in <cite|kendall_what_2017>.
  </footnote> Since I find this to be the take-away message, instead of going
  through a tedious literature review of the field of application, which the
  paper does much better, let's just dive in with a couple of quick comments:

  <section*|Goal>

  Given some photograph, we wish to recover position and orientation of the
  camera (6 dofs). Some application examples are:

  <\itemize-dot>
    <item>Overlay a 3D model onto an image, e.g. for <em|augmented reality>.

    <item>Locate pictures in maps, e.g. Google Maps' feature placing users'
    pictures automatically.

    <item>Estimate full pose of autonomous vehicles with inaccurate
    positioning: the idea is to fuse GPS and / or inertial sensor data with
    visual information for full 6 dof estimation in an \Pabsolute\Q frame of
    reference. Note that this is easier in a structured (road) environment
    than e.g. in the air or sea.<\footnote>
      Note too that LIDAR is not an option for ships due to the long ranges
      required and radar has too low resolution.
    </footnote>
  </itemize-dot>

  Some examples of what the <strong|training data> can be (links to datasets
  in the paper)

  <\itemize-dot>
    <item>Flat images and 3D points: costly because it requires careful
    crafting with LIDAR-like solutions or must be approximated with things
    like <hlink|structure from motion|https://en.wikipedia.org/wiki/Structure_from_motion>.

    <item>RGB-D sensor data: best used in indoor applications because depth
    info in RGB-D degrades with distance (quadratically<\footnote>
      <cite|walch_imagebased_2016>.
    </footnote>).

    <item>Stereoscopic images: this option seems best for outdoor
    applications: standard techniques can provide relative position and
    rotation modulo constant factors in the distance to objects: fusing this
    with sensor data should work well (?).
  </itemize-dot>

  <section*|Architecture>

  As is common, the authors use a pretrained CNN model for image
  classification<\footnote>
    <cite|szegedy_going_2015>.
  </footnote> with the top layer and softmax output removed. In their stead
  two independent fully connected layers perform regression to predict the
  position vector <math|\<b-p\>\<in\>\<bbb-R\><rsup|3>> and rotation
  quaternion <math|\<b-q\>\<in\>\<bbb-R\><rsup|4>>. Naturally, one wishes to
  train for <math|\<b-p\>,\<b-q\>> jointly since it is to be expected that,
  conditioned on an image, they are strongly related.

  <\quotation>
    The model learns a better representation for pose when supervised with
    both translation and orientation labels. (...) branching the network
    lower down into two separate components (...) was less effective (...):
    separating into distinct position and orientation features denies each
    the information necessary to factor out orientation from position, or
    vice versa.
  </quotation>

  <big-figure|<image|../static/img/kendall_geometric_2017-fig1-fake.jpg|755px|228px||>|Architecture.
  Picture shamelessly cannibalized without permission from
  <cite|walch_imagebased_2016>.>

  <section*|Improving the loss>

  The simplest loss function combining training for both quantities, is a
  simple linear combination including a hyperparameter <math|\<beta\>>
  interpolating the losses at each of the final regression layers:

  <\equation*>
    \<cal-L\>=\<cal-L\><rsub|p>+\<beta\>\<cal-L\><rsub|q>,
  </equation*>

  where both <math|\<cal-L\><rsub|p>> and <math|\<cal-L\><rsub|q>> are simple
  <math|L<rsup|2>> losses, modulo a little technicality with
  quaternions.<\footnote>
    For each rotation, there are two quaternions <math|\<b-q\>> and
    <math|-\<b-q\>> representing it. For this reason the authors constrain
    the optimization to happen on a half-space.
  </footnote> However, this extraneous <math|\<beta\>> can be avoided using
  so-called <dfn|homoscedastic uncertainty>, with a Laplace likelihood.
  Basically it amounts to assuming a model
  <math|p<around*|(|\<b-y\>\|\<b-x\>|)>\<sim\>\<cal-N\><around*|(|f<around*|(|\<b-x\>|)>,\<sigma\>|)>>
  where <math|f> is the neural network, <math|\<b-x\>> the input image and
  <math|\<b-y\>> is one of <math|\<b-p\>,\<b-q\>>. We leave the details of
  this idea for a later post on a previous paper,<\footnote>
    <cite|kendall_multitask_2017> but also <cite|kendall_what_2017>.
  </footnote> but for now, suffice to say that it boils down to adding two
  <em|trainable parameters> <math|<wide|\<sigma\>|^><rsub|p>,<wide|\<sigma\>|^><rsub|q>\<in\>\<bbb-R\>>
  to the model and optimising

  <\equation*>
    \<cal-L\><rsub|\<sigma\>>=\<cal-L\><rsub|p>*\<mathe\><rsup|-<wide|s|^><rsub|p>>+<wide|s|^><rsub|p>+\<cal-L\><rsub|q>*\<mathe\><rsup|-<wide|s|^><rsub|q>>+<wide|s|^><rsub|q>,
  </equation*>

  where for numerical stability reasons the actual parameters learned are
  <math|s<rsub|i>\<assign\>log \<sigma\><rsup|2><rsub|i>>. As the notation
  suggest and was hinted at above, these parameters represent the variance in
  position and rotation of the outputs of the model:

  <\quotation>
    This loss consists of two components; the residual regressions and the
    uncertainty regularization terms. We learn the variance,
    <math|\<sigma\><rsup|2>>, implicitly from the loss function. As the
    variance is larger, it has a tempering effect on the residual regression
    term; larger variances (or uncertainty) results in a smaller residual
    loss. The second regularization term prevents the network from predicting
    infinite uncertainty (and therefore zero loss)
  </quotation>

  <section*|Geometric reprojection>

  A better loss function specific to this domain can be computed based on the
  predicted camera coordinates: the transformation between true and predicted
  camera position is computed, then the 3D test data is reprojected onto the
  2D image from the second and the distance is computed. This is a natural
  way of combining translation and rotation on a per-image basis: in outdoor
  images the balance will be different from indoor images where objects are
  typically closer. Ditto for camera parameters.

  If <math|\<pi\><rsub|K><around*|(|\<b-p\>,\<b-q\>,\<b-x\>|)>=<around*|(|u,v|)>>
  is the projection of a point <math|\<b-x\>\<in\>\<bbb-R\><rsup|3>>, with
  <math|K> being the camera intrisics, the loss on an image <math|I>, with
  visible 3D points <math|G<rprime|'>> is

  <\equation*>
    \<cal-L\><rsub|g><around*|(|I|)>=<frac|1|<around*|\||G<rprime|'>|\|>>*<big|sum><rsub|x<rsub|i>\<in\>G<rprime|'>><around*|\<\|\|\>|\<pi\><around*|(|\<b-p\>,\<b-q\>,\<b-x\><rsub|i>|)>-\<pi\><around*|(|<wide|\<b-p\>|^>,<wide|\<b-q\>|^>,\<b-x\><rsub|i>|)>|\<\|\|\>>.
  </equation*>

  We omitted the index <math|K> because the same projection is applied using
  ground truth and predicted camera coordinates so the camera intrinsics can
  be taken to be the identity.

  It is noteworthy that the authors observed that this loss can fail to
  converge when initialised with random weights (because of high sensitivity
  to large residuals), but provided the best results when used to fine tune a
  model pretrained with the homoscedastic uncertainty loss.

  <section*|Results>

  As usual, check the paper for the benchmarks and the details of how it
  outperforms everyone else ;). We only remark that this model improves the
  already impressive results of <cite|walch_imagebased_2016>, using PoseNet
  with spatial LSTMs, but since this paper uses a linear combination for the
  loss, the fruit is hanging really low! Also, stay tuned:

  <\quotation>
    For many applications which require localization, such as mobile
    robotics, video data is readily available. Ultimately, we would like to
    extend the architecture to video input with further use of multi-view
    stereo.
  </quotation>

  <\bibliography|bib|tm-plain|paperwhy.bib>
    <\bib-list|4>
      <bibitem*|1><label|bib-kendall_what_2017>Alex Kendall<localize| and
      >Yarin Gal.<newblock> What Uncertainties Do We Need in Bayesian Deep
      Learning for Computer Vision?<newblock>
      <with|font-shape|italic|ArXiv:1703.04977 [cs]>, mar 2017.<newblock>
      ArXiv: 1703.04977.<newblock>

      <bibitem*|2><label|bib-kendall_multitask_2017>Alex Kendall, Yarin
      Gal<localize|, and >Roberto Cipolla.<newblock> Multi-Task Learning
      Using Uncertainty to Weigh Losses for Scene Geometry and
      Semantics.<newblock> <with|font-shape|italic|ArXiv:1705.07115 [cs]>,
      may 2017.<newblock> ArXiv: 1705.07115.<newblock>

      <bibitem*|3><label|bib-szegedy_going_2015>Christian Szegedy, Wei Liu,
      Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru
      Erhan, Vincent Vanhoucke<localize|, and >Andrew Rabinovich.<newblock>
      Going deeper with convolutions.<newblock> <localize|In
      ><with|font-shape|italic|2015 IEEE Conference on Computer Vision and
      Pattern Recognition (CVPR)>, <localize|page >9. Jun 2015.<newblock>
      Citecount: 03482.<newblock>

      <bibitem*|4><label|bib-walch_imagebased_2016>Florian Walch, Caner
      Hazirbas, Laura Leal-Taixé, Torsten Sattler, Sebastian
      Hilsenbeck<localize|, and >Daniel Cremers.<newblock> Image-based
      localization using LSTMs for structured feature correlation.<newblock>
      <with|font-shape|italic|ArXiv:1611.07890 [cs]>, nov 2016.<newblock>
      ArXiv: 1611.07890.<newblock>
    </bib-list>
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
    <associate|auto-2|<tuple|<with|mode|<quote|math>|\<bullet\>>|?>>
    <associate|auto-3|<tuple|1|?>>
    <associate|auto-4|<tuple|1|?>>
    <associate|auto-5|<tuple|6|?>>
    <associate|auto-6|<tuple|6|?>>
    <associate|auto-7|<tuple|6|?>>
    <associate|bib-kendall_multitask_2017|<tuple|2|?>>
    <associate|bib-kendall_what_2017|<tuple|1|?>>
    <associate|bib-szegedy_going_2015|<tuple|3|?>>
    <associate|bib-walch_imagebased_2016|<tuple|4|?>>
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
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      kendall_what_2017

      walch_imagebased_2016

      szegedy_going_2015

      walch_imagebased_2016

      kendall_multitask_2017

      kendall_what_2017

      walch_imagebased_2016
    </associate>
    <\associate|figure>
      <tuple|normal|Architecture. Picture shamelessly cannibalized without
      permission from [<write|bib|walch_imagebased_2016><reference|bib-walch_imagebased_2016>].|<pageref|auto-3>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Goal>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Architecture>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Improving
      the loss> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Geometric
      reprojection> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Results>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>