---
title: "Flexpoint: an adaptive numerical format for efficient training of deep neural networks"
author: "Miguel de Benito Delgado"
authors: ["Miguel de Benito Delgado"]
date: "2017-12-28"
tags: ["efficiency", "quantization", "hardware"]
paper_authors: ["Köster, Urs", "Webb, Tristan J.", "Wang, Xin", "Nassar, Marcel", "Bansal, Arjun K.", "Constable, William H.", "Elibol, Oğuz H.", "Gray, Scott", "Hall, Stewart", "Hornof, Luke", "Khosrowshahi, Amir", "Kloss, Carey", "Pai, Ruby J.", "Rao, Naveen"]
paper_key: "koster_flexpoint_2017"
---

**tl;dr:** A method for hardware accelerated, transparent quantization of 
networks both for training and testing is proposed (and later implemented in 
Intel's Nirvana architecture). It consists of fixed point operations with an 
N-bit mantissa and an M-bit exponent (*flexpointN+M*). The latter is shared for 
all entries in a single tensor, and is managed by the host (as opposed to the 
ASIC) in software: a statistical model predicts future over/underflows based on 
past values and shifts the exponent in order to avoid them. Performance is on 
par with float32 on AlexNet, residual nets and a GAN with no changes to the 
networks themselves nor hyperparameter tuning (for the exponent management 
algorithm).

---

Deep learning is compute- and data-hungry: recent models require dozens or 
even hundreds of GPUs and petabytes of data to train, and this makes results 
irreproducible for most researchers and practitioners. But run-time 
("inference") is also expensive with millions of parameters, which makes 
models unsuited for low-power devices. There are several solutions to this 
acute problem: devise more efficient architectures,[^1] hand-craft features to 
ease the task for the network (although often we are precisely tying to avoid 
this), quantize scalars to lower precision data types for storage and 
communication in parallel implementations,[^2] or in inference and even 
training,[^3] or use lower precision all around.

## The idea

Today's paper takes the last approach: they propose a hardware implementation 
of a new datatype, which is something they can do because, well, they are 
intel.[^4] The key observation leading to the main idea is simple: the whole 
dynamic range of weights in a typical deep network fits into 16 bits, and that 
is enough as long as one chooses the exponent adequately. So one can perform 
integer arithmetic on a 16 bit type and adjust the exponent (represented by 5 
bits) as needed to avoid under / overflows. This was already observed in the 
past, and computations had been made in fixed point 16 bit precision or lower 
but the corresponding exponents had not been predicted with a statistical model 
as now; instead they were changed after overflows occurred.[^5]

{{< figure src="/img/koster_flexpoint_2017_blog_fig_3.png" 
         title="Dynamic range of tensor entries of a ResNet on CIFAR10." >}}

Note that even though techniques like Batchnorm[^6] will improve this 
situation for activations, this phenomenon of concentration of tensor entries 
around their mean happens for weights and weight updates as well.

## flexpointN+M

In a nutshell, the format **flexpointN+M** is designed for whole tensors: each 
scalar is represented by

> (...) an N-bit mantissa storing an integer value in two's complement form, and 
> an M-bit exponent e, shared across all elements of a tensor.

It is essential that the exponent is the same for all scalars in a tensor. It 
is updated after each write to the tensor to adjust for possible 
over/underflows. A predictive model for this called Autoflex is implemented in 
a library on the host. The cost of doing this is paid only once per tensor, 
resulting in great gains:

> Compared to 32-bit floating point, Flexpoint reduces both memory and bandwidth 
> requirements in hardware, as storage and communication of the exponent can be 
> amortized over the entire tensor. Power and area requirements are also 
> reduced due to simpler multipliers compared to floating point. Specifically, 
> multiplication of entries of two separate tensors can be computed as a fixed 
> point operation since the common exponent is identical across all the output 
> elements. For the same reason, addition across elements of the same tensor 
> can also be implemented as fixed point operations.

So, what this technique achieves is almost fixed point performance for most of 
the operations carried while training and running a network. The authors show 
that 16 bit integer arithmetic plus a dynamically adjusted 5 bit exponent 
provide great speedups while retaining transparency for the network designer. 
"All" that is required is a library talking to the ASIC which changes 
exponents as required.

{{< figure src="/img/koster_flexpoint_2017_blog_fig_2.png" 
         title="Flexpoint16+5. " >}}

## Autoflex

For each tensor, a deque of maximum absolute values $\Gamma$ of the mantissas 
is kept. After an update to the tensor at timestep $t$ (either forward or 
backward pass for a minibatch) the deque is updated and the rescaled quantity 
$\phi \_{t} = \Gamma \_{t} \kappa \_{t}$, with $\kappa \_{t} = 2^{- e\_{t}}$ 
and $e\_{t}$ the exponent, is computed. This value is stored and

> We maintain a fixed length dequeue **f** of the maximum floating point values 
> encountered in the previous $l$ iterations, and predict the expected maximum 
> value for the next iteration based on the maximum and standard deviation of 
> values stored in the dequeue. If an overflow is encountered, the history of 
> statistics is reset and the exponent is increased by one additional bit.

Initialisation is a bit tricky since no history is available, so multiple 
guesses are required until valid exponents without overflows and maximising 
utilisation of the mantissa are found.

The paper concludes with simulations run on GPUs comparing performance. As 
announced flex16+5 is on par with float32.

{{< figure src="/img/koster_flexpoint_2017_blog_fig_7.png"
         title="Flexpoint versus floating point on AlexNet, ResNet 110 and a Wasserstein GAN." >}}

One can only wait expectantly for intel to deliver this awesome piece of tech! 
Hopefully they will be able to integrate Autoflex into known deep learning 
frameworks.

If this has piqued your interest, do read their [blog post accompanying the 
paper](https://www.intelnervana.com/flexpoint-numerical-innovation-underlying-intel-nervana-neural-network-processor/). 
It is very nicely written and provides very clear explanations to the ideas in 
the paper.


[^1]: {{< cite zhang_shufflenet_2017 >}}
[^2]: {{< cite dettmers_8bit_2015a >}}.
[^3]: For the extreme case see {{< cite courbariaux_binarized_2016a >}}, which use binary weights and activations for up to x7 speedups at run time with no loss in performance. However, gradients in SGD still must be computed in higher precision, so speedups during training are only limited to the forward pass and are therefore less noticeable. One reason why binarization does not lead to catastrophic failure may be that dot-products are almost preserved in high dimensions, see {{< cite anderson_highdimensional_2017 >}}.
[^4]: Note that Google already has their TPU ASICs which use lower precision for inference with huge speedups.
[^5]: In {{< cite courbariaux_training_2014a >}}, a fixed point data type is used. When the number of entries in a tensor under- or overflowing exceeds some threshold, the exponent for the whole tensor is shifted. The paper we study improves this reactive behaviour turning it into a predictive one.
[^6]: {{< cite ioffe_batch_2015 >}}.
