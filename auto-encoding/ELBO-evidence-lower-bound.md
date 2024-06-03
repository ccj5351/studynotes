# ELBO — What & Why

> see the original article at https://yunfanj.com/blog/2021/01/11/ELBO.html, writen by Jan 11, 2021

ELBO (evidence lower bound) is a key concept in Variational Bayesian Methods. It transforms inference problems, which are always intractable, into optimization problems that can be solved with, for example, gradient-based methods.

Updates
• May 26, 2021
I rewrote the entire story, added more figures, but left derivations unchanged. Wish that the second version could better help you :).

• April 16, 2021
An extensional derivation for the case of temporal sequences has been updated here.

Introduction
In this post, I’ll introduce an important concept in Variational Bayesian (VB) Methods — ELBO (evidence lower bound, also known as variational lower bound) — and its derivations, alongside some digging into it. ELBO enables the rewriting of statistical inference problems as optimization problems, the so-called inference-optimization duality. Combined with optimization methods such as gradient descent and modern approximation techniques, e.g., deep neural networks, inference on complex distributions can be achieved. Numerous applications can be found in VAE, DVRL, MERLIN, to name a few.

Table of Contents
Motivation
Amortized Variational Inference and Evidence Lower Bound
How Good is the Variational Posterior
Extension: ELBO for Temporal Sequence
Summaries
Motivation
We are interested in finding the distribution p(x) of some given observations x. Sometimes, this distribution can be fairly simple. For example, if the observations are the outcomes of flipping a coin, p(x) will be a Bernoulli distribution. In a continuous case, p(x) will be a simple Gaussian distribution if you are measuring the heights of people. However, sadly, we generally encounter observations with complicated distributions. The figure below, for instance, shows such a p(x), which is a mixed Gaussian distribution.

The distribution shown is fairly complicated. It is a Gaussian mixture model with 3 Gaussian distributions.

Similar to the law of total probability which relates marginal probabilities to conditional probabilities, we can think that the distribution of interest p(x) can be transformed from a simple distribution, let’s say, p(z). We will assume that p(z) is a simple Gaussian distribution. Any other types of distributions can play the same role.

p(z) is a simple Gaussian distribution.

Now we will try to use p(z) with some transformation f(⋅) to fit p(x). Concretely, we select several shifted copies of p(z) and multiply each of them with a weight wi. The result is shown in the figure below.

A demo to fit p(x) with p(z) and some transformation

I have to say that this is not a bad fitting consider its simplicity. We can improve this fitting by tweaking the weights, which leads to the following fitting.

We tweak the weights to improve the fitting.

Let’s define this intuition formally. Given observations x, we can build a latent variable model with the variable z (we call it “latent” as it is not observed) such that the distribution of interest p(x) can be decomposed asp(x)=∫zp(x|z)p(z)dz.(1)
The intuition behind Eq. (1) is: We condition our observations on some variables that we don’t know. Therefore, the probability of observations will be the multiplication of the conditional probability and the prior probability of those unknown variables. Subsequently, we integrate out all cases of unknowns to get the distribution of interest. In the above naive case, the shift means and those weights we applied correspond to the term p(x|z), while the transformation, i.e., the summation, corresponds to the integration.

Despite the convenience provided by decomposing a very complicated distribution into the multiplication of a simple Gaussian conditional distribution and a Gaussian prior distribution, there is a PROBLEM — the integration. It is intractable because it is performed over the whole latent space, which is impractical when latent variables are continuous.

Besides the above-mentioned intractable integration, another question would be that how to obtain a function that transforms p(z) into p(x). In other words, how to get the conditional distribution p(x|z). Despite that this function seems to be extremely non-linear, it is still not difficult to solve this problem as we know that neural networks are universal function approximators that can approximate any functions to arbitrary precisions. Therefore, we could use a neural network with parameters θ to approximate the distribution p(x|z), which gives us pθ(x|z). In the subsequent sections, we will see how we can avoid the intractable integration and optimize our parameters.

Amortized Variational Inference and Evidence Lower Bound
Parameters θ can be obtained from maximum likelihood estimation: θ=argmaxθlogpθ(x). To avoid integrating over the whole latent space, a natural question would be “Can we infer any information about z after observing a sample xi∈X?”. The answer is “Yes” and we can use qi(z) to approximate the distribution of z given the i-th observation. This idea is the statistical inference, which aims to infer the value of one random variable given the observed value of another random variable. Nevertheless, there is an obvious drawback behind this intuition. The number of parameters of qi(z) will scale up with the size of the set of observations because we build individual distribution after observing each data. To alleviate this problem, we introduce another network with parameters ϕ to parameterize an approximation of qi(z), i.e., qϕ(z|x)≈qi(z)∀xi∈X, such that the increase of the number of parameters is amortized. This is the amortized variational inference, which is also referred to as variational inference in recent literature. Up to this point, as shown below, we have explicitly built a probabilistic graphical model to represent our problem where the observation x is conditioned on the latent variable z and we aim to infer z after observing x.

A probabilistic graphical model showing relations between x and z

Now let’s revisit our objective to maximize the log-likelihood of observations x but with qϕ(z|x) this time.

logpθ(x)=log∫zpθ(x,z)dz=log∫zpθ(x,z)qϕ(z|x)qϕ(z|x)dz=logEz∼qϕ(z|x)[pθ(x,z)qϕ(z|x)]≥Ez[logpθ(x,z)qϕ(z|x)]by Jensen's inequality=Ez[logpθ(x,z)]+∫zqϕ(z|x)log1qϕ(z|x)dz=Ez[logpθ(x,z)]+H(qϕ(z|x)).(2)
In the above equation, the term H(⋅) is the Shannon entropy. By definition, the term “evidence” is the value of a likelihood function evaluated with fixed parameters. With the definition of L=Ez[logpθ(x,z)]+H(qϕ(z|x)), it turns out that L sets a lower bound for the evidence of observations and maximizes L will push up the log-likelihood of x. Hence, we call L the evidence lower bound (ELBO, sometimes referred to as variational lower bound as well).

Now let’s think about the rationale behind L. First, we focus on the term Ez[logpθ(x,z)] where z∼qϕ(z|x). Assuming that the neural network with parameters θ gives us the joint distribution pθ(x,z), the optimal distribution q∗ϕ(z|x) that maximizes L will be a Dirac delta which puts all the probability mass at the maximum of pθ(x,z). The interpretation is as follows. The operation of taking expectation is to just take a weighted average. In the case where data being averaged are fixed but weights can be varied (with the constraint that all weights sum to one), you just need to put 1 for the largest data point and 0 for others to maximize that average. With this intuition, we get the optimal distribution q∗ϕ(z|x) shown below.

The optimal distribution is a Dirac delta.

However, the story becomes different when we consider the second term in L, i.e., the entropy term. This term tells us the uncertainty of a distribution. Samples drawn from a distribution with higher entropy will become more uncertain. Sadly, the entropy of the optimal distribution q∗ϕ(z|x) we have just found is negative infinity. We can show this by constructing a random variable x drawn from a uniform distribution x∼U(x0−ϵ,x0+ϵ). Its entropy is Ex[log1p(x)]=log(2ϵ). As ϵ approaching zero, this distribution degenerates to a Dirac delta with entropy limϵ→0log(2ϵ)=−∞. The figure below shows the entropy varies as a function of qϕ(z|x).

The entropy varies as a function of different distributions.

Put all of them together, the maximization of L tries to find an optimal distribution q∗ϕ(z|x) which not only fits peaks of pθ(x,z) but also spreads as wide as possible. A visualization is given in the demo below.

A visualization of maximizing ELBO

The neural network with parameters ϕ is sometimes called the inference network, with the distribution qϕ(z|x) that it parameterizes named as the variational posterior.

How Good is the Variational Posterior
We care about the accuracy of the approximation performed by the inference network. As we mentioned earlier, the amortized variational inference leverages a distribution qϕ(z|x) to approximate the true posterior of z given x, i.e., p(z|x). We choose Kullback–Leibler divergence as the metric to measure how close is qϕ(z|x) to p(z|x).

DKL(qϕ(z|x)∥p(z|x))=∫zqϕ(z|x)logqϕ(z|x)p(z|x)dz=−∫zqϕ(z|x)logp(z|x)qϕ(z|x)dz=−∫zqϕ(z|x)logp(z,x)qϕ(z|x)p(x)dz=−(∫zqϕ(z|x)logp(z,x)qϕ(z|x)dz−∫zqϕ(z|x)logp(x)dz)=−∫zqϕ(z|x)logp(z,x)qϕ(z|x)dz+logp(x).(3)
It is easy to show that the term ∫zqϕ(z|x)logp(z,x)qϕ(z|x)dz is equal to L, i.e., ELBO we defined previously. Rewriting Eq. (3) gives

logp(x)=L+DKL(qϕ(z|x)∥p(z|x)).(4)
Although the true posterior p(z|x) is unknown and hence we cannot calculate the KL divergence term analytically, an important property of non-negativity of KL divergence allows us to write Eq. (4) into an inequality:logp(x)≥L,(5)which is consistent with Eq. (2) we derived before.

Another way to investigate ELBO is to rewrite it in the following way.L=∫zqϕ(z|x)logpθ(x,z)qϕ(z|x)dz=∫zqϕ(z|x)logpθ(x|z)p(z)qϕ(z|x)dz=Ez∼qϕ(z|x)[pθ(x|z)]−DKL(qϕ(z|x)∥p(z))(6)
It suggests that the ELBO is a trade-off between the reconstruction accuracy against the complexity of the variational posterior. The KL divergence term can be interpreted as a measure of the additional information required to express the posterior relative to the prior. As it approaches zero, the posterior is fully obtainable from the prior. Another intuition behind Eq. (6) is that we draw latent variables z from an approximated posterior distribution, which is very close to its prior, and then use them to reconstruct our observations x. As the reconstruction gets better, our approximated posterior will become more accurate as well. From the perspective of auto-encoder, the neural network with parameters ϕ is called encoder because it maps from the observation space to the latent space, while the network with parameters θ is called decoder because it maps from the latent to the observation space. Readers who are interested in this convention are referred to Kingma et al..

Additionally, let’s think about the reason behind the KL divergence we used to derive Eq. (3):

DKL(qϕ(z|x)∥p(z|x))=∫zqϕ(z|x)logqϕ(z|x)p(z|x)dz.(7)
It suggests that the variational posterior qϕ(z|x) is prevented from spanning the whole space relative to the true posterior p(z|x). Consider the case where the denominator in Eq. (7) is zero, the value of qϕ(z|x) has to be zero as well otherwise the KL divergence goes to infinity. The figure below demonstrates this. Note that the green region in the left figure indicates where qϕ(z|x)p(z|x)=0, while the red region in the right figure indicates where qϕ(z|x)p(z|x)=∞. In summary, the reverse KL divergence has the effect of zero-forcing as minimizing it leads to qϕ(z|x) being squeezed under p(z|x).

The zero-forcing effect of reverse KL divergence.

Extension: ELBO for Temporal Sequence
Consider the case that we wish to build a generative model p(x0:t,z0:t) for sequential data x0:t≡(x0,x1,…,xt) with a sequence of latent variable z0:t≡(z0,z1,…,zt), we can also derive a corresponding ELBO as a surrogate objective. Optimizing this objective leads to the maximization of the likelihood of the sequential observations.

logp(x0:t)=log∫z0:tp(x0:t,z0:t)dz0:t=log∫z0:tp(x0:t,z0:t)qϕ(z0:t|x0:t)qϕ(z0:t|x0:t)dz0:t=logEz0:t∼qϕ(z0:t|x0:t)[p(x0:t,z0:t)qϕ(z0:t|x0:t)]≥Ez0:t[logp(x0:t,z0:t)qϕ(z0:t|x0:t)]by Jensen's inequality(8)
So far, this is similar to what we have derived for the stationary case, i.e., Eq. (2) in the previous section. However, the following derivation will require some factorizations of the joint distribution and the variational posterior. Concretely, we factorize the temporal model p(x0:t,z0:t) and the approximation qθ(z0:t|x0:t) as

p(x0:t,z0:t)=∏τ=0tp(xτ|zτ)p(zτ|z0:τ−1),(9)
and

qϕ(z0:t|x0:t)=∏τ=0tqϕ(zτ|z0:τ−1,x0:τ),(10)
respectively.

To understand these factorizations, we can think that at each time step, the observation conditions on the latent variable at that time step, which also conditions on all latent variables before that time step. Expressing this relation recursively leads to Eq. (9). Similarly, the approximated latent variable at each time step conditions on the sequential observations up to that time and the history of latent variables, which is Eq. (10).

With these two factorizations, we can further derive Eq. (8) by plugging Eq. (9) and Eq. (10):

Ez0:t[logp(x0:t,z0:t)qϕ(z0:t|x0:t)]=Ez0:t[log∏tτ=0p(xτ|zτ)p(zτ|z0:τ−1)∏tτ=0qϕ(zτ|z0:τ−1,x0:τ)]=Ez0:t[∑τ=0tlogp(xτ|zτ)+logp(zτ|z0:τ−1)−logqϕ(zτ|z0:τ−1,x0:τ)]=∑τ=0tEz0:t[logp(xτ|zτ)+logp(zτ|z0:τ−1)−logqϕ(zτ|z0:τ−1,x0:τ)].(11)
Now we will use one trick to replace variables. Note that as the variable τ starts from 0 to t, those items being taken expectation, i.e., logp(xτ|zτ)+logp(zτ|z0:τ−1)−logqϕ(zτ|z0:τ−1,x0:τ) will become invalid for τ<τ′≤t. Therefore, we can write the original expectation term Ez0:t[⋅] as Ez0:τ[⋅]. Furthermore, another trick will allow us to factorize the expectation. Given the expectation taken w.r.t. z0:τ∼qϕ(z0:τ|x0:τ), i.e., Ez0:τ∼qϕ(z0:τ|x0:τ)[⋅], we can factorize it as Ezτ∼qϕ(zτ|z0:τ−1,x0:τ)Ez0:τ−1∼qϕ(z0:τ−1|x0:τ−1)[⋅]. With these tricks at hands, Eq. (11) can be written as

∑τ=0tEz0:t[logp(xτ|zτ)+logp(zτ|z0:τ−1)−logqϕ(zτ|z0:τ−1,x0:τ)]=∑τ=0tEzτEz0:τ−1[logp(xτ|zτ)+logp(zτ|z0:τ−1)−logqϕ(zτ|z0:τ−1,x0:τ)]=∑τ=0tEz0:τ−1Ezτ[logp(xτ|zτ)+logp(zτ|z0:τ−1)−logqϕ(zτ|z0:τ−1,x0:τ)]=∑τ=0tEz0:τ−1Ezτ[logp(xτ|zτ)−logqϕ(zτ|z0:τ−1,x0:τ)p(zτ|z0:τ−1)]=∑τ=0tEz0:τ−1[Ezτ[logp(xτ|zτ)]−Ezτ[logqϕ(zτ|z0:τ−1,x0:τ)p(zτ|z0:τ−1)]]=∑τ=0tEz0:τ−1[Ezτ[logp(xτ|zτ)]−DKL(qϕ(zτ|z0:τ−1,x0:τ)∥p(zτ|z0:τ−1))].(12)
Put all of them together, we have derived a lower bound for the log-likelihood of temporal sequence. Great!

logp(x0:t)≥∑τ=0tEz0:τ−1[Ezτ[logp(xτ|zτ)]−DKL(qϕ(zτ|z0:τ−1,x0:τ)∥p(zτ|z0:τ−1))](13)
If we compare the derived ELBO for temporal sequence, i.e., Eq. (13), with the ELBO for the stationary observation, i.e., Eq. (6), we will find that ELBO for sequential observations is computed firstly by calculating the ELBO for a certain time step. Then this result is taken expectation w.r.t. histories of latent variables considering the property of a sequence. Finally, results are summed up along time step. Don’t be scared by the math, it is fairly easy to understand if we start from the stationary case.

logp(x0:t)≥∑τ=0tEz0:τ−1⎡⎣⎢⎢Ezτ[logp(xτ|zτ)]−DKL(qϕ(zτ|z0:τ−1,x0:τ)∥p(zτ|z0:τ−1))Eq. (6)⎤⎦⎥⎥
Summaries
In this post, we begin with the motivation to fit complicated distributions, then notice the intractable integration, subsequently introduce the amortized variational inference and derive the ELBO from several points of view, and finally, dig deeper facts behind the ELBO. An extension of derivation for temporal sequences is also provided. As I mentioned at the very beginning, it plays an important role because it provides a framework in which statistical inference can be transformed into optimization, leading to more and more amazing applications in the deep learning community.

Thanks for your interest and reading!