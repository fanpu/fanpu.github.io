---
layout: summary
title: "Understanding Black-box Predictions via Influence Functions"
giscus_comments: true
bib_id: 1703.04730v3
published: false
---

### Three Important Things

#### 1. Foo

* instead of treating the model as parameters that makes some predictions...
* we can treat model as a function of training data

Counterfactual: if we didn't have this training data, or many copies of this training data, how would predictions change? 

ICML best paper award

standard ERM : pick $\hat{\theta}$ to minimize loss

reweighting it such that is $\epsilon$ mask on a new example - try to fit that example harder

see how does loss change when we have that extra \epsilon
input or not

Influence function is defined as how much this changes
with $\epsilon$. By applying the chain rule, it expands 
as being influenced by the gradient of the gradient of the test loss, the empirical Hessian, and the gradient of the training points.

$$\begin{aligned} \mathcal{I}_{\text {up }, \text { loss }}\left(z, z_{\text {test }}\right) & \left.\stackrel{\text { def }}{=} \frac{d L\left(z_{\text {test }}, \hat{\theta}_{\epsilon, z}\right)}{d \epsilon}\right|_{\epsilon=0} \\ & =\left.\nabla_\theta L\left(z_{\text {test }}, \hat{\theta}\right)^{\top} \frac{d \hat{\theta}_{\epsilon, z}}{d \epsilon}\right|_{\epsilon=0} \\ & =-\nabla_\theta L\left(z_{\text {test }}, \hat{\theta}\right)^{\top} H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta})\end{aligned}$$

Sanity check: if gradient of test point is large, it implies influence the test point has on the data is large.

Issue: the Hessian is not tractable.

Intuition for Hessian: how much the other datapoints want to stay at the original value vs change


#### 2. Bar

#### 3. Baz

### Most Glaring Deficiency

### Conclusions for Future Work
