---
title: "DynPartition: Automatic Optimal Pipeline Parallelism of Dynamic Neural Networks over Heterogeneous GPU Systems for Inference Tasks"
layout: post
tags: [ml-systems]
cover: hallstatt.webp
cover_preview: hallstatt.webp
caption: Hallstatt, Gmunden, Austria
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
  Dynamic neural networks are slowly gaining popularity due to their ability to
  adapt their structures or parameters to different inputs, leading to notable
  advantages in terms of accuracy, computational efficiency, and adaptivity, in
  comparison to static models which have fixed computational graphs and
  parameters. We propose a novel reinforcement learning-based scheduler called
  DynPartition that performs dynamic partitioning of computation across multiple
  heterogeneous GPUs for dynamic neural network inference tasks.
---

### Summary

Dynamic neural networks are slowly gaining popularity due to their ability to
adapt their structures or parameters to different inputs, leading to notable
advantages in terms of accuracy, computational efficiency, and adaptivity, in
comparison to static models which have fixed computational graphs and
parameters. We propose a novel reinforcement learning-based scheduler called
DynPartition that performs dynamic partitioning of computation across multiple
heterogeneous GPUs for dynamic neural network inference tasks. Our scheduler is
trained through previous iterations to generate an optimal forward schedule
across heterogeneous GPUs given the network input. Our experiments show that the
RL-based scheduler can successfully converge towards optimal distribution of
computation across devices during inference tasks.

Joint work with [Vivswan Shah](https://www.linkedin.com/in/vivswan/) and [Yudong Liu](https://yudongl2000.github.io/) for [15-712 Advanced and Distributed Operating Systems](https://www.cs.cmu.edu/~15712/).

### Paper

[Link to our paper]({% link /assets/research/DynPartition_Automatic_Pipeline_Parallelism_Of_Dynamic_Neural_Networks_For_Inference.pdf %}).
