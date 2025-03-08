---
title: "Notes on the Ultra-Scale Playbook"
layout: post
tags: [machine-learning, gpu]
cover: rainier-cowlitz-glacier.webp
cover_preview: rainier-cowlitz-glacier.webp
caption: Cowlitz Glacier on Mt Rainier. Washington, USA
class: post-template
author: fanpu
toc:
  sidebar: left
giscus_comments: true
description: >
    TODO
published: false
---

Print, frame, and hang this on your living room: https://nanotron-ultrascale-playbook.static.hf.space/dist/assets/images/ultra-cheatsheet.svg

# Batch Sizes
A sweet spot for recent LLM training is typically on the order of 4-60 million tokens per batch. The batch size as well as the training corpus have been steadily increasing over the years: Llama 1 was trained with a batch size of ~4M tokens for 1.4 trillion tokens while DeepSeek was trained with a batch size of ~60M tokens for 14 trillion tokens.

Distributed training profiling: https://nanotron-ultrascale-playbook.static.hf.space/dist/index.html#kernels

Torch cache allocator: https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html

Counting parameters: https://michaelwornow.net/2024/01/18/counting-params-in-transformer