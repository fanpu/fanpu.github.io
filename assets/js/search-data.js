// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cmu-course-reviews",
          title: "CMU Course Reviews",
          description: "Reviews and Recommendations for Computer Science and Math Courses at CMU",
          section: "Navigation",
          handler: () => {
            window.location.href = "/courses/";
          },
        },{id: "nav-cmu-online",
          title: "CMU Online",
          description: "Publicly Accessible CMU Courses",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cmu-online/";
          },
        },{id: "nav-ml-paper-summaries",
          title: "ML Paper Summaries",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/summaries/";
          },
        },{id: "nav-ml-reading-list",
          title: "ML Reading List",
          description: "Curated list of papers that I have bookmarked to read, well, someday...",
          section: "Navigation",
          handler: () => {
            window.location.href = "/reading-list/";
          },
        },{id: "post-bounding-mixing-times-of-markov-chains-via-the-spectral-gap",
      
        title: "Bounding Mixing Times of Markov Chains via the Spectral Gap",
      
      description: "An aperiodic and irreducible Markov chain will eventually converge to a stationary distribution. This is used in many applications in machine learning like Markov Chain Monte Carlo (MCMC) methods, where random walks on Markov chains are used to obtain a good estimate of the log likelihood of the partition function of a model, which is hard to compute directly as it is #P-hard (this is even harder than NP-hard). However, one common problem is that it is unclear how many steps we should take before we are guaranteed that the Markov chain has converged to the its stationary distribution.   In this post, we understand how the spectral gap of the transition matrix of the Markov chain relates to its mixing time.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/bounding-markov-chain-mixing-times-by-spectral-gap/";
        
      },
    },{id: "post-notes-on-39-the-llama-3-herd-of-models-39",
      
        title: "Notes on &#39;The Llama 3 Herd of Models&#39;",
      
      description: "Notes on the new Llama 3.1 technical report. It&#39;s a long paper, but one that&#39;s well-written with lots of interesting technical details and  design choices.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/llama-3.1-technical-report-notes/";
        
      },
    },{id: "post-playing-sound-voltex-at-home-setting-up-unnamed-sdvx-clone-with-the-yuancon-sdvx-controller",
      
        title: "Playing Sound Voltex at Home: Setting Up Unnamed SDVX Clone with the Yuancon...",
      
      description: "Rhythm is just a $200 controller and some hopefully-not-too-complicated open source software setup away!This beginner&#39;s guide will help to demystify the process of setting up Sound Voltex at home using a custom SDVX controller using Unnamed SDVX Clone.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/setting-up-yuancon-controller-sound-voltex/";
        
      },
    },{id: "post-creating-trackback-requests-for-static-sites",
      
        title: "Creating Trackback Requests for Static Sites",
      
      description: "A simple guide on creating manual Trackback requests for static sites to increase visibility and discoverability",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/creating-trackback-requests/";
        
      },
    },{id: "post-a-unified-framework-for-high-dimensional-analysis-of-m-estimators-with-decomposable-regularizers-a-guided-walkthrough",
      
        title: "A Unified Framework for High-Dimensional Analysis of M-Estimators with Decomposable Regularizers: A Guided...",
      
      description: "Imagine doing high-dimensional statistical inference, but instead of repeatedly studying different settings with specific low-dimensional constraints (such as linear regression with sparsity constraints, or estimation of structured covariance matrices), there is a method for performing a unified analysis using appropriate notions.    Well, you&#39;re in luck! &#39;A Unified Framework for High-Dimensional Analysis of \( M \)-Estimators with Decomposable Regularizers&#39; by Negahban, Ravikumar, Wainwright, and Yu shows that the \( \ell_2 \) difference between any regularized \(M\)-estimator and its true parameter can be bounded if the regularization function is decomposable, and the loss function satisfies restricted strong convexity.   The goal of this post is to provide intuition for the result and develop sufficient background for understanding the proof of this result, followed by a walkthrough of the proof itself.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/high-dimensional-analysis-of-m-estimators/";
        
      },
    },{id: "post-the-cmu-steam-tunnels-and-wean-9",
      
        title: "The CMU Steam Tunnels and Wean 9",
      
      description: "If you&#39;re curious about the infamous steam tunnels at CMU, or what the views from the roof of Wean Hall looks like, this post is for you!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/cmu-steam-tunnels/";
        
      },
    },{id: "post-cmu-15712-advanced-operating-systems-and-distributed-systems-course-review",
      
        title: "CMU 15712 Advanced Operating Systems and Distributed Systems Course Review",
      
      description: "15-712 Advanced OS was an excellent seminar-based graduate course that took us on a whirlwind tour through many of the most seminal SigOps Hall of Fame papers across several systems domains. It will prepare you to be a great systems designer and researcher. In this post, I will share my experience in the class, the course structure and content, what I thought were the biggest takeaways, and who this class might be suitable for.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/advanced-operating-systems-course-review/";
        
      },
    },{id: "post-score-based-diffusion-models",
      
        title: "Score-Based Diffusion Models",
      
      description: "Score-based diffusion models are a promising direction for generative models, as they improve on both likelihood-based approaches like variational autoencoders, as well as adversarial methods like Generative Adversarial Networks (GANs).  In this blog post, we survey recent developments in the field centered around the line of results developed in (Song &amp; Ermon, 2019), analyze the current strengths and limitations of score-based diffusion models, and discuss possible future directions that can address its drawbacks. Joint work with Owen Wang.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/score-based-diffusion-models/";
        
      },
    },{id: "post-the-art-of-latex-common-mistakes-and-advice-for-typesetting-beautiful-delightful-proofs",
      
        title: "The Art of LaTeX: Common Mistakes, and Advice for Typesetting Beautiful, Delightful Proofs...",
      
      description: "When was the first time you had to use LaTeX? If you are like most people, it was probably suddenly forced upon you during your first math or CS class where you had to start writing proofs, with minimal guidance on how to get started. Unfortunately, this meant that while many people have good operational knowledge of LaTeX, there are still many small mistakes and best practices which are not followed, which are not corrected by TAs as they are either not severe enough to warrant a note, or perhaps even the TAs themselves are not aware of them.  In this post, we cover some common mistakes that are made by LaTeX practitioners (even in heavily cited papers), and how to address them.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2023/latex-tips/";
        
      },
    },{id: "post-a-concise-proof-of-the-central-limit-theorem-and-its-actually-useful-version-the-berry-esseen-theorem",
      
        title: "A Concise Proof of the Central Limit Theorem, and Its Actually Useful Version,...",
      
      description: "The Central Limit Theorem is widely used in statistics and machine learning, as it allows us to assume that given enough samples, the mean of the samples will follow a normal distribution. This holds even if the samples come from a distribution that is not normally distributed. In this post, we prove the Central Limit Theorem, and then take a look at the Berry-Esseen Theorem, which actually provides a quantitative bound on the convergence of the distribution and can therefore be actually used in deriving theoretical bounds.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/central-limit-theorem-and-berry-esseen/";
        
      },
    },{id: "post-reinforcement-learning-policy-optimization-deriving-the-policy-gradient-update",
      
        title: "Reinforcement Learning Policy Optimization: Deriving the Policy Gradient Update",
      
      description: "Reinforcement learning algorithms that learn a policy (as opposed to implicit policy methods like \(\epsilon\)-greedy) optimize their policies by updating their policies in the direction of the gradient. However,  the precise environment dynamics are not usually known to us,  and the state space is usually also too large to enumerate, which means that we still cannot compute the gradient analytically. In this post, we derive the policy gradient update from scratch, and show how it can be approximated by sampling sufficiently many trajectories.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/deriving-the-policy-gradient-update/";
        
      },
    },{id: "post-pseudo-determinism-for-graph-streaming-problems",
      
        title: "Pseudo-determinism for Graph Streaming Problems",
      
      description: "Given a fixed input for a search problem, pseudo-deterministic algorithms produce the same answer over multiple independent runs, with high probability. For example, we can efficiently find a certificate for inequality of multivariate polynomials pseudo-deterministically, but it is not known how to do so deterministically. The same notion can be extended to the streaming model. The problem of finding a nonzero element from a turnstile stream is previously shown to require linear space for both deterministic and pseudo-deterministic algorithms. Another model of streaming problems is that of graphs, where edge insertions and deletions occur along a stream. Some natural problems include connectivity, bipartiteness, and colorability of a graph. While the randomized and deterministic graph streaming algorithms have been mostly well-studied, we investigate pseudo-deterministic space lower bounds and upper bounds for graph theoretic streaming problems.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/pseudo-determinism-for-graph-streaming-problems/";
        
      },
    },{id: "post-graphical-bayesian-networks-with-topic-modeling-priors-for-predicting-asset-covariances",
      
        title: "Graphical Bayesian Networks with Topic Modeling Priors for Predicting Asset Covariances",
      
      description: "Covariance matrix prediction is a long-standing challenge in modern portfolio theory and quantitative finance. In this project, we investigate the effectiveness of Bayesian networks in predicting the covariance matrix of financial assets (specifically a subset of the S&amp;P 500), evaluated against Heterogeneous Autoregressive (HAR) models. In particular, we consider both HAR-DRD, based on the DRD decomposition of the covariance matrix, and Graphical HAR (GHAR)-DRD, which is also based on DRD decomposition but also makes use of graphical relationships between the assets. To build the graph representing relationships between the assets, we apply Latent Dirichlet allocation (LDA) on the 10-K filings of each of the companies, and infer edges based on topic overlap.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/graphical-bayesian-network-for-predicting-asset-covariances/";
        
      },
    },{id: "post-analysis-of-symmetry-and-conventions-in-off-belief-learning-obl-in-hanabi",
      
        title: "Analysis of Symmetry and Conventions in Off-Belief Learning (OBL) in Hanabi",
      
      description: "Hanabi has been proposed as the new frontier for developing strategies in cooperative AI, currently a very nascent area of AI research. A recent algorithm that has been developed for multi-agent reinforcement learning in a cooperative context is the Off-Belief Learning (OBL) algorithm, which is based on iterated reasoning starting from a base policy. We investigate if policies learnt by agents using the OBL algorithm in the multi-player cooperative game Hanabi in the zero-shot coordination (ZSC) context are invariant across symmetries of the game, and if any conventions formed during training are arbitrary or natural, both of which are desirable properties.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/symmetry-and-conventions-in-obl-hanabi/";
        
      },
    },{id: "post-improving-domain-adaptation-of-transformer-models-for-generating-reddit-comments",
      
        title: "Improving Domain Adaptation of Transformer Models For Generating Reddit Comments",
      
      description: "We improve upon the recent success of large language models based on the transformer architecture by investigating and showing several methods that have empirically improved its performance in domain adaptation. We use a pre-trained GPT-2 model and perform fine-tuning on 5 different subreddits, and use different methods of ordering the training data based on our priors about the input to see how this affects the prediction quality of the trained model. We propose a new metric for evaluating causal language modeling tasks called APES (Average Perplexity Evaluation for Sentences) to address the limitations of existing metrics, and apply them to our results. Our results are evaluated against both LSTM and GPT-2 baselines.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/improving-domain-adaptation-of-transformer-models-for-reddit-comments/";
        
      },
    },{id: "post-efficient-low-rank-approximation-via-affine-embeddings",
      
        title: "Efficient Low Rank Approximation via Affine Embeddings",
      
      description: "Suppose you have a \(n \times d\) matrix \(A\), where both dimensions are large. This could represent something like a customer-product matrix used in online recommender systems, where each cell \(A_{i,j}\) denotes how many times customer \(i\) purchased item \(j\). Then it is typically the case that \(A\) can be well-approximated by a low-rank matrix. For instance, using the previous example, there might only be a few dominant patterns that describes purchasing behavior in \(A\), and the rest of it is just noise.Therefore, if we can find such a low-rank approximation, we can achieve significant space savings, and can also help to make the data more interpretable. In this post, we explore how affine embeddings via the CountSketch matrix allows us to perform low rank approximation in time \(O\left(\nnz{A}+(n+d) \text{poly} \left( \frac{k}{\epsilon} \right)\right)\).",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/affine-embeddings-for-low-rank-approximation/";
        
      },
    },{id: "post-cmu-15-441-641-computer-networks-course-review",
      
        title: "CMU 15-441/641 Computer Networks Course Review",
      
      description: "Computer Networks is one of the lesser-known systems classes at Carnegie Mellon that turned out to be surprisingly fun and informative. In this post I&#39;ll talk about the projects and content covered, followed by my own thoughts on the usefulness on the class and who should take it.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/cmu-computer-networks-review/";
        
      },
    },{id: "post-solving-genshin-impact-39-s-ancient-azure-stars-quest-in-linear-time",
      
        title: "Solving Genshin Impact&#39;s Ancient Azure Stars quest in Linear Time",
      
      description: "It is summer yet again, and miHoYo has blessed us with the Summertime Odyssey event that explores the (often dark and painful) backstories of the cast comprising Kazuha, Xinyan, Fischl, and Mona, back on the setting of the Golden Apple Archipelago.One puzzle that I found interesting from a computational perspective was a major part of Mona&#39;s questline Ancient Azure Stars, which is the main topic of this post. In this puzzle, you are given a pattern that resembles a constellation that you need to imitate. The puzzle is interesting because even though its mechanics allows for an exponential search space (and also multiple possible solutions), clever algorithmic techniques can speed up finding a valid solution to almost linear time. This post is meant to be accessible to people with only some exposure to algorithms, and takes things step by step.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/solving-genshin-impact-ancient-azure-stars-in-linear-time/";
        
      },
    },{id: "post-impagliazzo-39-s-five-worlds-or-the-computational-im-possibilities-of-the-world-that-we-live-in",
      
        title: "Impagliazzo&#39;s Five Worlds, or The Computational (Im)Possibilities of The World That We Live...",
      
      description: "Most people have probably heard of the P = NP? problem in some shape or form, which asks whether the class of languages decidable in deterministic polynomial time is the same as the class of languages decidable in non-deterministic polynomial time. However, there are also several other interesting classes of intermediate possibilities that can arise if it was the case that P != NP, as this post explores.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/impagliazzos-five-worlds/";
        
      },
    },{id: "post-my-sharing-at-the-hwa-chong-undergrad-alumni-forum",
      
        title: "My Sharing at the Hwa Chong Undergrad Alumni Forum",
      
      description: "Last week, I had the wonderful opportunity to participate in Hwa Chong&#39;s Undergrad Alumni Forum to share my experiences studying at Carnegie Mellon&#39;s School of Computer Science, and to answer any questions that the students had about studying in the US and pursuing Computer Science as a degree. I was a student at Hwa Chong Institution from 2010-2015, during which I made many happy memories and learnt a lot about myself and the world.I had personally found these sharings really helpful back when I was still a student, and am very grateful for the chance to pass it on and hopefully help to inspire and encourage some of the participants to pursue their education overseas. It has personally had brought about incredible personal and intellectual growth, and exposed me to people and ideas that I would otherwise never have had the chance to meet.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/cmu-scs-ecg-talk/";
        
      },
    },{id: "post-the-delightful-consequences-of-the-graph-minor-theorem",
      
        title: "The Delightful Consequences of the Graph Minor Theorem",
      
      description: "The graph minor theorem, also known as the Robertson–Seymour theorem, is generally regarded as the most important result in graph theory. In this post we introduce the graph minor theorem, provide the necessary background, and discover its delightfully deep algorithmic and philosophical implications.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/the-delightful-consequences-of-the-graph-minor-theorem/";
        
      },
    },{id: "post-universal-types-and-your-type-checker-doesn-39-t-suck-as-much-as-you-think",
      
        title: "Universal types, and your type checker doesn&#39;t suck as much as you think...",
      
      description: "Universal types are very useful for performing generic programming, which allows you to use the same code over different types. For instance, the C++ STL (Standard Template Library) allows you to work on things like containers over any arbitrary type. You would surely not want to re-implement the logic for every concrete type that you use. Such a feature is known as parametric polymorphism. It is different from the other kind of polymorphism normally found in object-oriented languages that allows for overloading and run-time dispatch, which is known as ad-hoc polymorphism.  In this post, we understand the theoretical foundations underpinning universal types, and conclude with the landmark result that typechecking with universal types is undecidable.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2022/universal-types/";
        
      },
    },{id: "post-against-government-scholarships",
      
        title: "Against Government Scholarships",
      
      description: "Over the past few years, I&#39;ve received numerous questions about studying abroad and whether it is advisable to take a government scholarship to do so from my juniors.  Having witnessed firsthand the early career trajectories of many scholars and non-scholars alike, and having heard their different perspectives, I feel like I have a good understanding of the concerns at hand to answer this question confidently. This post is aimed at Singaporean junior college students debating whether they should take on a scholarship from the government to pursue a fully-paid education abroad. In this post, I will consider both the pros and cons of taking a scholarship. The relevance of some of the points is highly dependent on your current situation, so ultimately take it with a grain of salt and consider your unique circumstances. It is useful to keep in mind that a scholarship is designed to benefit the government&#39;s interests, not your own.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/against-government-scholarships/";
        
      },
    },{id: "post-why-you-can-39-t-create-a-value-with-the-bottom-type-and-why-it-39-s-still-useful",
      
        title: "Why you can&#39;t create a value with the Bottom type (and why it&#39;s...",
      
      description: "If you&#39;ve used any sort of object-oriented language, you may be familiar with the notion but not the name of the top-level type \(\top\). For instance, in Java, the `Object` class forms the root of the class hierarchy, and similarly, in Python, all objects inherit from `object`.  Analogously, there is also a bottom-level type \(\bot\), but it cannot be instantiated by any value in any language.  This post will explain why.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/why-you-cant-create-the-bottom-type/";
        
      },
    },{id: "post-quantum-physics-inaccuracies-in-seishun-buta-yaro-rascal-does-not-dream-of-bunny-girl-senpai",
      
        title: "Quantum physics inaccuracies in Seishun Buta Yaro (Rascal Does Not Dream of Bunny...",
      
      description: "I finished watching Seishun Buta Yaro (Rascal Does Not Dream of Bunny Girl Senpai) recently, and really enjoyed the anime. It explored themes of finding one&#39;s identity, and dealing with social anxiety during adolescent years. In the show, students are inflicted by a fictional disease called Adolescence Syndrome when they are severely affected by the things happening around them, whose symptoms depends on the specific reason why a person is mentally distraught. Rio Futaba, the president of the science club in the school, tries to explain the reasons for Adolescence Syndrome using quantum mechanics principles throughout the show. Unfortunately, most of it was poorly applied, and could be harmful in reinforcing misconceptions of quantum mehanics. In this post, I examine each of her claims and describe why they don&#39;t really make sense. This post is aimed at a general audience and requires no knowledge of quantum mechanics.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/quantum-inaccuracies-in-seishun-buta-yaro/";
        
      },
    },{id: "post-writing-a-dpll-sat-solver",
      
        title: "Writing a DPLL SAT Solver",
      
      description: "Boolean satisfiability (SAT) solvers have played an important role in software and hardware verification, automatic test pattern generation, planning, scheduling, and solving challenging problems in algebra. In this post, I talk about my experience writing my own SAT solver, its implementation details, designs and algorithms used, some comparisons between using different heuristics for splitting and choosing of variable assignments in the unforced case, challenges faced, and possible future directions.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/a-dpll-sat-solver/";
        
      },
    },{id: "post-playing-atari-using-deep-reinforcement-learning",
      
        title: "Playing Atari using Deep Reinforcement Learning",
      
      description: "In this post, we study the first deep reinforcement learning model that was successfully able to learn control policies directly from high dimensional sensory inputs, as applied to games on the Atari platform. This is achieved by Deep Q Networks (DQN).",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/atari-with-deep-rl/";
        
      },
    },{id: "post-hopskipjumpattack-an-efficient-adversarial-attack-against-machine-learning-algorithms",
      
        title: "HopSkipJumpAttack: An Efficient Adversarial Attack against Machine Learning Algorithms",
      
      description: "Many machine learning algorithms have been shown to be susceptible to adversarial examples. For example, image classification neural networks can wrongly classify an image when a small perturbation, unnoticeable to the human eye, is added to the original image which it has previously correctly classified. The goal of an adversarial attack can thus be rephrased as an optimization problem to compute the &quot;smallest&quot; perturbation needed, such that the perturbed example will be misclassified.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2021/hop-skip-jump-attack/";
        
      },
    },{id: "post-breaking-cmu-39-s-bomblab-with-angr-for-fun-and-profit-part-7",
      
        title: "Breaking CMU&#39;s Bomblab with Angr for Fun and Profit - Part 7",
      
      description: "The final part of cracking CMU&#39;s Bomblab with Angr, where we crack the secret phase together!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-7/";
        
      },
    },{id: "post-breaking-cmu-39-s-bomblab-with-angr-for-fun-and-profit-part-6",
      
        title: "Breaking CMU&#39;s Bomblab with Angr for Fun and Profit - Part 6",
      
      description: "We tackle Part 6 of this series on cracking CMU&#39;s Bomblab with Angr!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-6/";
        
      },
    },{id: "post-breaking-cmu-39-s-bomblab-with-angr-for-fun-and-profit-part-5",
      
        title: "Breaking CMU&#39;s Bomblab with Angr for Fun and Profit - Part 5",
      
      description: "Part 5 on cracking CMU&#39;s Bomblab with Angr, where we solve Phase 5 together!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-5/";
        
      },
    },{id: "post-breaking-cmu-39-s-bomblab-with-angr-for-fun-and-profit-part-4",
      
        title: "Breaking CMU&#39;s Bomblab with Angr for Fun and Profit - Part 4",
      
      description: "Part 4 of cracking CMU&#39;s Bomblab using Angr!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-4/";
        
      },
    },{id: "post-breaking-cmu-39-s-bomblab-with-angr-for-fun-and-profit-part-3",
      
        title: "Breaking CMU&#39;s Bomblab with Angr for Fun and Profit - Part 3",
      
      description: "We&#39;re back now with Part 3 of this series on cracking CMU&#39;s Bomblab using Angr!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-3/";
        
      },
    },{id: "post-breaking-cmu-39-s-bomblab-with-angr-for-fun-and-profit-part-2",
      
        title: "Breaking CMU&#39;s Bomblab with Angr for Fun and Profit - Part 2",
      
      description: "This is Part 2 of the series on cracking CMU&#39;s Bomblab using Angr, where we solve Phase 2 by pushing symbolic values onto the stack to",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-2/";
        
      },
    },{id: "post-breaking-cmu-39-s-bomblab-with-angr-for-fun-and-profit-part-1",
      
        title: "Breaking CMU&#39;s Bomblab with Angr for Fun and Profit - Part 1",
      
      description: "I have recently been learning about Angr, a binary analysis framework developed by UC Santa Barbara and Arizona State University. It caught my eye because of its versatility and utility in reverse engineering binaries whose disassembly and decompilation are hard to understand manually. Oftentimes, it is simply due to the fact that it was compiled from newer or relatively less popular languages like Rust or Haskell, where the state of currently publicly available decompilers leaves much to be desired. Angr&#39;s ability to perform symbolic execution therefore allows us to blackbox certain functionality within the program (or even the entire program) by attempting to find the right input for a desired output.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/breaking-cmu-bomblab-with-angr-for-fun-and-profit-part-1/";
        
      },
    },{id: "post-uiuctf20-accounting-accidents-pwn-writeup",
      
        title: "UIUCTF20 Accounting Accidents Pwn Writeup",
      
      description: "UIUCTF20 was a really fun Animal Crossing themed CTF that ran from July 17-19 2020. While I have not played the game before, I somewhat knew what it was about from watching Youtubers play it, and also from memes about the turnip stock market. Our team PPP came in third place, which went above my expectations as most people playing were relatively new and I am quite happy with the result. Now, on to the writeup!",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/UIUCTF20-accounting-accidents-pwn-write-up/";
        
      },
    },{id: "post-pink-floyd-and-alienation",
      
        title: "Pink Floyd and Alienation",
      
      description: "In this post, I talk about how themes of alienation pervades the music of the English rock band Pink Floyd. Pink Floyd is one of the greatest bands in popular music history, achieving international commercial success and widespread influence in subsequent musical developments. They formed in London in 1965 to humble beginnings as a student group,  and comprises Syd Barrett (guitar and lead vocals), Nick Mason (drums), Roger Waters (bass guitar, vocals), and Richard Wright (keyboard, vocals).",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2020/pink-floyd-and-alienation/";
        
      },
    },{id: "post-persisting-xserver-multimonitor-setup",
      
        title: "Persisting Xserver Multimonitor Setup",
      
      description: "I have been using a multiple monitor setup on my Linux system for a long time, and one thing that always annoyed me was that the default monitor arrangement detected by X Server doesn&#39;t reflect its physical positioning (why should it?). Due to a combination of laziness and the fact that reordering it with nvidia-settings&#39;s X Server Display Configuration takes only a few seconds, I never bothered to find the time to find a proper fix for it. However, the fix is actually incredibly simple.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2019/persisting-xserver-multimonitor-setup/";
        
      },
    },{id: "post-a-new-look",
      
        title: "A New Look",
      
      description: "My Jekyll blog has been on the default minima theme since its inception. It was functional and served its purpose, but as someone who has extensive web development experience having designed and implemented sophisticated web interfaces back when I was working at Saleswhale, I felt like I could definitely do much better. This post talks about how I re-themed the blog.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2019/a-new-look/";
        
      },
    },{id: "post-dev-cheat-sheets",
      
        title: "Dev Cheat Sheets",
      
      description: "Today I created a new tab for my blog, &#39;Dev Cheat Sheets&#39;. This page will contain common commmand line commands that I use often. I have a habit of simply performing a reverse-i-search on my terminal when I need a particular command, and therefore I have never really bothered to firmly commit them to memory or find a proper place to record them. However, this gave me problems when...",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2019/dev-cheat-sheets/";
        
      },
    },{id: "post-my-involvement-in-the-autolab-project",
      
        title: "My Involvement in the Autolab Project",
      
      description: "I joined the Autolab team at the close of the semester this year. The Autolab team builds and maintains Autolab, an autograding platform for programming assignments that is currently being used by around 20 universities around the world. It is used very extensively in most computer science courses to grade programming assignments in Carnegie Mellon. In this post, I will share how I came to join Autolab, and the role that I am planning to play in it.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2019/my-involvement-in-the-autolab-project/";
        
      },
    },{id: "post-ssh-to-andrew-file-system-afs-without-credentials",
      
        title: "SSH to Andrew File System (AFS) Without Credentials",
      
      description: "This is the first part of a 2-part guide on how you can SSH to the Andrew File System without keying in your credentails, as well as mounting the Andrew File System (AFS) locally on your Linux machine. This is highly useful if you are a student or faculty member from one of the many universities around the world whose computing systems runs on AFS, such as CMU, MIT, Stanford, Caltech, to list a few.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/ssh-to-AFS-without-credentials/";
        
      },
    },{id: "post-notes-on-founding-a-startup-to-my-future-self",
      
        title: "Notes On Founding A Startup To My Future Self",
      
      description: "This post is addressed to my future self, who (hopefully) would found a startup of his own someday. This is my last week at Saleswhale and I would like to pack as much wisdom into this post from my experiences there and from the wonderful people that I have talked to. It is a very personal post for me and it speaks straight from my heart. To be honest, I teared up a few times while writing this. I hope you will feel the sincerity in my words, and I hope it will inspire you.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/notes-on-founding-a-startup-to-my-future-self/";
        
      },
    },{id: "post-leveraging-the-power-law",
      
        title: "Leveraging The Power Law",
      
      description: "It was three years ago that I was introduced the concept of the &quot;power law&quot; in Peter Thiel&#39;s book, Zero to One. Thiel states that &quot;We don&#39;t live in a normal world, we live under a power law. Exponential equations describe severely unequal distributions&quot;. This, in effect, is very similar to the 80/20 rule that I discussed in my previous post, and observations of such a distribution can be found ranging from income distributions to the fact that the best venture capital fund investments grow to a value that eclipses the rest of their portfolios combined.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/leveraging-the-power-law/";
        
      },
    },{id: "post-lessons-from-saleswhale-about-startup-culture",
      
        title: "Lessons From Saleswhale About Startup Culture",
      
      description: "08 July marks 8 months since I started working in Saleswhale (08 November 2017), and also the last month that I will be here before enrolling in college. It is an opportune time to look back and reflect on my growth as a software engineer and to collate the lessons that I have learned so that I may remember them in my future endeavors. Actually, given the stark contrast between academia and industry, I am quite certain that I will gradually lose sense of the things that I learned, especially those regarding working in teams and software engineering best practices.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/lessons-from-saleswhale-about-startup-culture/";
        
      },
    },{id: "post-ruthless-prioritization-is-the-right-way-to-manage-your-time",
      
        title: "Ruthless Prioritization Is The Right Way To Manage Your Time",
      
      description: "Last week I had the good fortune of having a one-on-one session with Gabriel, the CEO of Saleswhale.  As I rarely have the chance to go on a one-on-one with Gabriel I decided to ask him questions more pertinent to his role as a CEO. This has been something that I have been very curious to learn more about after my previous one-on-one with Ethan, who shared with me his experiences, challenges, and many thoughts as the CTO. In this post, I will share what I feel is probably the most important takeaway for me during the session with Gabriel.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/ruthless-prioritization-is-the-right-way-to-manage-your-time/";
        
      },
    },{id: "post-creating-a-deploy-script-for-jekyll-with-rsync",
      
        title: "Creating A Deploy Script For Jekyll With Rsync",
      
      description: "Static site generators like Jekyll makes it easy to write and build static websites. However, this still comes with the problem of a suitable deployment method. I will share about my thought process and the best approach I found for tackling this problem.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/creating-a-deploy-script-for-jekyll-with-rsync/";
        
      },
    },{id: "post-welcome-to-my-blog",
      
        title: "Welcome To My Blog",
      
      description: "It has been a while since I had intended to start my own technical blog. The idea began seriously floating in my mind after my one-on-one with my senior colleague Javier in Saleswhale. Starting a technical blog was one of the valuable pieces of advice that he gave to me, and it was something that he regretted not doing earlier.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2018/welcome-to-my-blog/";
        
      },
    },{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-i-migrated-my-blog-from-jasper2-to-al-folio-as-it-has-better-support-for-non-blog-post-pages",
          title: 'I migrated my blog from jasper2 to al-folio as it has better support...',
          description: "",
          section: "News",},{id: "news-i-wrote-a-new-blog-post-reinforcement-learning-policy-optimization-deriving-the-policy-gradient-update",
          title: 'I wrote a new blog post, Reinforcement Learning Policy Optimization: Deriving the Policy...',
          description: "",
          section: "News",},{id: "news-happy-to-share-that-a-new-post-is-up-a-concise-proof-of-the-central-limit-theorem-and-its-actually-useful-version-the-berry-esseen-theorem",
          title: 'Happy to share that a new post is up, A Concise Proof of...',
          description: "",
          section: "News",},{id: "news-i-wrote-a-new-article-the-art-of-latex-common-mistakes-and-advice-for-typesetting-beautiful-delightful-proofs-which-i-hope-will-be-useful-to-many-students",
          title: 'I wrote a new article The Art of LaTeX: Common Mistakes, and Advice...',
          description: "",
          section: "News",},{id: "news-i-will-be-a-ta-for-10-708-probabilistic-graphical-models-this-spring-taught-by-andrej-risteski",
          title: 'I will be a TA for 10-708 Probabilistic Graphical Models this Spring, taught...',
          description: "",
          section: "News",},{id: "news-i-added-a-new-page-to-curate-cmu-cs-courses-that-have-publicly-available-lecture-videos-and-course-content-available-on-the-cmu-online-tab-hope-this-will-be-helpful-for-many-people-feel-free-to-let-me-know-if-there-are-things-that-i-missed-which-would-be-good-to-add-via-email",
          title: 'I added a new page to curate CMU CS courses that have publicly...',
          description: "",
          section: "News",},{id: "news-quot-the-art-of-latex-common-mistakes-and-advice-for-typesetting-beautiful-delightful-proofs-quot-made-it-onto-the-front-page-of-hacker-news-on-01-08-2023",
          title: '&amp;quot;The Art of LaTeX: Common Mistakes, and Advice for Typesetting Beautiful, Delightful Proofs&amp;quot;...',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/latex-hn/";
            },},{id: "news-i-earned-my-master-of-science-in-computer-science-degree-from-carnegie-mellon-university",
          title: 'I earned my Master of Science in Computer Science degree from Carnegie Mellon...',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/mscs-graduation/";
            },},{id: "news-a-new-blog-post-on-score-based-diffusion-models-written-in-collaboration-with-owen-wang",
          title: 'A new blog post on Score-Based Diffusion Models, written in collaboration with Owen...',
          description: "",
          section: "News",},{id: "news-i-really-enjoyed-15-712-last-semester-and-wrote-a-course-review-cmu-15-712-advanced-operating-systems-and-distributed-systems-course-review",
          title: 'I really enjoyed 15-712 last semester and wrote a course review: CMU 15-712...',
          description: "",
          section: "News",},{id: "news-i-updated-an-old-blog-post-against-government-scholarships-with-some-new-content-and-perspectives",
          title: 'I updated an old blog post Against Government Scholarships with some new content...',
          description: "",
          section: "News",},{id: "news-new-blog-post-about-the-infamous-cmu-steam-tunnels",
          title: 'New blog post about the infamous CMU steam tunnels',
          description: "",
          section: "News",},{id: "news-i-wrote-a-guided-walkthrough-of-high-dimensional-analysis-of-m-estimators-with-decomposable-regularizers-based-on-this-paper-by-negahban-et-al",
          title: 'I wrote a guided walkthrough of high-dimensional analysis of M-estimators with decomposable regularizers...',
          description: "",
          section: "News",},{id: "news-i-hosted-an-ama-session-on-applying-to-and-studying-cs-in-the-us-for-a-hci-ecg-event-the-slides-can-be-found-here",
          title: 'I hosted an AMA session on applying to and studying CS in the...',
          description: "",
          section: "News",},{id: "news-created-a-new-paper-summaries-tab-for-keeping-track-of-my-academic-reading-and-to-also-hold-myself-accountable-to-read-consistently",
          title: 'Created a new paper summaries tab for keeping track of my academic reading...',
          description: "",
          section: "News",},{id: "news-migrated-all-image-files-from-avif-to-webp-format-as-i-was-recently-informed-by-a-friend-that-edge-does-not-yet-support-avif-my-apologies-if-you-had-issues-viewing-images-in-the-past",
          title: 'Migrated all image files from .avif to .webp format, as I was recently...',
          description: "",
          section: "News",},{id: "news-wrote-a-pretty-interesting-and-accessible-summary-for-a-recent-paper-generative-agents-interactive-simulacra-of-human-behavior",
          title: 'Wrote a pretty interesting and accessible summary for a recent paper: Generative Agents:...',
          description: "",
          section: "News",},{id: "news-started-full-time-as-a-linux-engineer-at-jane-street",
          title: 'Started full-time as a Linux Engineer at Jane Street!',
          description: "",
          section: "News",},{id: "news-wrote-one-of-my-more-interesting-paper-summaries-efficiently-modeling-long-sequences-with-structured-state-spaces",
          title: 'Wrote one of my more interesting paper summaries: Efficiently Modeling Long Sequences with...',
          description: "",
          section: "News",},{id: "news-wrote-a-post-on-creating-trackback-requests-manually-for-static-sites-motivated-by-my-own-usage",
          title: 'Wrote a post on creating Trackback requests manually for static sites, motivated by...',
          description: "",
          section: "News",},{id: "news-wrote-a-tutorial-on-setting-up-the-japanese-arcade-rhythm-game-sound-voltex-at-home",
          title: 'Wrote a tutorial on setting up the Japanese arcade rhythm game Sound Voltex...',
          description: "",
          section: "News",},{id: "news-wrote-a-pretty-interesting-summary-with-high-level-proof-sketches-for-the-implicit-bias-of-gradient-descent-on-separable-data",
          title: 'Wrote a pretty interesting summary with high-level proof sketches for The Implicit Bias...',
          description: "",
          section: "News",},{id: "news-read-a-really-interesting-paper-on-image-translation-via-diffusion-models-this-weekend-and-wrote-a-more-detailed-than-usual-summary-for-it-zero-shot-image-to-image-translation",
          title: 'Read a really interesting paper on image translation via diffusion models this weekend...',
          description: "",
          section: "News",},{id: "news-i-m-really-excited-to-be-joining-the-ai-assistants-team-at-jane-street-to-work-on-large-language-models",
          title: 'I’m really excited to be joining the AI Assistants team at Jane Street...',
          description: "",
          section: "News",},{id: "news-i-wrote-my-50th-paper-summary-on-this-blog-today-with-reconciling-modern-machine-learning-practice-and-the-bias-variance-trade-off-just-a-little-over-a-year-from-when-i-published-my-first-summary",
          title: 'I wrote my 50th paper summary on this blog today with Reconciling modern...',
          description: "",
          section: "News",},{id: "news-after-a-year-of-training-and-preparation-a-night-of-bad-sleep-on-camp-muir-and-tons-of-excitement-and-adrenaline-i-summited-tahoma-mt-rainier-in-clear-skies-and-beautiful-weather",
          title: 'After a year of training and preparation, a night of bad sleep on...',
          description: "",
          section: "News",},{id: "news-i-will-be-at-neurips-from-12-10-12-15-let-s-chat-if-you-re-also-there",
          title: 'I will be at NeurIPs from 12/10-12/15. Let’s chat if you’re also there!...',
          description: "",
          section: "News",},{id: "news-wrote-on-the-beautiful-connection-between-how-long-a-markov-chain-takes-to-mix-commonly-used-in-mcmc-methods-in-ml-and-the-spectral-gap-of-its-transition-matrix-bounding-mixing-times-of-markov-chains-via-the-spectral-gap",
          title: 'Wrote on the beautiful connection between how long a Markov Chain takes to...',
          description: "",
          section: "News",},{id: "projects-graphical-bayesian-networks-for-predicting-asset-covariances",
          title: 'Graphical Bayesian Networks for Predicting Asset Covariances',
          description: "Covariance matrix prediction is a long-standing challenge in modern portfolio theory and quantitative finance. In this project, we investigate the effectiveness of Bayesian networks in predicting the covariance matrix of financial assets (specifically a subset of the S&amp;P 500), evaluated against Heterogeneous Autoregressive (HAR) models.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/bayesian_networks_covariance_prediction_project/";
            },},{id: "projects-improving-domain-adaptation-of-transformer-models-for-generating-reddit-comments",
          title: 'Improving Domain Adaptation of Transformer Models For Generating Reddit Comments',
          description: "We improve upon the recent success of large language models based on the transformer architecture by investigating and showing several methods that have empirically improved its performance in domain adaptation. We use a pre-trained GPT-2 model and perform fine-tuning on 5 different subreddits, and use different methods of ordering the training data based on our priors about the input to see how this affects the prediction quality of the trained model. We propose a new metric for evaluating causal language modeling tasks called APES (Average Perplexity Evaluation for Sentences) to address the limitations of existing metrics, and apply them to our results. Our results are evaluated against both LSTM and GPT-2 baselines.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/improving_domain_adaptation_of_transformer_models_for_generating_reddit_commentsproject/";
            },},{id: "projects-pseudo-determinism-for-graph-streaming-problems",
          title: 'Pseudo-determinism for Graph Streaming Problems',
          description: "Given a fixed input for a search problem, pseudo-deterministic algorithms produce the same answer over multiple independent runs, with high probability. For example, we can efficiently find a certificate for inequality of multivariate polynomials pseudo-deterministically, but it is not known how to do so deterministically. The same notion can be extended to the streaming model. The problem of finding a nonzero element from a turnstile stream is previously shown to require linear space for both deterministic and pseudo-deterministic algorithms. Another model of streaming problems is that of graphs, where edge insertions and deletions occur along a stream. Some natural problems include connectivity, bipartiteness, and colorability of a graph. While the randomized and deterministic graph streaming algorithms have been mostly well-studied, we investigate pseudo-deterministic space lower bounds and upper bounds for graph theoretic streaming problems.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/pseudodeterminism_for_graph_streaming_problems/";
            },},{id: "projects-analysis-of-symmetry-and-conventions-in-off-belief-learning-in-hanabi",
          title: 'Analysis of Symmetry and Conventions in Off-Belief Learning in Hanabi',
          description: "Hanabi has been proposed as the new frontier for developing strategies in cooperative AI, currently a very nascent area of AI research. A recent algorithm that has been developed for multi-agent reinforcement learning in a cooperative context is the Off-Belief Learning (OBL) algorithm, which is based on iterated reasoning starting from a base policy. We investigate if policies learnt by agents using the OBL algorithm in the multi-player cooperative game Hanabi in the zero-shot coordination (ZSC) context are invariant across symmetries of the game, and if any conventions formed during training are arbitrary or natural, both of which are desirable properties.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/symmetry_and_conventions_in_obl_in_hanabi_project/";
            },},{id: "summaries-deep-contextualized-word-representations-elmo",
          title: 'Deep contextualized word representations (ELMo)',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-02-deep-contextualized-word-representations/";
            },},{id: "summaries-foundations-and-trends-in-multimodal-machine-learning-principles-challenges-and-open-questions",
          title: 'Foundations and Trends in Multimodal Machine Learning: Principles,  Challenges, and Open Questions',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-02-foundations-and-trends-in-multimodal-machine-learning-principles--challenges-and-open-questions/";
            },},{id: "summaries-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding",
          title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-03-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/";
            },},{id: "summaries-chain-of-thought-prompting-elicits-reasoning-in-large-language-models",
          title: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-03-chain-of-thought-prompting-elicits-reasoning-in-large-language-models/";
            },},{id: "summaries-training-language-models-to-follow-instructions-with-human-feedback-instructgpt",
          title: 'Training language models to follow instructions with human feedback (InstructGPT)',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-05-training-language-models-to-follow-instructions-with-human-feedback/";
            },},{id: "summaries-evaluating-large-language-models-trained-on-code-codex",
          title: 'Evaluating Large Language Models Trained on Code (Codex)',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-06-evaluating-large-language-models-trained-on-code/";
            },},{id: "summaries-bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-translation-and-comprehension",
          title: 'BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-09-bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-translation-and-comprehension/";
            },},{id: "summaries-dense-passage-retrieval-for-open-domain-question-answering",
          title: 'Dense Passage Retrieval for Open-Domain Question Answering',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-10-dense-passage-retrieval-for-open-domain-question-answering/";
            },},{id: "summaries-language-models-are-unsupervised-multitask-learners-gpt-2",
          title: 'Language Models are Unsupervised Multitask Learners (GPT-2)',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-10-language-models-are-unsupervised-multitask-learners/";
            },},{id: "summaries-simple-synthetic-data-reduces-sycophancy-in-large-language-models",
          title: 'Simple synthetic data reduces sycophancy in large language models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-10-simple-synthetic-data-reduces-sycophancy-in-large-language-models/";
            },},{id: "summaries-generative-agents-interactive-simulacra-of-human-behavior",
          title: 'Generative Agents: Interactive Simulacra of Human Behavior',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-11-generative-agents-interactive-simulacra-of-human-behavior/";
            },},{id: "summaries-improving-language-understanding-by-generative-pre-training-gpt",
          title: 'Improving Language Understanding by Generative Pre-Training (GPT)',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-11-improving-language-understanding-by-generative-pre-training/";
            },},{id: "summaries-metagpt-meta-programming-for-multi-agent-collaborative-framework",
          title: 'MetaGPT: Meta Programming for Multi-Agent Collaborative Framework',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-11-metagpt-meta-programming-for-multi-agent-collaborative-framework/";
            },},{id: "summaries-transformers-in-speech-processing-a-survey",
          title: 'Transformers in Speech Processing: A Survey',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-19-transformers-in-speech-processing-a-survey/";
            },},{id: "summaries-accurate-detection-of-wake-word-start-and-end-using-a-cnn",
          title: 'Accurate Detection of Wake Word Start and End Using a CNN',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-22-accurate-detection-of-wake-word-start-and-end-using-a-cnn/";
            },},{id: "summaries-exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer",
          title: 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-22-exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer/";
            },},{id: "summaries-efficiently-modeling-long-sequences-with-structured-state-spaces",
          title: 'Efficiently Modeling Long Sequences with Structured State Spaces',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-25-efficiently-modeling-long-sequences-with-structured-state-spaces/";
            },},{id: "summaries-a-watermark-for-large-language-models",
          title: 'A Watermark for Large Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-27-a-watermark-for-large-language-models/";
            },},{id: "summaries-extracting-training-data-from-large-language-models",
          title: 'Extracting Training Data from Large Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-08-29-extracting-training-data-from-large-language-models/";
            },},{id: "summaries-loss-landscapes-and-optimization-in-over-parameterized-non-linear-systems-and-neural-networks",
          title: 'Loss Landscapes and Optimization in Over-Parameterized Non-Linear Systems and Neural Networks',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-09-04-loss-landscapes-and-optimization-in-over-parameterized-non-linear-systems-and-neural-networks/";
            },},{id: "summaries-gradient-descent-provably-optimizes-over-parameterized-neural-networks",
          title: 'Gradient Descent Provably Optimizes Over-parameterized Neural Networks',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-09-07-gradient-descent-provably-optimizes-over-parameterized-neural-networks/";
            },},{id: "summaries-the-implicit-bias-of-gradient-descent-on-separable-data",
          title: 'The Implicit Bias of Gradient Descent on Separable Data',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-09-09-the-implicit-bias-of-gradient-descent-on-separable-data/";
            },},{id: "summaries-understanding-deep-learning-requires-rethinking-generalization",
          title: 'Understanding Deep Learning Requires Rethinking Generalization',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-09-10-understanding-deep-learning-requires-rethinking-generalization/";
            },},{id: "summaries-calibrate-before-use-improving-few-shot-performance-of-language-models",
          title: 'Calibrate Before Use: Improving Few-Shot Performance of Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-09-23-calibrate-before-use-improving-few-shot-performance-of-language-models/";
            },},{id: "summaries-rethinking-the-role-of-demonstrations-what-makes-in-context-learning-work",
          title: 'Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-09-24-rethinking-the-role-of-demonstrations-what-makes-in-context-learning-work/";
            },},{id: "summaries-repository-level-prompt-generation-for-large-language-models-of-code",
          title: 'Repository-Level Prompt Generation for Large Language Models of Code',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-09-26-repository-level-prompt-generation-for-large-language-models-of-code/";
            },},{id: "summaries-on-the-dangers-of-stochastic-parrots-can-language-models-be-too-big",
          title: 'On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-10-04-on-the-dangers-of-stochastic-parrots-can-language-models-be-too-big/";
            },},{id: "summaries-an-image-is-worth-one-word-personalizing-text-to-image-generation-using-textual-inversion",
          title: 'An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-10-15-an-image-is-worth-one-word-personalizing-text-to-image-generation-using-textual-inversion/";
            },},{id: "summaries-instructpix2pix-learning-to-follow-image-editing-instructions",
          title: 'InstructPix2Pix: Learning to Follow Image Editing Instructions',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-10-17-instructpix2pix-learning-to-follow-image-editing-instructions/";
            },},{id: "summaries-zero-shot-image-to-image-translation",
          title: 'Zero-shot Image-to-Image Translation',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-10-22-zero-shot-image-to-image-translation/";
            },},{id: "summaries-universal-and-transferable-adversarial-attacks-on-aligned-language-models",
          title: 'Universal and Transferable Adversarial Attacks on Aligned Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-10-31-universal-and-transferable-adversarial-attacks-on-aligned-language-models/";
            },},{id: "summaries-high-resolution-image-synthesis-with-latent-diffusion-models",
          title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-11-03-high-resolution-image-synthesis-with-latent-diffusion-models/";
            },},{id: "summaries-large-language-models-for-software-engineering-survey-and-open-problems",
          title: 'Large Language Models for Software Engineering: Survey and Open Problems',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2023-12-12-large-language-models-for-software-engineering-survey-and-open-problems/";
            },},{id: "summaries-matryoshka-representation-learning",
          title: 'Matryoshka Representation Learning',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-02-19-matryoshka-representation-learning/";
            },},{id: "summaries-colbert-efficient-and-effective-passage-search-via-contextualized-late-interaction-over-bert",
          title: 'ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-02-22-colbert-efficient-and-effective-passage-search-via-contextualized-late-interaction-over-bert/";
            },},{id: "summaries-dspy-compiling-declarative-language-model-calls-into-self-improving-pipelines",
          title: 'DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-03-15-dspy-compiling-declarative-language-model-calls-into-self-improving-pipelines/";
            },},{id: "summaries-scaling-laws-for-neural-language-models",
          title: 'Scaling Laws for Neural Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-03-23-scaling-laws-for-neural-language-models/";
            },},{id: "summaries-training-compute-optimal-large-language-models",
          title: 'Training Compute-Optimal Large Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-03-23-training-compute-optimal-large-language-models/";
            },},{id: "summaries-longrope-extending-llm-context-window-beyond-2-million-tokens",
          title: 'LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-03-27-longrope-extending-llm-context-window-beyond-2-million-tokens/";
            },},{id: "summaries-lora-low-rank-adaptation-of-large-language-models",
          title: 'LoRA: Low-Rank Adaptation of Large Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-03-31-lora-low-rank-adaptation-of-large-language-models/";
            },},{id: "summaries-prefix-tuning-optimizing-continuous-prompts-for-generation",
          title: 'Prefix-Tuning: Optimizing Continuous Prompts for Generation',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-04-26-prefix-tuning-optimizing-continuous-prompts-for-generation/";
            },},{id: "summaries-large-language-models-as-optimizers",
          title: 'Large Language Models as Optimizers',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-07-13-large-language-models-as-optimizers/";
            },},{id: "summaries-optimizing-instructions-and-demonstrations-for-multi-stage-language-model-programs",
          title: 'Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-07-14-optimizing-instructions-and-demonstrations-for-multi-stage-language-model-programs/";
            },},{id: "summaries-weak-to-strong-generalization-eliciting-strong-capabilities-with-weak-supervision",
          title: 'Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-07-21-weak-to-strong-generalization-eliciting-strong-capabilities-with-weak-supervision/";
            },},{id: "summaries-constitutional-ai-harmlessness-from-ai-feedback",
          title: 'Constitutional AI: Harmlessness from AI Feedback',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-07-23-constitutional-ai-harmlessness-from-ai-feedback/";
            },},{id: "summaries-distilling-the-knowledge-in-a-neural-network",
          title: 'Distilling the Knowledge in a Neural Network',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-07-25-distilling-the-knowledge-in-a-neural-network/";
            },},{id: "summaries-editing-a-classifier-by-rewriting-its-prediction-rules",
          title: 'Editing a classifier by rewriting its prediction rules',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-07-27-editing-a-classifier-by-rewriting-its-prediction-rules/";
            },},{id: "summaries-domain-adjusted-regression-or-erm-may-already-learn-features-sufficient-for-out-of-distribution-generalization",
          title: 'Domain-Adjusted Regression or: ERM May Already Learn Features Sufficient for Out-of-Distribution Generalization',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-07-28-domain-adjusted-regression-or-erm-may-already-learn-features-sufficient-for-out-of-distribution-generalization/";
            },},{id: "summaries-deep-double-descent-where-bigger-models-and-more-data-hurt",
          title: 'Deep Double Descent: Where Bigger Models and More Data Hurt',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-08-05-deep-double-descent-where-bigger-models-and-more-data-hurt/";
            },},{id: "summaries-reconciling-modern-machine-learning-practice-and-the-bias-variance-trade-off",
          title: 'Reconciling modern machine learning practice and the bias-variance trade-off',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-08-05-reconciling-modern-machine-learning-practice-and-the-bias-variance-trade-off/";
            },},{id: "summaries-lift-yourself-up-retrieval-augmented-text-generation-with-self-memory",
          title: 'Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-09-22-lift-yourself-up-retrieval-augmented-text-generation-with-self-memory/";
            },},{id: "summaries-query-rewriting-for-retrieval-augmented-large-language-models",
          title: 'Query Rewriting for Retrieval-Augmented Large Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-09-22-query-rewriting-for-retrieval-augmented-large-language-models/";
            },},{id: "summaries-dense-x-retrieval-what-retrieval-granularity-should-we-use",
          title: 'Dense X Retrieval: What Retrieval Granularity Should We Use?',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-09-23-dense-x-retrieval-what-retrieval-granularity-should-we-use/";
            },},{id: "summaries-precise-zero-shot-dense-retrieval-without-relevance-labels-hyde",
          title: 'Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-09-27-precise-zero-shot-dense-retrieval-without-relevance-labels/";
            },},{id: "summaries-inpars-data-augmentation-for-information-retrieval-using-large-language-models",
          title: 'InPars: Data Augmentation for Information Retrieval using Large Language Models',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-09-29-inpars-data-augmentation-for-information-retrieval-using-large-language-models/";
            },},{id: "summaries-lost-in-the-middle-how-language-models-use-long-contexts",
          title: 'Lost in the Middle: How Language Models Use Long Contexts',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-09-29-lost-in-the-middle-how-language-models-use-long-contexts/";
            },},{id: "summaries-open-source-large-language-models-are-strong-zero-shot-query-likelihood-models-for-document-ranking",
          title: 'Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking...',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-02-open-source-large-language-models-are-strong-zero-shot-query-likelihood-models-for-document-ranking/";
            },},{id: "summaries-asqa-factoid-questions-meet-long-form-answers",
          title: 'ASQA: Factoid Questions Meet Long-Form Answers',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-03-asqa-factoid-questions-meet-long-form-answers/";
            },},{id: "summaries-enhancing-retrieval-augmented-large-language-models-with-iterative-retrieval-generation-synergy",
          title: 'Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-03-enhancing-retrieval-augmented-large-language-models-with-iterative-retrieval-generation-synergy/";
            },},{id: "summaries-interleaving-retrieval-with-chain-of-thought-reasoning-for-knowledge-intensive-multi-step-questions",
          title: 'Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-03-interleaving-retrieval-with-chain-of-thought-reasoning-for-knowledge-intensive-multi-step-questions/";
            },},{id: "summaries-generate-rather-than-retrieve-large-language-models-are-strong-context-generators",
          title: 'Generate rather than Retrieve: Large Language Models are Strong Context Generators',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-05-generate-rather-than-retrieve-large-language-models-are-strong-context-generators/";
            },},{id: "summaries-unsupervised-dense-information-retrieval-with-contrastive-learning",
          title: 'Unsupervised Dense Information Retrieval with Contrastive Learning',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-07-unsupervised-dense-information-retrieval-with-contrastive-learning/";
            },},{id: "summaries-easy-to-hard-generalization-scalable-alignment-beyond-human-supervision",
          title: 'Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-12-easy-to-hard-generalization-scalable-alignment-beyond-human-supervision/";
            },},{id: "summaries-training-language-models-to-self-correct-via-reinforcement-learning",
          title: 'Training Language Models to Self-Correct via Reinforcement Learning',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-10-12-training-language-models-to-self-correct-via-reinforcement-learning/";
            },},{id: "summaries-ragas-automated-evaluation-of-retrieval-augmented-generation",
          title: 'RAGAS: Automated Evaluation of Retrieval Augmented Generation',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2024-11-04-ragas-automated-evaluation-of-retrieval-augmented-generation/";
            },},{id: "summaries-bge-m3-embedding-multi-lingual-multi-functionality-multi-granularity-text-embeddings-through-self-knowledge-distillation",
          title: 'BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation',
          description: "",
          section: "Summaries",handler: () => {
              window.location.href = "/summaries/2025-01-06-bge-m3-embedding-multi-lingual-multi-functionality-multi-granularity-text-embeddings-through-self-knowledge-distillation/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%66%7A%65%6E%67@%61%6C%75%6D%6E%69.%63%6D%75.%65%64%75", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/fanpu", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/FanPu_Zeng", "_blank");
        },
      },];
