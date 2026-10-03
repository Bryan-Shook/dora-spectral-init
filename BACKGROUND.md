# Background: spectral initialization and DoRA

This guide explains the ideas needed to read our [proposal](README.md) and [test plan](TEST_PLAN.md). It assumes basic familiarity with vectors, matrices, and gradients. Paper links point to specific versions and sections; the reference IDs match the test plan. Sources checked **2026-10-03**.

**Our question:** after separately tuning learning rate and initialization scale, does spectral initialization help more under exact DoRA than under detached DoRA—or vice versa?

The central idea is that **identical starting predictions do not imply identical learning behavior**. Two adapters can initially reconstruct the same pretrained weights while their trainable factors transform gradients differently. DoRA adds another choice: whether gradients pass through its normalization denominator.

Read Sections 1–4 for initialization, Sections 5–7 for learning dynamics, and Sections 8–10 for the papers' implications and our experiment. Numerical examples below are constructed illustrations, not reported experimental results.

## 1. What are we adapting?

A language model contains many linear transformations. For one such layer, write

$$
y = W_0x+b.
$$

Here, $x$ is an input vector, $y$ an output vector, $W_0$ the pretrained weight matrix, and $b$ a bias. Fine-tuning changes model parameters to reduce a task loss: a numerical measure of prediction error.

Full fine-tuning trains all model weights. **Parameter-efficient fine-tuning (PEFT)** trains a smaller set of parameters while keeping most pretrained parameters frozen. Frozen weights still participate in computation; fewer trainable parameters do not eliminate the cost of running the backbone.

We use the following notation throughout:

| Symbol | Shape or meaning |
| --- | --- |
| $W_0$ | Original weight, $d_{out}\times d_{in}$ |
| $r$ | Adapter rank, usually much smaller than either layer dimension |
| $B$, $A$ | Trainable factors of shapes $d_{out}\times r$ and $r\times d_{in}$ |
| $W_{eff}$ | Weight actually used to produce the layer's output |
| $\mathcal L$ | Training loss |
| $\eta$ | Learning rate: controls optimizer step size |
| $\lVert z\rVert_2$ | Length of a vector |
| $\lVert M\rVert_F$ | Frobenius norm: square root of the sum of squared matrix entries |

Our orientation matches PyTorch's stored linear weights. Some papers transpose the weights or exchange the letters $A$ and $B$; compare shapes and products rather than letters alone.

## 2. LoRA: learn an adjustment through a small bottleneck

LoRA freezes $W_0$ and trains two smaller matrices:

$$
W_{eff}=W_0+sBA,\qquad s=\alpha/r.
$$

The adapter first maps the input into $r$ coordinates through $A$, then maps those coordinates to the output through $B$. Its adjustment has rank at most $r$: it can express at most $r$ independent output directions at a given moment. A **subspace** is the set of all linear combinations of a set of directions.

For a $1024\times1024$ weight and $r=8$, the factors have $8(1024+1024)=16{,}384$ parameters, versus $1{,}048{,}576$ entries in the full weight. That is 64 times fewer trainable parameters for this matrix; training speed does not follow that ratio.

Standard LoRA initializes one factor randomly and the other to zero, so the initial adjustment is zero. The original paper uses random Gaussian $A$ and zero $B$; our standard baseline uses Kaiming-uniform $A$ and zero $B$. Both preserve the initial model. Initializing **both** factors to zero would also make both factor gradients zero. [R7, §4.1](https://arxiv.org/html/2106.09685v2#S4.SS1)

The factors remain trainable: their subspaces can move. Initializing in a particular subspace does not permanently restrict training to it.

For the rest of this guide, take **$s=1$**, as in our plan.

## 3. SVD and PiSSA: choose the starting directions from the weights

### SVD separates directions from their strengths

The singular value decomposition (SVD) expresses a matrix as

$$
W_0=U\Sigma V^T=\sum_i\sigma_i u_i v_i^T.
$$

The vectors $v_i$ are perpendicular unit input directions, $u_i$ are perpendicular unit output directions, and $\sigma_i\geq0$ are their strengths, called singular values. In particular, $W_0v_i=\sigma_i u_i$: the layer maps one distinguished input direction to an output direction and scales its length.

The top $r$ components have the largest singular values. They give the best rank-$r$ approximation in Frobenius norm. This describes the pretrained matrix; it does **not** establish which directions are most useful for a particular downstream task.

### PiSSA makes the principal component trainable

Using the top-$r$ components, PiSSA's construction can be written in our notation as

$$
B_0=U_r\Sigma_r^{1/2},\qquad
A_0=\Sigma_r^{1/2}V_r^T,\qquad
W_{res}=W_0-B_0A_0.
$$

It freezes $W_{res}$ and trains $A,B$. Initially,

$$
W_{res}+B_0A_0=W_0.
$$

Thus, the trainable adapter starts by representing a principal component of the pretrained weight, while the residual completes the original matrix. The square root splits each singular value evenly between the factors. [R2, §3, Eqs. 2–5](https://arxiv.org/html/2404.02948v4#S3)

PiSSA reports gains over its LoRA baselines. Our question requires an additional control: its initialization changes both the directions and the factor magnitudes, which can both affect learning.

## 4. Our control: change directions while matching singular values

We compare spectral factors with random directions that have the **same factor and product singular values at a shared initialization scale**.

Let $\gamma>0$ control initialization scale. Let $Q_L,Q_R$ contain random orthonormal columns—unit vectors perpendicular to one another. Define

$$
\begin{aligned}
B_{S,0}&=\sqrt{\gamma}\,U_r\Sigma_r^{1/2}, &
A_{S,0}&=\sqrt{\gamma}\,\Sigma_r^{1/2}V_r^T,\\
B_{R,0}&=\sqrt{\gamma}\,Q_L\Sigma_r^{1/2}, &
A_{R,0}&=\sqrt{\gamma}\,\Sigma_r^{1/2}Q_R^T.
\end{aligned}
$$

Here, S means spectral and R means random. Each factor has nonzero singular values $\sqrt{\gamma\sigma_i}$, and each initial product has singular values $\gamma\sigma_i$. The scaled factors themselves generally are not orthonormal.

For **each** initialization, freeze its own correction:

$$
C_0=B_0A_0,\qquad W_{res}=W_0-C_0,\qquad Z_t=W_{res}+B_tA_t.
$$

Consequently, $Z_0=W_0$ in both cases. $C_0$ stays fixed throughout training.

### A two-dimensional example

Take $W_0=\begin{bmatrix}4&0\\0&1\end{bmatrix}$, $r=1$, and $\gamma=1$.

| Quantity | Spectral direction | An alternative direction at 45 degrees |
| --- | --- | --- |
| $B_0$ | $(2,0)^T$ | $(\sqrt2,\sqrt2)^T$ |
| $A_0$ | $(2,0)$ | $(\sqrt2,\sqrt2)$ |
| $B_0A_0$ | $\begin{bmatrix}4&0\\0&0\end{bmatrix}$ | $\begin{bmatrix}2&2\\2&2\end{bmatrix}$ |
| $W_{res}$ | $\begin{bmatrix}0&0\\0&1\end{bmatrix}$ | $\begin{bmatrix}2&-2\\-2&-1\end{bmatrix}$ |

Both factors have singular value 2; both products have singular value and Frobenius norm 4. Both residual-plus-adapter sums equal $W_0$. The second column illustrates changing orientation; actual experiments draw random bases rather than always using 45 degrees.

This control matches starting outputs and spectra. It does not match later gradients, weight steps, or predictions—those are outcomes to measure. It also requires pretrained singular values, so our random control still incurs SVD cost.

One algebraic distinction matters when interpreting standard LoRA: for our additive conditions,

$$
W_{eff,t}-W_0=B_tA_t-B_0A_0
$$

can have rank up to $2r$, whereas standard zero-initialized LoRA's adjustment has rank at most $r$. Equal factor counts therefore do not imply identical sets of possible net adjustments. Both sides of our main S/R comparison use the same nonzero construction; standard LoRA is an additional practical reference.

## 5. Why can identical starting models learn differently?

A gradient says how the loss changes when a parameter changes. The path from a factor to the effective weight determines the gradient that factor receives.

Let $G=\partial\mathcal L/\partial Z$. The chain rule gives

$$
\frac{\partial\mathcal L}{\partial B}=GA^T,\qquad
\frac{\partial\mathcal L}{\partial A}=B^TG.
$$

Even with identical $Z$ and $G$, different factors transform the gradient differently. For intuition, one simultaneous plain gradient-descent step gives

$$
\Delta Z\approx-\eta\left(BB^TG+GA^TA\right),
$$

ignoring terms of order $\eta^2$. The matrices $BB^T$ and $A^TA$ weight different output and input directions of the gradient. Spectral and random initializations rotate these weights differently.

This is a first-order illustration, not the exact update of our planned AdamW optimizer, which also uses gradient history and coordinate-wise scaling.

It helps to distinguish four quantities that are sometimes all called “scale”:

| Quantity | What it controls |
| --- | --- |
| Learning rate $\eta$ | Optimizer step size |
| Adapter multiplier $s=\alpha/r$ | Adapter contribution in the forward computation; fixed to 1 here |
| Initialization scale $\gamma$ | Starting factor/product singular values, with a compensating frozen residual |
| DoRA magnitude $m$ | Learned length parameter for each effective weight row |

Changing one is not generally equivalent to changing another. Even replacing $B,A$ by $cB,A/c$ preserves their product while changing their separate gradients. Our balanced factors avoid introducing that extra imbalance.

## 6. DoRA: learn length and direction separately

For each output row, DoRA uses

$$
w=m\frac{z}{\|z\|_2}=mu,
$$

where $z$ is the corresponding row of $Z=W_{res}+BA$, $u=z/\|z\|_2$ is its unit direction, and $m$ is a trainable scalar initialized to that row's pretrained length. We write the row as a column vector when taking derivatives below.

Intuitively, $z$ determines where the vector points, while $m$ controls its length. For $z=(3,4)^T$, its length is 5 and its unit direction is $(0.6,0.8)^T$. With $m=5$, the effective weight is $(3,4)^T$; with $m=6$, it is $(3.6,4.8)^T$.

Because our initialization gives $Z_0=W_0$, setting $m_i=\|W_{0,i:}\|_2$ preserves every effective row and hence the starting model. This adapts DoRA's normalization to our residual construction. [R1, §4.1, Eq. 5](https://arxiv.org/html/2402.09353v6#S4.SS1)

We normalize **each output row**, over its input coordinates. This corresponds to `weight.norm(dim=1)` for our stored weight orientation.

Multiplying an entire $z$ by a positive constant leaves its normalized direction unchanged. Scaling $BA$ inside $W_{res}+BA$ is a different operation. Therefore, normalization does not make adapter initialization scale or learning rate irrelevant.

## 7. Exact versus detached DoRA: same forward value, different gradient

### What detachment means

Both variants compute the current denominator $\|z\|_2$ on every forward pass. **Exact DoRA** differentiates through it. **Detached DoRA** treats its current value as a constant during backpropagation:

$$
w_E=m\frac{z}{\|z\|_2},\qquad
w_D=m\frac{z}{\operatorname{stopgrad}(\|z\|_2)}.
$$

At the same parameter values, these produce the same output. Detachment does not freeze the denominator across training steps or freeze $m$. DoRA introduces this approximation to reduce training overhead. [R1, §4.3, Eq. 11](https://arxiv.org/html/2402.09353v6#S4.SS3)

### The gradient difference

Let $g=\partial\mathcal L/\partial w$ be the loss gradient at the effective row and $u=z/\|z\|_2$. For a nonzero row, the gradients with respect to $z$ are

$$
g_E=\frac{m}{\|z\|_2}\left(g-u(u^Tg)\right),\qquad
g_D=\frac{m}{\|z\|_2}g.
$$

The term $u(u^Tg)$ is the component of $g$ parallel to $z$. Exact differentiation removes it: changing only the length of $z$ does not change $z/\|z\|_2$. The remaining component lies along the tangent to the sphere of directions. Both variants give $\partial\mathcal L/\partial m=u^Tg$. [R1, §4.2, Eqs. 6–7](https://arxiv.org/html/2402.09353v6#S4.SS2)

For $z=(3,4)^T$, $m=5$, and $g=(1,0)^T$:

$$
u(u^Tg)=(0.36,0.48)^T,\quad
g_E=(0.64,-0.48)^T,\quad
g_D=(1,0)^T.
$$

The exact gradient is perpendicular to $z$ because $3(0.64)+4(-0.48)=0$. These are gradients; a gradient-descent update points in the negative gradient direction.

### Why the difference might depend on initialization

At initialization, all our conditions share $Z_0$, $m_0$, and the starting predictions. Within a gradient rule, the gradient with respect to $Z$ is therefore shared on the same batch. The S/R difference enters through the factors and optimizer, as in Section 5.

A radial difference in a direct update to $z$ disappears to first order after normalization. However, we update low-rank factors, whose transformations can turn that difference into a change in effective direction. AdamW further changes the relationship between raw gradients and weight steps.

This makes the interaction plausible; it does not predict its sign or guarantee an accuracy benefit. “Exact” identifies the derivative of the normalized function, not an empirical performance ranking. Our diagnostics must compare actual effective weight steps and predictions.

For implementation context, the saved [PEFT source snapshot](references/metadata/peft_dora_source_snapshot.py) detaches both the adapter contribution used to compute the norm and the final norm. Enabling `use_dora=True` alone therefore does not establish an exact variant. The detailed checks are in [TEST_PLAN.md, Sections 4 and 6](TEST_PLAN.md#4-exact-definitions-of-the-experimental-conditions).

## 8. What the other reference papers contribute

### DuDe: spectral initialization plus weight decomposition already exists

DuDe combines principal singular factors, a frozen residual, and a learned magnitude with normalized direction. Its gradient analysis also considers differentiation through the normalization. [R3, §§3.2–3.3](https://arxiv.org/html/2505.14367v2#S3.SS2)

**Implication:** combining PiSSA-like initialization with DoRA is already represented in the literature. Our contribution would be evidence about the interaction between initialization and detachment under matched controls and separate tuning. This is a scoped research question, not an exhaustive novelty claim.

### LoRAM: magnitude is a competing explanation

LoRAM analyzes how initialization magnitude and related hyperparameters influence low-rank optimization. It motivates an initialization using a structured orthogonal basis and magnitude scaling, without requiring an SVD. [R4, §§2–3](https://arxiv.org/html/2507.06558v2#S2)

Crucially, its Appendix I limits the claim: magnitude interacts with rank, learning rate, and task gradients; increasing it does not always help. [R4, Appendix I](https://arxiv.org/html/2507.06558v2#A9)

**Implication:** a spectral gain could partly reflect favorable update scale. Match initial spectra and examine later effective movement before attributing gains to pretrained directions. Our random control is not the published LoRAM algorithm and still uses SVD-derived singular values.

### Learning Rate Matters: a method needs its own tuning

This paper re-evaluates LoRA variants across learning rates and reports that separate tuning can substantially narrow previously reported performance gaps in its studied settings. [R5, §§3.2–3.3](https://arxiv.org/html/2602.04998v2#S3.SS2)

**Implication:** a shared learning rate can favor the parameterization that happens to work well at that rate. We give each condition an equal search budget and select using development data. Equal search effort is a practical comparison protocol; it does not prove that each method's global optimum has been found.

### RLPO/RLMO: separate orientation from singular-value scaling

RLPO uses principal right singular vectors for $A_0$; RLMO uses minor ones. Both set $B_0=0$ and make the rows of $A_0$ orthonormal, without multiplying them by the pretrained singular values. [R6, §5.2](https://arxiv.org/html/2606.31813v1#S5.SS2)

The paper studies **reinforcement learning with verifiable rewards (RLVR)**: models generate answers and receive checkable rewards. Our **supervised fine-tuning (SFT)** instead learns from supplied target solutions.

**Implication:** this work motivates distinguishing geometry from scaling. Its theory and experiments do not establish the outcome of our SFT study, and its zero-factor initialization differs from our two nonzero factors.

## 9. How these ideas become a testable hypothesis

The four primary conditions cross two choices:

| | Spectral initialization | Matched random initialization |
| --- | --- | --- |
| Exact DoRA | E-S | E-R |
| Detached DoRA | D-S | D-R |

We also include additive spectral/random controls, A-S and A-R, and standard LoRA, L0. The additive pair helps assess behavior without normalization; L0 supplies a familiar practical baseline.

Let $a$ denote final test accuracy, expressed on a 0–100 scale. Define

$$
\Delta_E=a_{E,S}-a_{E,R},\qquad
\Delta_D=a_{D,S}-a_{D,R},\qquad
I=\Delta_E-\Delta_D.
$$

$I$ is the **interaction**: how much the spectral advantage changes when the gradient rule changes.

For example, hypothetical exact accuracies of 62% and 59% give a 3-percentage-point spectral advantage. Detached accuracies of 60% and 59% give a 1-point advantage. Then $I=2$ percentage points. If both advantages were 3 points, $I=0$, even though spectral initialization helped both variants.

The plan separates four hypotheses:

- **H1, interaction:** detachment changes the spectral advantage after the specified tuning. Either sign is possible.
- **H2, spectral value:** within a rule, spectral initialization retains an advantage after matching and tuning. This can hold even if H1 does not.
- **H3, mechanism:** the factors make detachment affect effective weight steps differently. A gradient difference alone is insufficient evidence for a task-performance effect.
- **H4, cost:** any quality difference must be evaluated alongside initialization, search, training, and evaluation cost.

Track the correct notion of movement:

| Measurement | Meaning |
| --- | --- |
| $\lVert B_tA_t\rVert_F$ | Adapter-product magnitude; it is already nonzero at initialization |
| $\lVert W_{eff,t}-W_0\rVert_F$ | Net change from the original model; initially zero |
| $\lVert W_{eff,t+1}-W_{eff,t}\rVert_F$ | Size of one effective weight step |
| Angle between effective rows | Directional movement, complementing magnitude |

Independently tuned conditions may select different $\gamma$ values. Their final comparison measures performance under our search protocol. Comparisons at shared $(\gamma,\eta)$ are also needed to explain mechanisms under matched hyperparameters.

Repeat every condition with fresh, paired seed blocks and report uncertainty. A wide interval containing zero is inconclusive. The plan's 2-percentage-point practical margin is a proposed decision threshold, not a published effect size or an observed result.

## 10. Model, data, and practical training background

### Why Qwen2.5-1.5B and GSM8K?

The initial plan uses **Qwen2.5-1.5B base**, rank-16 adapters on query and value projections, and **GSM8K**. In attention, query projections help determine which positions a token attends to; value projections transform the information combined from those positions.

The official model card lists 1.54 billion parameters and 28 layers. The Qwen2.5 report describes the model family and architecture; the exact checkpoint card/configuration determines our implementation dimensions. [R8, §2](https://arxiv.org/html/2412.15115v2#S2), [official base-model card](https://huggingface.co/Qwen/Qwen2.5-1.5B)

GSM8K supplies arithmetic word problems and worked solutions. Its official `main` configuration has 7,473 training and 1,319 test examples. Our plan reserves two 500-example validation partitions from training, leaving 6,473 training examples. The GSM8K paper also studies answer verifiers; our experiment uses its dataset with SFT and greedy generation. [R9, §2](https://arxiv.org/html/2110.14168v2#S2), [official dataset](https://huggingface.co/datasets/openai/gsm8k)

During SFT, the model predicts the next solution token given the question and preceding correct solution tokens. The loss penalizes low probabilities assigned to those targets. During evaluation, the model generates its own solution; we score the final numerical answer. Lower token loss and higher answer accuracy are related but distinct outcomes.

### Why the small, repeated experiment?

We have **one H100 80 GB and 100–200 total GPU-hours**, including pilots, tuning, and evaluation. A small model preserves budget for separate searches and repeated comparisons. The pilot must establish learning progress and measure full-run cost; feasibility and expected accuracy are not established by model size alone.

The result will be conditional on the chosen model, task, adapter placement, rank, and search budget. It will not automatically generalize to larger models or RLVR. The public benchmark also does not establish absence from pretraining data.

### Why 3% warmup?

Warmup increases the learning rate from a small initial value before reaching its peak, limiting early steps while optimization begins. The plan's **3% warmup plus cosine decay** has a concrete precedent in *Learning Rate Matters*, Appendix D.3, Table 5. That paper's other settings differ from ours. [R5, Appendix D.3](https://arxiv.org/html/2602.04998v2#A4.SS3)

There is no established reason that exactly 3% is optimal here. It is a provisional recipe choice. With the planned 609 optimizer steps, rounding up $0.03\times609$ gives 19 warmup steps. Any pilot adjustment should be documented and frozen for all main conditions.

## 11. References and where to verify the claims

These are reading pointers, not claims that our protocol reproduces each paper. The pinned arXiv versions make the section references reproducible. Papers remain online links; additional source metadata is indexed in [references/README.md](references/README.md).

| ID | Paper | Focused reading |
| --- | --- | --- |
| R1 | Liu et al. **DoRA: Weight-Decomposed Low-Rank Adaptation**. ICML 2024. [2402.09353v6](https://arxiv.org/abs/2402.09353v6) | [§§4.1–4.3](https://arxiv.org/html/2402.09353v6#S4): normalization, gradient derivation, denominator detachment; Appendix A.2 for the detachment ablation |
| R2 | Meng et al. **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models**. NeurIPS 2024. [2404.02948v4](https://arxiv.org/abs/2404.02948v4) | [§3, Eqs. 2–5](https://arxiv.org/html/2404.02948v4#S3): balanced principal factors and frozen residual |
| R3 | Han et al. **Dual Decomposition of Weights and Singular Value Low Rank Adaptation**. 2025. [2505.14367v2](https://arxiv.org/abs/2505.14367v2) | [§§3.2–3.3](https://arxiv.org/html/2505.14367v2#S3.SS2): spectral initialization with weight decomposition and its gradient |
| R4 | Zhang et al. **The Primacy of Magnitude in Low-Rank Adaptation**. NeurIPS **2025**. [2507.06558v2](https://arxiv.org/abs/2507.06558v2); [publication record](https://papers.nips.cc/paper_files/paper/2025/hash/0010665e949927b74faf6e3ada6d7f72-Abstract-Conference.html) | [§3](https://arxiv.org/html/2507.06558v2#S3): spectral gains and LoRAM; [Appendix I](https://arxiv.org/html/2507.06558v2#A9): limits of the magnitude explanation |
| R5 | Lee et al. **Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning**. 2026. [2602.04998v2](https://arxiv.org/abs/2602.04998v2) | [§3](https://arxiv.org/html/2602.04998v2#S3): separate tuning; [Appendix D.3, Table 5](https://arxiv.org/html/2602.04998v2#A4.SS3): shared recipe and 3% warmup |
| R6 | Zhang et al. **Geometry-Preserving Orthonormal Initialization for Low-Rank Adaptation in RLVR**. 2026. [2606.31813v1](https://arxiv.org/abs/2606.31813v1) | [§5](https://arxiv.org/html/2606.31813v1#S5): assumptions and initialization definitions; [§6](https://arxiv.org/html/2606.31813v1#S6): RLVR experiments |
| R7 | Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022. [2106.09685v2](https://arxiv.org/abs/2106.09685v2) | [§4.1](https://arxiv.org/html/2106.09685v2#S4.SS1): additive factorization, scaling, and zero-factor initialization |
| R8 | Qwen Team. **Qwen2.5 Technical Report**. [2412.15115v2](https://arxiv.org/abs/2412.15115v2) | [§2](https://arxiv.org/html/2412.15115v2#S2): architecture/tokenizer; use the [base checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B) for model-specific details |
| R9 | Cobbe et al. **Training Verifiers to Solve Math Word Problems**. 2021. [2110.14168v2](https://arxiv.org/abs/2110.14168v2) | [§2](https://arxiv.org/html/2110.14168v2#S2): GSM8K; [§4](https://arxiv.org/html/2110.14168v2#S4): distinguish supervised training from verification |

The experiment's central uncertainty remains empirical: after these controls and tuning, does changing the gradient rule change the value of pretrained singular directions?
