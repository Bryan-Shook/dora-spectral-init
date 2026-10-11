# Background quiz: question bank

Based on `../BACKGROUND.md`, as read on 2026-10-11. For a slower explanation before the quiz, read [the beginner companion](../background-beginner.md). This is a diagnostic learning sequence, not an exam. All numerical scenarios are constructed examples. There are 47 core questions across all 11 sections. Answers and teaching notes are in `instructor-guide.md`; the current position is in `progress.md`.

Ask one question per turn. Start with the next unanswered question, assess the learner's reasoning, explain any gap, and then ask the next question. Insert a short follow-up when an essential concept is missing. Do not reveal later answers while presenting the current question. The learner can answer in plain language, say “I don't know,” or request a hint.

| Section | Topic | Questions |
| --- | --- | ---: |
| 1 | Shapes, frozen weights, and norms | 3 |
| 2 | LoRA and zero initialization | 4 |
| 3 | SVD and PiSSA | 5 |
| 4 | Matching spectra while changing directions | 5 |
| 5 | Factor gradients and learning dynamics | 5 |
| 6 | DoRA's length and direction | 4 |
| 7 | Exact and detached gradients | 6 |
| 8 | What the reference papers imply | 4 |
| 9 | Interaction, controls, and uncertainty | 5 |
| 10 | Training and evaluation decisions | 4 |
| 11 | Evidence and reproducibility | 2 |

## Section 1 — What are we adapting?

### S01-Q01 — Matrix shapes

A layer maps an input vector with 4 coordinates to an output vector with 3 coordinates:

\[
y=W_0x+b.
\]

What shape must \(W_0\) have, and how does that shape make the multiplication work?

### S01-Q02 — Frozen does not mean absent

A colleague says, “The pretrained weights are frozen, so training only needs to run the little adapter.” What is wrong with that reasoning?

### S01-Q03 — Vector versus matrix norms

For

\[
z=\begin{bmatrix}3\\4\end{bmatrix},\qquad
M=\begin{bmatrix}3&0\\0&4\end{bmatrix},
\]

calculate the vector length \(\|z\|_2\) and the Frobenius norm \(\|M\|_F\). Do the equal numerical answers mean these objects have the same shape or represent the same operation?

## Section 2 — LoRA

### S02-Q01 — The bottleneck

For a weight with 6 output coordinates, 4 input coordinates, and adapter rank \(r=2\), explain the path \(x\to Ax\to B(Ax)\). State the shapes of \(A\) and \(B\), and compare their total parameter count with the full weight's count.

### S02-Q02 — Rank and moving subspaces

For the same layer, someone claims that rank-2 LoRA permanently restricts its adjustment to the two output directions chosen at initialization. Which part of this claim is right, and which part is wrong?

### S02-Q03 — Why one factor starts at zero

Take \(s=1\) and \(G=\partial\mathcal L/\partial W_{eff}\). The factor gradients are

\[
\nabla_B\mathcal L=GA^T,\qquad \nabla_A\mathcal L=B^TG.
\]

If \(A_0\) is random and \(B_0=0\), which factor can receive a nonzero loss gradient on the first step? What changes if both factors start at zero?

### S02-Q04 — Parameter savings versus speed

A rank-8 adapter for a \(1024\times1024\) weight has 64 times fewer trainable parameters than that full weight. Does this establish 64 times faster training? Explain what the count leaves out.

## Section 3 — SVD and PiSSA

### S03-Q01 — The meaning of a singular triplet

Suppose

\[
W_0v_i=5u_i,\qquad \|v_i\|_2=\|u_i\|_2=1.
\]

What does the layer do to an input \(x=2v_i\)? Explain the separate roles of \(v_i\), \(u_i\), and the singular value 5.

### S03-Q02 — Orthonormal basis coordinates

Consider

\[
v_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\qquad
v_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix},\qquad
x=\begin{bmatrix}2\\0\end{bmatrix}.
\]

Find the coefficients that reconstruct \(x\) from \(v_1,v_2\). Why does taking dot products give the coefficients here?

### S03-Q03 — Why SVD chooses perpendicular outputs

If the perpendicular unit vectors \(v_i,v_j\) are eigenvectors of \(W_0^TW_0\), why are their nonzero outputs \(W_0v_i,W_0v_j\) perpendicular? You may use

\[
(W_0v_i)^T(W_0v_j)=v_i^TW_0^TW_0v_j.
\]

### S03-Q04 — Constructing PiSSA

Let

\[
W_0=\begin{bmatrix}9&0\\0&4\end{bmatrix},\qquad r=1.
\]

Choose the balanced principal factors \(B_0,A_0\) and frozen residual \(W_{res}\). Show that their initial sum reconstructs \(W_0\), and explain the square root in the factors.

### S03-Q05 — Best approximation versus best task direction

Suppose the top singular direction of a pretrained weight gives its best rank-1 approximation. Does that establish that it is the best rank-1 direction for fine-tuning on a new task? Explain the missing connection.

## Section 4 — Matched random control

### S04-Q01 — What is being scaled?

Let \(U_r\) have orthonormal columns and

\[
B_0=\sqrt\gamma\,U_r\Sigma_r^{1/2},\qquad
\gamma=9,\qquad \Sigma_r=\operatorname{diag}(4,1).
\]

What are the lengths of the two columns of \(B_0\)? Are they perpendicular, orthonormal, both, or neither? Identify which object is the scalar.

### S04-Q02 — Same spectrum, different matrix

At the same \(\gamma\), spectral and random factors use the same singular values but different orthonormal directions. Which properties are guaranteed to match: the factor singular values, product singular values, product Frobenius norm, product entries, and factor gradients? Explain where the guarantees stop.

### S04-Q03 — Why each initialization needs its own residual

For

\[
W_0=\begin{bmatrix}4&0\\0&1\end{bmatrix},\quad
B_{R,0}=\begin{bmatrix}\sqrt2\\\sqrt2\end{bmatrix},\quad
A_{R,0}=\begin{bmatrix}\sqrt2&\sqrt2\end{bmatrix},
\]

find the residual that preserves the starting weight. What starting weight would result if the random factors instead used the spectral residual \(\operatorname{diag}(0,1)\)?

### S04-Q04 — Nonzero product versus model movement

With the initialization-specific residual frozen, the additive effective weight is \(W_{eff,t}=W_{res}+B_tA_t\). What expression measures its net change from \(W_0\), and why can that change have rank as large as \(2r\), even though \(B_tA_t\) has rank at most \(r\)?

### S04-Q05 — What the control establishes

Someone says, “Because the random condition matches singular values and starting predictions, it must learn identically and is also a cheap SVD-free baseline.” Assess both parts of the statement.

## Section 5 — Why learning can differ

### S05-Q01 — Reading the chain rule

Given \(Z=W_{res}+BA\) and an upstream gradient \(G\), explain why

\[
\nabla_B\mathcal L=GA^T,\qquad \nabla_A\mathcal L=B^TG.
\]

Check their shapes, using \(B\in\mathbb R^{d_{out}\times r}\) and \(A\in\mathbb R^{r\times d_{in}}\). Why can matching \(Z\) and \(G\) leave the factor gradients different?

### S05-Q02 — One product, two learning speeds

Use a scalar illustration with \(Z=BA\), upstream gradient \(G=1\), and plain gradient descent at \(\eta=0.01\). Compare \((B,A)=(2,2)\) with \((4,1)\): both start at \(Z=4\). Using

\[
\Delta Z\approx-\eta(B^2+A^2)G,
\]

calculate their first-order changes. What does this teach about the parameterization?

### S05-Q03 — Deriving the effective step

For a simultaneous plain gradient-descent step,

\[
B'=B-\eta GA^T,\qquad A'=A-\eta B^TG,
\]

expand \(B'A'-BA\). Which term does the guide drop in its first-order expression, and why is that expression not the exact AdamW update?

### S05-Q04 — Four different scales

Match each operation with \(\eta\), \(s\), \(\gamma\), or \(m\): changing the optimizer step size; multiplying the adapter contribution in the forward computation; changing the starting factor sizes while compensating with a residual; changing the learned length of a DoRA row. Why are these not interchangeable?

### S05-Q05 — Directional weighting

In

\[
\Delta Z\approx-\eta(BB^TG+GA^TA),
\]

what do \(BB^T\) and \(A^TA\) do to the gradient? If spectral and random factors have matched singular values, does that force these matrices to be equal?

## Section 6 — DoRA

### S06-Q01 — Separate length and direction

For

\[
z=\begin{bmatrix}3\\4\end{bmatrix},\qquad m=10,\qquad
w=m\frac{z}{\|z\|_2},
\]

find the unit direction, the effective vector \(w\), and its length. What changes if \(m\) changes but \(z\) stays fixed?

### S06-Q02 — Preserving the original model

Our factors and residual give \(Z_0=W_0\). How should each DoRA magnitude \(m_i\) be initialized to preserve the pretrained effective weight? Which coordinates are included in each row's normalization?

### S06-Q03 — Scaling the whole row versus its adapter

Let \(W_{res}\)'s row be \((1,0)\), and let \(BA\)'s row be \((0,1)\), so \(z=(1,1)\). Compare replacing the entire \(z\) by \(2z\) with replacing only \(BA\) by \(2BA\), holding the residual and \(m\) fixed. Do these operations preserve the same direction?

### S06-Q04 — Forward invariance versus training invariance

For \(c>0\), \(z\) and \(cz\) give the same normalized forward direction. Does this make initialization scale and learning rate irrelevant to DoRA training? Explain using the trainable factors and their gradients.

## Section 7 — Exact versus detached DoRA

### S07-Q01 — What detachment freezes

Exact and detached DoRA use the same current denominator \(\|z\|_2\) in their forward computation. What does detached DoRA hold constant, and during which operation? Is the denominator fixed across training steps?

### S07-Q02 — Compute the two gradients

Let \(z=(3,4)^T\), \(m=5\), and the upstream effective-row gradient be \(g=(1,0)^T\). Compute the component of \(g\) parallel to \(z\), the exact gradient with respect to \(z\), and the detached gradient. Verify the exact gradient is perpendicular to \(z\).

### S07-Q03 — Why normalization removes a component

Explain why the exact gradient removes the part of \(g\) parallel to \(z\). What happens to the two \(z\)-gradients when \(g\) is entirely parallel to \(z\), and can the magnitude \(m\) still receive a gradient?

### S07-Q04 — From raw gradients to actual steps

The exact gradient with respect to \(z\) is tangent to the sphere of directions. Does that guarantee that an AdamW update to \(A,B\) produces a tangent change in \(Z\), or that exact DoRA has better accuracy? Explain the two gaps in reasoning.

### S07-Q05 — Can factors turn a radial gradient into direction change?

For one row, hold \(m\) fixed to isolate the factor update. Let

\[
z=(1,1)^T,\quad m=\sqrt2,\quad g=(1,1)^T,\quad
B=1,\quad A=\begin{bmatrix}1&0\end{bmatrix},\quad
W_{res}=\begin{bmatrix}0&1\end{bmatrix}.
\]

Here the exact \(z\)-gradient is zero, while the detached gradient is radial. Use the factor-update formula to find the detached first-order change in \(Z\). Is that change parallel to the original row? What does the example demonstrate?

### S07-Q06 — Auditing an implementation

You enable a library's DoRA option. What must you inspect or check before calling it the exact-gradient condition? Describe a small gradient check that could distinguish exact from detached normalization.

## Section 8 — Contributions of the other papers

### S08-Q01 — A defensible contribution

According to the guide, DuDe already combines spectral initialization with weight decomposition. How should that change the contribution we claim for this proposed experiment?

### S08-Q02 — Magnitude as an alternative explanation

Suppose a spectral initialization beats a random baseline that uses much smaller factors. Why is “pretrained directions caused the gain” premature? How does the guide's matched control address this concern, and what does it still leave to measure?

### S08-Q03 — Fair tuning and its limits

Method X wins at one learning rate shared by every method. Does that establish that X is the better method after tuning? Explain the guide's separate-tuning policy and one limit of that policy.

### S08-Q04 — Transferring evidence from RLVR

The guide describes RLPO/RLMO as using orthonormal rows in \(A_0\), zero \(B_0\), and RLVR. How does this differ from our matched spectral/random SFT experiment, and why can't its results directly settle our question?

## Section 9 — Testable hypotheses

### S09-Q01 — Calculate the interaction

Suppose final accuracies are E-S = 64%, E-R = 60%, D-S = 61%, and D-R = 60%. Calculate the spectral advantage under each gradient rule and the interaction \(I\). What does the sign of \(I\) mean in this example?

### S09-Q02 — Spectral gain without interaction

If spectral initialization improves accuracy by 3 percentage points under both exact and detached DoRA, what is \(I\)? Which hypothesis could still be supported even though the interaction is zero?

### S09-Q03 — Why include the additive pair?

What can comparing A-S with A-R tell us alongside the four DoRA conditions? Why should standard LoRA L0 be treated as a practical reference rather than a control with an identical set of possible net adjustments?

### S09-Q04 — Measuring model movement

At initialization, \(B_0A_0\) has a large Frobenius norm, but the reconstructed effective model is exactly pretrained. Which measurement correctly reports zero net model movement? Distinguish it from adapter magnitude and one-step movement.

### S09-Q05 — Interpreting tuned and uncertain results

Separate tuning chooses different \(\gamma\) and \(\eta\) values for different conditions, and an estimated interaction has a wide uncertainty interval containing zero. What conclusion is justified? Why also study shared hyperparameters and paired seed blocks?

## Section 10 — Practical training background

### S10-Q01 — Loss versus answer accuracy

A fine-tuning run achieves lower next-token loss on worked solutions, but generated final-answer accuracy does not improve. Is that contradictory? Explain how the training and evaluation procedures differ.

### S10-Q02 — Protecting evaluation

The plan starts with 7,473 training examples and reserves two disjoint sets of 500. How many examples remain for training? Why should the test set not decide the learning rate, initialization scale, or pilot recipe?

### S10-Q03 — Pilot and scope

Why does the guide favor a small model and repeated conditions within its GPU budget? Name two things a pilot must establish and explain why a result on this setup would not automatically establish a general result for larger models or RLVR.

### S10-Q04 — Warmup as a provisional choice

For 609 optimizer steps, how many warmup steps result from rounding up 3% of the total? Explain what warmup does and why a precedent for 3% does not prove it is optimal for our experiment.

## Section 11 — References and verification

### S11-Q01 — Choose the evidence that owns the claim

Where would you look to check each of these: the derivative of normalized DoRA; whether a particular library implementation detaches the denominator; and whether our own experiment achieved a spectral accuracy gain? Explain why one source cannot establish all three.

### S11-Q02 — Evidence status and reproducibility

What is the difference between a paper's reported result, a numerical illustration in `background.md`, and a proposed experiment outcome? Why record a paper's exact version and inspect the actual code version when reproducing a claim?
