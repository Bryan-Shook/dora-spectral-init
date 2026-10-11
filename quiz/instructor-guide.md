# Background quiz: answer and teaching guide

Instructor notes for `question-bank.md`, based on the local `BACKGROUND.md` read on 2026-10-11. These notes explain the guide's mathematics and proposed protocol; they are not a fresh audit of the linked papers or current library behavior. Numerical examples are constructed, not experimental results.

The learner requested a slower explanation before starting. See `../background-beginner.md` for the expanded foundations, worked examples, and figures. The quiz is on hold as recorded in `progress.md`; wait for readiness or an answer to the pending question.

## How to conduct the quiz

1. Read `progress.md` and present only the pending question.
2. Assess both the answer and the explanation. Use **understood**, **partly understood**, or **needs explanation**. A correct number with a wrong explanation is not full understanding.
3. Start feedback with what the learner got right, if anything, then identify the precise gap. If they say “I don't know,” teach without making them guess repeatedly.
4. Explain the gap using plain language and a small example. Use display equations when they help. Do not reveal later questions' answers unnecessarily.
5. For an essential gap, ask one targeted follow-up before advancing. A follow-up checks learning; preserve the initial assessment and record the follow-up outcome separately.
6. Update `progress.md` after each response and ask only one next question. At a section boundary, briefly describe any remaining gap and continue unless the learner requests a pause.
7. Accept equivalent explanations, valid sign choices for singular vectors, and alternative correct derivations. Distinguish loss gradients from negative-gradient updates.

Use these hints only when asked or when the learner is stuck. Treat follow-ups as optional diagnostic tools, not a second fixed exam. The goal is to locate and repair knowledge gaps.

## Important teaching boundaries

- A scalar gamma scales matrix factors; “orthonormal” here describes their columns or rows.
- Nonzero diagonal scaling of orthonormal vectors preserves perpendicularity, but generally changes their unit lengths.
- A complete orthonormal basis reconstructs every input. A truncated basis reconstructs only the projection onto its span.
- Top singular directions optimize reconstruction of the pretrained matrix, not automatically downstream task loss.
- Equal products or effective weights do not force equal gradients with respect to differently parameterized factors.
- The effective-step formula in Section 5 assumes simultaneous plain gradient descent. The guide's AdamW experiment needs actual optimizer-step diagnostics.
- A tangent gradient with respect to a directly trainable row does not guarantee a tangent factor-induced update or finite optimizer step.
- Normalization formulas assume nonzero rows. Do not introduce behavior at a zero norm without specifying an implementation safeguard.
- The rank-up-to-2r statement concerns the additive net change. Do not transfer it without qualification to DoRA's row-scaled effective weights.
- Hold batch and forward stochastic state fixed when comparing starting gradients across conditions. Matching weights alone is not sufficient if dropout or other randomness differs.
- Frozen means not optimized. It does not mean omitted from forward computation, and frozen layers can still be needed for backpropagation to trainable components elsewhere.
- An uncertainty interval containing zero is not proof of no effect. Check precision and the proposed practical margin.
- The guide describes proposed work. Do not present pilot feasibility, accuracy gains, novelty, or a recipe's optimality as established results.

## Section 1

### S01-Q01

**Expected:** W0 is 3 by 4: three output rows, four input columns. A 3-by-4 matrix times a 4-by-1 vector gives a 3-by-1 output. Each row computes a weighted sum of the four inputs; the bias has three entries.

**Check:** Understands input dimension matches columns and output dimension matches rows, rather than guessing the order.

**Hint:** How many input weights does one output need?

**Follow-up if transposed:** Write the weighted sum for just the first output. How many coefficients does that row contain?

### S01-Q02

**Expected:** The effective layer still needs the pretrained transformation W0x. Most backbone weights remain frozen but still compute representations. Training may need derivatives through frozen operations to update adapters elsewhere. PEFT reduces trainable parameters and associated storage/optimizer work; it does not remove backbone execution.

**Misconception:** “No gradient stored for a weight” means “no computation involving the weight.”

**Follow-up:** In y = W0x + BAx, what would omitting W0x do to the initial pretrained prediction?

### S01-Q03

**Expected:** Both norms are 5 because each squares and sums the relevant entries, then takes a square root. z is a 2-coordinate vector; M is a 2-by-2 matrix with four entries. Equal norm values do not make the objects or their operations identical.

**Check:** Distinguishes Frobenius norm from the largest singular value or matrix spectral norm. The matrix's spectral norm is 4 here.

**Follow-up:** If M's two diagonal entries were both 4, what would its Frobenius norm be? Answer: 4 sqrt(2), not 4.

## Section 2

### S02-Q01

**Expected:** A is 2 by 4 and maps the input into two coordinates. B is 6 by 2 and maps them into six outputs. The factors have 8 + 12 = 20 entries, compared with 6 times 4 = 24 in the full weight. This small example gives modest savings; the benefit grows when r is small relative to both dimensions.

**Check:** Matrix composition BA, not AB; no addition of the full frozen weight to the trainable count.

**Follow-up:** What is the shape of BA? Answer: 6 by 4.

### S02-Q02

**Expected:** At each instant, the adjustment's column space is contained in the current columns of B and its rank is at most 2. B is trainable, so its output directions can move. Its rank constraint does not fix the initial subspace forever, nor does it mean exactly rank 2 at every step.

**Follow-up:** If B starts at zero, does it have a fixed two-dimensional output subspace already? Answer: no; its initial column space is just zero and subsequent columns can develop.

### S02-Q03

**Expected:** Random A and zero B give zero initial product. The gradient of A is initially zero; B's gradient GA^T can be nonzero, although that is not guaranteed for every G or batch. Once B changes, A can receive a loss gradient. If both are zero, both displayed loss gradients are zero, so these loss-driven updates cannot start adaptation.

**Check:** Does not equate a zero product with both gradients being zero. This question concerns the displayed task-loss gradients, not an unspecified regularizer.

**Follow-up:** In the scalar product BA, which derivative is A and which is B?

### S02-Q04

**Expected:** No. The count measures trainable parameters for one matrix. Forward computation, backpropagation through the model, adapter computation, optimizer overhead, activations, memory bandwidth, and the rest of the model affect training time. A speed ratio requires measurement.

**Follow-up:** Does the original 1024-by-1024 multiplication still happen? Answer: yes, unless a separately specified execution strategy changes it; freezing itself does not remove it.

## Section 3

### S03-Q01

**Expected:** W0(2vi) = 10ui by linearity. vi is a unit input direction, ui is the associated unit output direction, and 5 is the stretch factor. The input length is 2 and output length is 10.

**Follow-up:** What if the singular value were zero? Answer: the input direction would be mapped to zero.

### S03-Q02

**Expected:** Both coefficients are sqrt(2). Each vector has unit length, and their dot product is zero. Dotting x with vj leaves its own coefficient because vj^Tvi is 0 for other directions and 1 for itself. The two vectors span the plane, so this reconstruction covers every 2D x.

**Misconception:** Every pair of perpendicular vectors gives coefficients directly by dot product regardless of length. For nonunit orthogonal vectors, divide by the squared norm.

**Follow-up:** With only v1 retained, does (v1^T x)v1 reconstruct this x? Answer: no; it gives (1,1), its projection.

### S03-Q03

**Expected:** Substitute W0^TW0 vj = lambda_j vj. The output dot product is lambda_j vi^T vj = 0 for different perpendicular input eigenvectors. Squared output length is lambda_i, so divide each nonzero output by sqrt(lambda_i) to obtain orthonormal output directions.

**Check:** Does not claim arbitrary perpendicular inputs remain perpendicular under every matrix. Symmetry of W0^TW0 allows an orthonormal eigenbasis, including a choice within repeated-eigenvalue subspaces.

**Follow-up:** For a generic matrix, does the zero input dot product alone make vi^T W0^TW0 vj zero? Answer: no.

### S03-Q04

**Expected:** B0 = (3,0)^T, A0 = (3,0), product diag(9,0), residual diag(0,4). The sum is diag(9,4). sqrt(9) = 3 distributes the product strength evenly: 3 times 3 = 9. Joint sign flips of the corresponding factor vectors are valid.

**Check:** Using singular value 9 in both factors would incorrectly produce 81.

**Follow-up:** Which components are trained and which part stays fixed? Answer: A,B are trained, residual is frozen.

### S03-Q05

**Expected:** No. Best rank-r approximation minimizes reconstruction error of W0 in the stated norm. The downstream objective depends on data and loss, not solely the weight's singular values. A task could rely on a low-strength direction. Empirical controls and tuning are needed to assess task value.

**Follow-up:** Could a weak pretrained input direction be important if the new task's useful signal lies there? Answer: yes.

## Section 4

### S04-Q01

**Expected:** gamma = 9 is the scalar, sqrt(gamma) = 3, and Sigma_r^(1/2) = diag(2,1). Column lengths are 6 and 3. They remain perpendicular but are not unit vectors, hence not orthonormal. B0 is a matrix, not a scalar. Its Gram matrix B0^T B0 = diag(36,9).

**Follow-up:** Can one shared gamma make both lengths 1 if their singular values differ? Answer: no; it would require gamma sigma_i = 1 for each i.

### S04-Q02

**Expected:** Factor singular values sqrt(gamma sigma_i), product singular values gamma sigma_i, and product Frobenius norm gamma sqrt(sum_i sigma_i^2) match. Product entries and factor gradients need not match. Orthonormal bases preserve the prescribed singular values while changing orientation. The guarantee is at shared gamma and initialization; it is not a guarantee after updates.

**Follow-up:** Do two matrices with equal Frobenius norms necessarily act identically on x? Answer: no.

### S04-Q03

**Expected:** Random product is [[2,2],[2,2]]. Its own residual is [[2,-2],[-2,-1]], making the sum W0. Reusing the spectral residual produces [[2,2],[2,3]], which differs from W0 and changes the starting model.

**Check:** The correction is initialization-specific and stays frozen; do not recompute it as W0 - B_t A_t after every step, which would cancel learning.

**Follow-up:** What happens if the residual is recomputed to cancel the current product on every step? Answer: the additive effective weight remains W0.

### S04-Q04

**Expected:** Net change is B_t A_t - B0 A0. Rank(X-Y) <= rank(X) + rank(Y) <= 2r, also bounded by matrix dimensions. Changing subspaces lets the difference contain directions from both products. For r=1 in 2D, subtracting diag(0,1) from diag(1,0) gives diag(1,-1), rank 2. This is a maximum bound, not a claim that every change reaches it.

**Follow-up:** What makes standard zero-product LoRA's net adjustment different? Answer: there is no nonzero initial product to subtract, so its additive change has rank at most r.

### S04-Q05

**Expected:** Neither conclusion follows. Different A,B transform the same upstream gradient differently, despite matched initial effective weights and spectra. The random control still uses singular values from W0, requiring the planned SVD computation. It is an orientation control, not a demonstrated way to eliminate initialization cost.

**Follow-up:** Which part of an SVD does the random control discard and which does it retain? Answer: replaces singular-vector directions while retaining selected singular values.

## Section 5

### S05-Q01

**Expected:** Perturbing B gives dZ = dB A; perturbing A gives dZ = B dA. Rewriting dL = trace(G^T dZ) gives the displayed gradients. GA^T is d_out by r; B^TG is r by d_in. Different factor values mean different maps from the upstream G to trainable coordinates.

**Teaching option:** If trace notation is unfamiliar, use Z_ij = sum_k B_ik A_kj, then differentiate with respect to one factor entry.

**Follow-up:** If B is zero, which displayed gradient is necessarily zero? Answer: the gradient of A.

### S05-Q02

**Expected:** For (2,2), first-order change is -0.01(4+4) = -0.08. For (4,1), it is -0.01(16+1) = -0.17. The same product and same upstream gradient need not give the same effective step. Exact scalar changes for the given simultaneous step are -0.0796 and -0.1696, with a +0.0004 second-order term in each.

**Check:** Does not interpret the larger step as automatically better optimization or accuracy.

**Follow-up:** Why doesn't preserving BA under B -> cB, A -> A/c preserve B^2+A^2?

### S05-Q03

**Expected:** The exact difference for the displayed updates is

\[
-\eta(GA^TA+BB^TG)+\eta^2GA^TB^TG.
\]

The last term multiplies both factor updates and is second order in eta. The first-order formula drops it. AdamW uses moment estimates, coordinate-wise adaptive scaling, and weight decay, so its steps are not generally -eta times these raw gradients. The expansion also assumes both gradients were evaluated at the old factors.

**Follow-up:** Could you use the same expression unchanged if A's gradient were evaluated after updating B? Answer: no; that would be a different update rule.

### S05-Q04

**Expected:** Optimizer size -> eta; forward adapter multiplier -> s; compensated initialization size -> gamma; effective row magnitude -> m. They act at different points in the computation and chain rule. In this plan s=1. Gamma changes factor sizes and the initial residual while preserving Z0; eta changes optimizer steps; m is a learned row length.

**Follow-up:** If gamma changes and the correction is recomputed properly, must starting predictions change? Answer: no.

### S05-Q05

**Expected:** BB^T weights/mixes output directions by left multiplication; A^TA weights/mixes input directions by right multiplication. For balanced spectral factors these are gamma U_r Sigma_r U_r^T and gamma V_r Sigma_r V_r^T. Random factors replace U_r,V_r with Q_L,Q_R. Their eigenvalues match, but the matrices need not match because their directions differ.

**Check:** They are weighted subspace operators, not generally orthogonal projectors. Eigenvalues are gamma sigma_i rather than necessarily 1.

**Follow-up:** Does matching eigenvalues make a rotated ellipse point in the same direction? Answer: no; a useful geometric analogy for matched strengths and different orientation.

## Section 6

### S06-Q01

**Expected:** Norm(z)=5, u=(0.6,0.8)^T, w=(6,8)^T, norm(w)=10. Changing positive m changes length without changing direction. In general the formula gives norm(w)=|m|; the example uses positive m.

**Follow-up:** If z becomes (6,8) while m stays 10, what is w? Answer: still (6,8).

### S06-Q02

**Expected:** Initialize m_i to the pretrained row's Euclidean norm. Because z_i equals that pretrained row at initialization, m_i z_i / norm(z_i) recovers it. Normalize separately over the d_in entries of each output row; in the stated PyTorch weight orientation, dim=1. Assumes nonzero rows.

**Follow-up:** Would one norm for the entire matrix generally recover every row using its individual m_i? Answer: no.

### S06-Q03

**Expected:** 2z=(2,2), same direction as (1,1), so normalized w is unchanged. Scaling only the adapter gives (1,2), a different direction: (1,2)/sqrt(5) versus (1,1)/sqrt(2). Scaling one summand is not scaling the whole sum. This example deliberately holds the residual fixed, unlike initialization-time compensation.

**Follow-up:** What additional change would make the operation a uniform doubling of the entire row? Answer: double the residual too.

### S06-Q04

**Expected:** No. Forward invariance does not establish invariance of gradients or optimizer updates in factor coordinates. Gamma changes A,B and their gradient maps, with a compensating residual. Eta changes their steps. Even directly parameterized normalized rows have a derivative factor m/norm(z), so equal forward directions can have different derivatives with respect to their underlying coordinates.

**Follow-up:** If z is doubled with m fixed, does m/norm(z) stay unchanged? Answer: it halves.

## Section 7

### S07-Q01

**Expected:** Detached normalization treats the current norm as a constant for differentiation in that backward pass. It recomputes the norm on the next forward pass. It does not freeze z or m. The two forward values agree at the same parameters, though later training trajectories can diverge.

**Follow-up:** If z changes length from 5 to 7, what denominator does the next detached forward pass use? Answer: 7.

### S07-Q02

**Expected:** u=(0.6,0.8), u^Tg=0.6, parallel component u(u^Tg)=(0.36,0.48). Since m/norm(z)=1, g_E=(0.64,-0.48), g_D=(1,0). Dotting g_E with z gives 1.92-1.92=0. Both give magnitude gradient 0.6 if asked, but that is not needed to answer this question.

**Check:** These are gradients; a gradient-descent update uses their negative.

**Follow-up:** What happens to the norm's contribution if g is already perpendicular to z? Answer: no component is removed and the two z-gradients agree.

### S07-Q03

**Expected:** A small change along z changes its length, not its normalized direction. The exact derivative projects g onto the perpendicular space: P=I-uu^T. If g is parallel, exact z-gradient is zero; detached is (m/norm(z))g, possibly nonzero. Magnitude gradient is u^Tg and can still be nonzero, so a radial effective-weight adjustment can occur through m.

**Misconception:** Exact DoRA cannot learn row length. It learns that through m, despite removing the radial component from the direction-coordinate gradient.

**Follow-up:** Which trainable parameter should carry a request to change only effective row length? Answer: m.

### S07-Q04

**Expected:** No to both. The tangent statement is about the derivative with respect to z. Factor gradient maps and an adaptive optimizer can alter the direction of the induced Z step, and finite steps require re-normalization. Mathematical exactness of the derivative is not evidence of better task accuracy. Compare actual effective steps and empirical performance.

**Follow-up:** Would an unequal diagonal rescaling of a tangent vector necessarily keep it perpendicular to z? Answer: no; this illustrates why coordinate-wise scaling matters.

### S07-Q05

**Expected:** norm(z)=m=sqrt(2), so g_D=(1,1), g_E=0. Treat G_D as row (1,1). B's gradient is 1; A's gradient is (1,1). To first order, delta(BA) = delta B A + B delta A = (-2 eta,-eta), hence the same delta Z. This is not parallel to (1,1). Its tangent component is (-eta/2,+eta/2), so normalization retains a nonzero first-order direction change. Equivalently, the row acts through H=B^2 I+A^TA=diag(2,1): P g=0 does not imply P H g=0.

**Key boundary:** m is deliberately fixed. If it were trained, both rules would also receive the same nonzero magnitude gradient, 2/sqrt(2)=sqrt(2), at this state. Do not claim the entire exact DoRA update vanishes with trainable m.

**Hint:** Calculate the two factor gradients first, then add their first-order contributions to BA.

**Follow-up:** Why would updating z directly along the radial g be different? Answer: a radial direct step preserves its direction locally; the factor map here creates a tangent component.

### S07-Q06

**Expected:** Inspect the actual implementation used in the experiment, including any detach/stop-gradient on the norm or on an adapter contribution entering the norm. A configuration flag alone does not establish the derivative. At a nonzero test row, hold m fixed and choose a radial g: analytic exact z-gradient is zero while detached is generally nonzero. Also verify same forward value and recomputation of the denominator after changing z. For an exact implementation, compare autograd with the analytic formula and finite differences of the normalized forward function.

**Boundary:** Finite differences of the shared forward function validate its true exact derivative. The detached rule intentionally differs from that derivative; it must be checked against its stated stop-gradient rule, not expected to pass the same finite-difference test.

**Follow-up:** Can identical forward outputs detect detachment by themselves? Answer: no.

## Section 8

### S08-Q01

**Expected:** Do not claim the combination of spectral initialization and weight decomposition is new merely from building it. The proposed contribution is evidence about initialization-by-detachment interaction with matched controls, separate tuning, and diagnostics. This is a scoped claim; broader novelty requires literature checking.

**Follow-up:** Does proposing the experiment establish the interaction already? Answer: no; it remains empirical.

### S08-Q02

**Expected:** Direction and magnitude both changed, so gain cannot yet be attributed to direction alone. Matching factor and product spectra at shared gamma removes a specified initial magnitude mismatch while rotating directions. Later gradients, optimizer steps, movement, and task outcomes remain quantities to measure. Separate tuning further changes the comparison's interpretation.

**Check:** Does not claim the control matches every possible property or isolates a universal cause independent of optimizer and data.

**Follow-up:** Does a larger step norm necessarily mean a more useful task update? Answer: no.

### S08-Q03

**Expected:** A shared rate can suit one parameterization better. Give each condition an equal search budget and select using development data. This is a practical protocol, not proof of each method's global optimum; conclusions are conditional on ranges, budget, and selection rules.

**Follow-up:** Would test-set selection repair a small search budget? Answer: no; it contaminates evaluation.

### S08-Q04

**Expected:** The described RLPO/RLMO uses unscaled orthonormal A rows and zero B, selecting principal or minor right singular directions. Our main conditions have two nonzero balanced factors with prescribed singular-value scaling and residual correction. RLVR learns from checkable rewards for generated answers; SFT trains on supplied target solutions. Different initialization, objectives, data, and setting prevent direct transfer of an outcome.

**Follow-up:** Are our scaled A rows generally orthonormal? Answer: no, although their initial rows are perpendicular.

## Section 9

### S09-Q01

**Expected:** Delta_E=4 percentage points, Delta_D=1, I=3. The observed spectral advantage is 3 points larger under exact than detached. This does not establish statistical certainty or general superiority from these four numbers alone.

**Check:** Difference of differences, not just E-S minus D-S. Use percentage points, not relative percent gain.

**Follow-up:** Could exact have higher absolute accuracy in both orientations but I be zero? Answer: yes, if both spectral advantages are equal.

### S09-Q02

**Expected:** I=0, while within-rule spectral value H2 could be supported. H1 is about whether the spectral advantage changes across gradient rules, not whether spectral initialization helps at all. Uncertainty must still be considered in real results.

**Follow-up:** Is zero interaction equivalent to zero spectral advantage? Answer: no.

### S09-Q03

**Expected:** The additive pair measures spectral versus random behavior without DoRA normalization and helps interpret whether normalization/gradient rules alter that pattern. It does not by itself prove a mechanism. L0 is a familiar reference but uses zero-product initialization; its additive net change has rank at most r, whereas nonzero initial-product subtraction can allow up to 2r. Equal parameter counts do not give identical attainable net changes.

**Follow-up:** Is a DoRA-versus-additive accuracy difference alone enough to attribute the cause to denominator detachment? Answer: no.

### S09-Q04

**Expected:** norm(W_eff,t-W0)_F measures net movement and is zero initially. norm(B_t A_t)_F measures the already-nonzero product, not departure from pretrained weights. norm(W_eff,t+1-W_eff,t)_F measures one effective step. Angles can complement norm measurements for directional movement.

**Follow-up:** If the adapter product is large but exactly canceled by the frozen residual correction initially, has the model already moved? Answer: no.

### S09-Q05

**Expected:** The estimate concerns performance after the specified tuning protocol, not a pure fixed-hyperparameter derivative experiment. A wide interval containing zero is inconclusive: neither a clear interaction nor evidence of practical equivalence follows. Shared gamma,eta comparisons help explain behavior under matched settings. Fresh paired seed blocks support repeated comparisons while controlling shared sources of variation; pairing must be defined rather than assuming unrelated draws magically match. Report uncertainty and compare precision with the proposed practical margin.

**Follow-up:** Would a very narrow interval around zero answer the same question as a wide one? Answer: no; precision can support a limited practical-null conclusion relative to a specified margin.

## Section 10

### S10-Q01

**Expected:** No contradiction. Teacher-forced SFT scores next-token probabilities on supplied solution prefixes; evaluation generates solutions using the model's own previous tokens and scores final numerical answers. Small probability improvements need not change the selected greedy tokens or exact answer; generated errors can change later prefixes. Better token loss and answer accuracy are related but distinct.

**Follow-up:** Does a high probability for several correct tokens guarantee the final number is correct? Answer: no.

### S10-Q02

**Expected:** 7473 - 500 - 500 = 6473 training examples. Development/pilot data guide choices; the test set estimates outcomes after those choices are fixed. Repeatedly selecting on test performance makes the reported evaluation reflect selection on its answers. Maintain the specified disjoint splits.

**Boundary:** The background names two validation partitions but does not fully define their distinct roles. Consult TEST_PLAN.md for that detail rather than inventing it.

**Follow-up:** If a recipe is changed after seeing test accuracy, is the same test still an untouched evaluation? Answer: no.

### S10-Q03

**Expected:** A small model preserves budget for separate tuning, controls, and repetition. The pilot must show learning progress and measure the cost of a full run including relevant training/evaluation overhead; source details can determine additional checks. Feasibility is not proven merely by model size. Results remain conditional on the model, task, rank, adapter locations, data, and search budget; larger models and RLVR require further evidence.

**Follow-up:** Should unmeasured expected speed or accuracy be described as pilot results? Answer: no.

### S10-Q04

**Expected:** 0.03 times 609 = 18.27, rounded up to 19. Warmup gradually increases the learning rate before the peak, limiting early steps. A documented precedent supplies a provisional recipe, not evidence of optimality in this setup. Pilot changes should be documented and then fixed across main conditions.

**Follow-up:** Is 3% warmup itself an observed accuracy gain in this project? Answer: no.

## Section 11

### S11-Q01

**Expected:** The mathematical derivation in DoRA (R1, Sections 4.1–4.3) addresses the normalized derivative; the actual source/version/configuration and a gradient check address the library path; our completed, validated experiment artifacts would establish our accuracy results. The guide's saved source snapshot addresses only that saved implementation, not every current release. Citations to another paper do not establish our results.

**Follow-up:** Does a source-code comment promising “exact DoRA” outweigh a contradictory executed gradient check? Answer: no; inspect and resolve the discrepancy.

### S11-Q02

**Expected:** A reported result is evidence within the paper's evaluated setting; a numerical illustration demonstrates an idea with invented inputs; a proposal describes work and expected measurements before outcomes exist. Pinning paper and code versions makes definitions and behavior reproducible and exposes drift. Neither references nor plans establish that our protocol reproduces every paper or that our hypothesis is true.

**Follow-up:** Can the guide's illustrative 62% accuracy be cited as our achieved score? Answer: no.

## End-of-quiz report

After the final question, report the learner's demonstrated strengths, remaining gaps, and a short recommended review sequence. Separate initial understanding from understanding after hints or explanations. Link to the progress record. Use targeted examples for unresolved prerequisites rather than assigning every source indiscriminately. Do not claim mastery just from completing all questions.
