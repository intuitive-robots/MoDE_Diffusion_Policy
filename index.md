---
layout: project_page
permalink: /

title: "Variational Distillation of Diffusion Policies into Mixture of Experts"
authors: Hongyi Zhou, Denis Blessing, Ge Li, Onur Celik, Xiaogang Jia, <a href="https://alr.iar.kit.edu/21_65.php">Gerhard Neumann</a>, <a href="https://rudolf.intuitive-robots.net/">Rudolf Lioutikov</a>
affiliations: <a href="https://www.irl.iar.kit.edu/">KIT Intuitive Robots Lab</a>
venue: "NeurIPS 2024"
paper: https://arxiv.org/abs/2406.12538
# video:
code: https://github.com/VDD-Anonymous/Variational-Diffusion-Distillation
# data:
# title-bg-landing-include: fpl-video.html
---


<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
This work introduces Variational Diffusion Distillation (VDD), a novel method
that distills denoising diffusion policies into Mixtures of Experts (MoE) through
variational inference. Diffusion Models are the current state-of-the-art in generative
modeling due to their exceptional ability to accurately learn and represent complex,
multi-modal distributions. This ability allows Diffusion Models to replicate the
inherent diversity in human behavior, making them the preferred models in behavior
learning such as Learning from Human Demonstrations (LfD). However, diffusion
models come with some drawbacks, including the intractability of likelihoods and
long inference times due to their iterative sampling process. The inference times,
in particular, pose a significant challenge to real-time applications such as robot
control. In contrast, MoEs effectively address the aforementioned issues while
retaining the ability to represent complex distributions but are notoriously difficult
to train. VDD is the first method that distills pre-trained diffusion models into MoE
models, and hence, combines the expressiveness of Diffusion Models with the
benefits of Mixture Models. Specifically, VDD leverages a decompositional upper
bound of the variational objective that allows the training of each expert separately,
resulting in a robust optimization scheme for MoEs. VDD demonstrates across
nine complex behavior learning tasks, that it is able to: i) accurately distill complex
distributions learned by the diffusion model, ii) outperform existing state-of-the-art
distillation methods, and iii) surpass conventional methods for training MoE.
        </div>
    </div>
</div>

## Method Overview

<div class="columns is-centered">
    <img src="./static/image/overview.png" alt="VDD" class="column is-four-fifths">
</div>

VDD distills a diffusion policy into an MoE. Learning from Demonstrations (LfD) is challenging due to the multimodality
of human behaviour. For example, tele-operated demonstrations of an avoiding task often contain
multiple solutions. A diffusion policy can predict high quality actions but relies on an
iterative sampling process from noise to data, shown as the red arrows in the figure above. VDD uses the score
function to distill a diffusion policy into an MoE, unifying the advantages of both approaches.

---

## Training of VDD in a 2D Toy Task

<div class="column is-one-third is-pulled-right p-0">
    <video width="100%" autoplay muted loop playsinline>
        <source src="./static/video/toy_task_animation.mp4" type="video/mp4">
    </video>
</div>

Illustration of training VDD using the score function for a fixed state in a 2D toy task.
The probability density of the distribution is depicted by the color map. The score function is shown
by the gradient field, visualized as white arrows. From (b) to (f), we initialize and train VDD until
convergence. We initialize 8 components, each represented by an orange circle. These components
are driven by the score function to match the data distribution and avoid overlapping modes by
utilizing the learning objective in Eq. 11 in the paper. Eventually, they align with all data modes.

<div class="columns is-centered">
    <img src="./static/image/toy_task_ov.png" alt="VDD training in a 2D Toy Task" class="column is-full">
</div>

## Competitive Distillation Performance in Imitation Learning Datasets

The results on the widely recognized imitation learning datasets Relay Kitchen and XArm Block Push indicate that VDD achieves a performance comparable to CD in both tasks, with
slightly better outcomes in the block push dataset. An additional interesting finding is that BESO,
with only one denoising step (BESO-1), already proves to be a strong baseline in these tasks, as
the original models outperformed the distillation results in both cases. We attribute this interesting
observation to the possibility that the Relay Kitchen and the XArm Block Push tasks are comparably
easy to solve and do not provide diverse, multi-modal data distributions. We therefore additionally
evaluate the methods on a more recently published dataset (D3IL) which is explicitly generated
for complex robot imitation learning tasks.

|                | **DDPM** | **BESO** | **DDPM-1** | **BESO-1** | **CD**             | **CTM**            | **VDD-DDPM (ours)**  | **VDD-BESO (ours)** |
|----------------|----------|----------|------------|------------|--------------------|--------------------|----------------------|----------------------|
| **Kitchen**     | 3.35     | 4.06     | 0.22       | 4.02       | 3.87 ± 0.05        | **3.89 ± 0.11**     | 3.24 ± 0.12          | 3.85 ± 0.10          |
| **Block Push**  | 0.96     | 0.96     | 0.09       | 0.94       | 0.89 ± 0.05        | 0.89 ± 0.04        | 0.93 ± 0.03          | **0.91 ± 0.03**    |
| **Avoiding**    | 0.94     | 0.96     | 0.09       | 0.84       | 0.82 ± 0.05        | 0.93 ± 0.02        | 0.92 ± 0.02          | **0.95 ± 0.01**    |
| **Aligning**    | 0.85     | 0.85     | 0.00       | 0.93       | **0.94 ± 0.08**  | 0.81 ± 0.11        | 0.70 ± 0.07          | 0.86 ± 0.04          |
| **Pushing**     | 0.74     | 0.78     | 0.00       | 0.70       | 0.66 ± 0.05        | 0.80 ± 0.07        | 0.61 ± 0.04          | **0.85 ± 0.02**    |
| **Stacking-1**  | 0.89     | 0.91     | 0.00       | 0.75       | 0.69 ± 0.06        | 0.54 ± 0.17        | 0.81 ± 0.08          | **0.85 ± 0.02**    |
| **Stacking-2**  | 0.68     | 0.70     | 0.00       | 0.53       | 0.46 ± 0.11        | 0.30 ± 0.09        | **0.60 ± 0.07**    | 0.57 ± 0.06          |
| **Sorting (Image)**  | 0.69 | 0.70 | 0.20       | 0.68       | 0.71 ± 0.07        | 0.70 ± 0.07        | **0.80 ± 0.04**    | 0.76 ± 0.04          |
| **Stacking (Image)** | 0.58 | 0.66 | 0.00       | 0.58       | 0.63 ± 0.01        | 0.59 ± 0.10        | **0.78 ± 0.02**    | 0.60 ± 0.04          |


## Comparison with MoE learning from scratch

Using VDD consistently outperforms direct MoE learning approachesm, using EM-GPT and IMC-GPT as baselines.

| **Environments**    | **EM-GPT**      | **IMC-GPT**      | **VDD-DDPM**  | **VDD-BESO**      |
|---------------------|------------------|------------------|------------------|-------------------|
|                     |                  |                  | (Ours)           | (Ours)            |
| **Avoiding**        | 0.65 ± 0.18      | 0.75 ± 0.08      | 0.92 ± 0.02      | **0.95 ± 0.01**   |
| **Aligning**        | 0.78 ± 0.04      | 0.83 ± 0.02      | 0.70 ± 0.07      | **0.86 ± 0.04**   |
| **Pushing**         | 0.16 ± 0.07      | 0.76 ± 0.04      | 0.61 ± 0.04      | **0.85 ± 0.02**   |
| **Stacking-1**      | 0.58 ± 0.06      | 0.54 ± 0.05      | 0.81 ± 0.08      | **0.83 ± 0.09**   |
| **Stacking-2**      | 0.34 ± 0.07      | 0.29 ± 0.07      | **0.60 ± 0.07**  | 0.57 ± 0.06       |
| **Sorting (image)** | 0.69 ± 0.02      | 0.74 ± 0.04      | **0.80 ± 0.04**  | 0.76 ± 0.03       |
| **Stacking (image)**| 0.04 ± 0.03      | 0.39 ± 0.10      | **0.78 ± 0.02**  | 0.60 ± 0.04       |
| **Relay Kitchen**   | 3.62 ± 0.10      | 3.67 ± 0.05      | 3.24 ± 0.12      | **3.85 ± 0.10**   |
| **Block Push**      | 0.88 ± 0.04      | 0.89 ± 0.04      | **0.93 ± 0.03**  | 0.91 ± 0.03       |



<!-- ## BibTeX

```bibtex
@inproceedings{
    reuss2023goal,
    title={Goal Conditioned Imitation Learning using Score-based Diffusion Policies},
    author={Reuss, Moritz and Li, Maximilian and Jia, Xiaogang and Lioutikov, Rudolf},
    booktitle={Robotics: Science and Systems},
    year={2023}
}
``` -->

<!-- ## Acknowledgements

The work presented here was funded by the German Research Foundation (DFG) – 448648559. -->

<!-- ## Related Projects
<h3><a href="https://intuitive-robots.github.io/mdt_policy/">Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals</a></h3>
<div class="column is-full columns">
    <div class="column is-half">
        <img src="./static/image/mdt-v-figure.png" alt="MDT-V Overview">
    </div>
    <div class="column is-half">
        <p>
        The Multimodal Diffusion Transformer (MDT) is a novel framework that learns versatile behaviors from multimodal goals with minimal language annotations. Leveraging a transformer backbone, MDT aligns image and language-based goal embeddings through two self-supervised objectives, enabling it to tackle long-horizon manipulation tasks. In benchmark tests like CALVIN and LIBERO, MDT outperforms prior methods by 15% while using fewer parameters. Its effectiveness is demonstrated in both simulated and real-world environments, highlighting its potential in settings with sparse language data.
        </p>
    </div>
</div>

<h3><a href="https://robottasklabeling.github.io/">Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models</a></h3>
<div class="column is-full columns">
    <div class="column is-half">
        <img src="./static/image/nils-ow.png" alt="NILS Overview">
    </div>
    <div class="column is-half">
        <p>
Using pre-trained vision-language models, NILS detects objects, identifies changes, segments tasks, and annotates behavior datasets. Evaluations on the BridgeV2 and kitchen play datasets demonstrate its effectiveness in annotating diverse, unstructured robot demonstrations while addressing the limitations of traditional human labeling methods.
        </p>
    </div>
</div> -->
