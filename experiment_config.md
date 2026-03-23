
\subsection{Experimental setup and comparison methods}
\vspace{0.5em}
\noindent\textbf{Training Settings.} We train AgriDINO in two stages. In the first stage, we adapt the DINOv3 vision backbone to the agricultural domain using a standard DINO-style self-supervised learning pipeline on 240k agricultural pest and disease images. global and local crop scales are set to $[0.4, 1.0]$ and $[0.05, 0.4]$, respectively. Training is conducted for 30 epochs using the AdamW optimizer with a base learning rate of $0.0005$ and weight decay of $0.04$.
In the second stage, after self-supervised adaptation, we freeze the DINOv3 visual backbone and train the text encoder, linear projection heads, and the MAP module on the Agricap dataset. The batch size is set to $128$, and training runs for 20 epochs using AdamW (learning rate $0.0006$, weight decay $0.001$, and a linear warmup of $10\%$). All input images are resized to $224 \times 224$, and experiments are conducted on two NVIDIA A800 GPUs.

\noindent\textbf{Evaluation Protocol.} We evaluate our model on eight target benchmarks that are out-of-distribution (OOD) with respect to the Agricap pre-training distribution: Onion \citep{chilli_onion}, Durian \citep{durian}, SoursopDB \citep{SoursopBD2024}, Radish \citep{Radish}, Black Gram \citep{black_gram}, Spinach \citep{spinach}, Insect Diff \citep{InsectDiff}, and Li Pest \citep{Lipest}, see Table~\ref{tab:ood}, as well as on the Agricap test split. On OOD datasets, we conduct zero-shot and few-shot classification experiments. Zero-shot classification employs a unified prompt template. Few-shot classification includes both linear probing and fine-tuning, with Top-1 accuracy reported as the metric. On Agricap, we perform image-text retrieval, reporting Recall@\{1, 5, 10\} for both image-to-text(I$\rightarrow$T) and text-to-image(T$\rightarrow$I) retrieval.
\begin{table}[width=\linewidth,pos=htbp]
    \caption{Statistics of out-of-distribution agricultural pest and disease datasets.}
    \label{tab:ood}
    % 使用 tabular* 并设置宽度为 \linewidth
    % @{\extracolsep{\fill}} 会自动把多余的空白均匀分配到列之间
    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} lrr @{}} 
        \toprule
        Dataset & Classes & Images Num \\ 
        \midrule
        Onion & 4 & 816 \\
        Durian & 4 & 413 \\
        Soursop & 6 & 3838 \\
        Radish & 5 & 2081 \\
        Black Gram & 5 & 4038 \\
        Spinach & 5 & 3006 \\
        Insect Diff & 9 & 4482 \\
        Li Pest & 10 & 5869 \\
        \bottomrule
    \end{tabular*}
\end{table}
For few-shot classification, we attach a linear classifier head. To ensure fair comparison and optimal downstream performance, we adopt an end-to-end adaptation strategy during few-shot fine-tuning: the entire image encoder is unfrozen and jointly trained with the linear classification head under identical optimization epochs and random sampling strategies, with $K$-shot settings $K \in \{1, 4, 16\}$. 

For linear probing, only the linear classification head is trained while the image encoder remains frozen, with $K \in \{1, 4, 16, 32, 64\}$. We report the macro-average accuracy across $D$ datasets by aggregating the per-dataset mean $\mu_d$ and standard deviation $\sigma_d$ as
\begin{equation}
\mu_{\mathrm{macro}} \pm \sigma_{\mathrm{macro}}
= \frac{1}{D}\sum_{d=1}^{D}\mu_d \ \pm\  \frac{1}{D}\sqrt{\sum_{d=1}^{D}\sigma_d^{2}},\quad D=8.
\end{equation}

All experiments are trained using cross-entropy loss for 10 epochs, and each setting is repeated 10 times with different random seeds. 

\noindent\textbf{Baselines.} We compare AgriDINO with the following models: (i) General-purpose VLMs: OpenCLIP \citep{10-CLIP}, SigLIP2 \citep{siglip2}, and FG-CLIP2 \citep{fgclip2}; (ii) Domain-specific VLMs: AgriCLIP \citep{agriclip}, SCOLD \citep{scold}, and BioCLIP2 \citep{gu2025bioclip2}. All baseline models were fine-tuned on AgriCap using identical settings and evaluated using their official inference implementations. The architectures, pretraining data, and parameter sizes of these models are summarized in Table~\ref{tab:models}.
\begin{table*}[width=\linewidth,cols=5,pos=htbp]
\caption{Architectures and pretraining details of AgriDINO and baseline models.}
\label{tab:models}
\centering
\footnotesize 
\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} llllr @{}}
\toprule
Model & Vision Backbone & Text Backbone & Pretraining Data & Params (M) \\
\midrule
OpenCLIP & ViT-L/14 & Masked Self-attention Transformer & LAION-2B & 428 \\
FG-CLIP2 & ViT-L/16 & Causal Transformer  & LAION-2B + FineHARD & 1200 \\
SigLIP2  & ViT-L/16 & Causal Transformer  & WebLI-10B  & 900\\
BioCLIP2 & ViT-L/14 &  Masked Self-attention Transformer & TreeOfLife-10M & 428 \\
AgriCLIP & DINO-ResNet-50 & Self-attention Transformer & Alive & 173 \\
SCOLD & Swin-T & RoBERTa-base & LeafNet & 171 \\
\textbf{AgriDINO} & DINOv3 ViT-L/16 & Causal Transformer & Agricap & 866 \\
\bottomrule
\end{tabular*}
\end{table*}
