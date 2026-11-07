# minRNN Foundations

A notebook implementing minGRU and minLSTM from [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201). It compares sequential, parallel log-space, and vanilla scans, with correctness tests and a numerical-stability comparison.

Open [`minrnn_foundations_colab.ipynb`](minrnn_foundations_colab.ipynb) in Google Colab and run the cells. 

## Result 1: Efficiency benchmark

Run the environment, scan-primitives, minGRU, and minLSTM definition cells (Sections 1–4), then the **Result 1: Efficiency benchmark** section at the end of [`minrnn_foundations_colab.ipynb`](minrnn_foundations_colab.ipynb) in Google Colab with a GPU runtime selected. The section contains the experiment code, configuration, correctness preflight, measurements, CSV export, and plots; no helper script is required. The intervening foundations demos and correctness/stability tests are optional for Result 1.

The benchmark covers minGRU, minLSTM, plain-PyTorch GRUCell/LSTMCell baselines, and Mamba when its official package is available. It measures full forward/loss/backward steps while varying sequence length. It reports CUDA total and incremental peak allocated memory; CPU memory is explicitly unavailable. Change the configuration cell for a short smoke run or the full sweep.

The paper specifies batch size 64, the same layer count across models, forward/loss/backward measurement, and 100 timing runs; verify hidden dimension and Mamba settings against its appendix before describing the run as an exact reproduction. The standard PyTorch GRUCell/LSTMCell implementations here are explicitly unrolled over time rather than using fused sequence kernels. Record the selected GPU and software versions, and export the CSV and plots with the notebook.
