# LSTM Air Pollution Forecasting

Boyu Yue

ybyleo@126.com

5/3/25

## Introduction

In this assignment, I will use LSTM architecture to understand and apply multiple variables together to contribute more accuracy towards forecasting based on the dataset that reports on the weather and the level of pollution each hour for five years at the US embassy in Beijing, China.

## Methodology

**Long Short-Term Memory (LSTM)** networks are a specialized form of recurrent neural networks (RNNs) designed to effectively capture long-term dependencies in sequential data. Unlike traditional RNNs, which suffer from vanishing and exploding gradient problems during backpropagation through time, LSTMs address these issues through the introduction of a memory cell and gating mechanisms. Each LSTM unit comprises three main gates: the input gate, forget gate, and output gate. These gates regulate the flow of information into and out of the memory cell, allowing the network to retain relevant information over extended time intervals and discard irrelevant or outdated data. LSTM networks are particularly well-suited for tasks involving time-series prediction, natural language processing, and any domain where sequential context is critical.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1200px-LSTM_Cell.svg.png" alt="Long short-term memory - Wikipedia" style="zoom:35%;" />

At each time step $t$, given the input vector $\mathbf{x}_t$, previous hidden state $\mathbf{h}_{t-1}$, and previous cell state $\mathbf{c}_{t-1}$, the LSTM unit computes the following:

**Forget Gate:** Decides what information to discard from the cell state:
$$
\mathbf{f}_t = \sigma\left(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)
$$
**Input Gate:** Determines what new information to store in the cell state:
$$
\mathbf{i}_t = \sigma\left(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\right)
$$

$$
\tilde{\mathbf{c}}_t = \tanh\left(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\right)
$$

**Cell state update:** Combines the forget and input gate operations to update the cell state:
$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$
**Output Gate:** Determines the output and the next hidden state:
$$
\mathbf{o}_t = \sigma\left(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\right)
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

## Data Preprocessing and Hyperparameters

Explanation of data preprocessing and hyperparameters are listed in the code (Jupyter notebook).

## Experimental Studies

Using the LSTM model I built, with hyperparameters: $dropout=0.2$, $loss=huber$, $leaning\: rate=0.001$, $epochs=150$, $batch\:size=64$, $validation\:frequency=5$, the result is as below.

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1746247251476.png" alt="1746247251476" style="zoom:50%;" />

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1746247273231.png" alt="1746247273231" style="zoom:50%;" />

<img src="../../../HuaweiMoveData/Users/岳伯禹/Documents/WeChat Files/wxid_7a4ye1q6cjvf32/FileStorage/Temp/1746247292939.png" alt="1746247292939" style="zoom:60%;" />

For the test results, $test\:RMSE=0.06669$, and $test\:r^2=0.54930$.

## Conclusion

From the training results, we can see that using LSTM, we can predict future pollution relatively accurately based on historical weather data. LSTM is very suitable for time-series prediction tasks. Adjusting the hyperparameters and network structure can make predictions more accurate.

 