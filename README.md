## <u> Reinforcement Learning </u>

This repository contains implementations of various Reinforcement Learning algorithms. Each subdirectory corresponds to a specific lecture and includes code examples, explanations, and relevant environments.

## Folder Structure

```bash
.
├── 01-DynamicProgramming
│   ├── 01-IterativePolicyEvaluation
│   └── 02-ValueIteration
├── 02-MonteCarlo
│   ├── 01-MonteCarloPrediction
│   ├── 02-MonteCarloExploringStarts
│   ├── 03-OnPolicyMonteCarlo
│   └── 04-OffPolicyMonteCarlo
├── 03-TemporalDifference
│   ├── 01-TDPrediction
│   ├── 02-SARSA
│   ├── 03-Q-Learning
│   └── 04-Double-Q-Learning
├── 04-nStepBootstrap
│   ├── 01-nPrediction
│   ├── 02-nSARSA
│   └── 03-nTreeBackup
├── 05-FunctionApproximation
│   ├── model.py
│   ├── test.py
│   ├── tile_coding.py
│   └── train.py
├── 06-DeepQNetwork
│   ├── model.py
│   ├── test.py
│   └── train.py
├── 07-PolicyGradient
│   ├── model.py
│   ├── test.py
│   └── train.py
├── 08-ProximalPolicyOptimization
│   ├── model.py
│   ├── test.py
│   └── train.py
├── 09-MonteCarloTreeSearch
│   ├── model.py
│   └── train.py
├── LICENSE
├── README.md
├── pseudocode
└── requirements.txt
```

## Environments 

```markdown
.
├── 01-DynamicProgramming `GridWorldEnv`
├── 02-MonteCarlo `GridWorldEnv`
├── 03-TemporalDifference `GridWorldEnv`
├── 04-nStepBootstrap `GridWorldEnv`
├── 05-FunctionApproximation `MountainCarContinuous-v0`
├── 06-DeepQNetwork `CartPole-v1`
├── 07-PolicyGradient `CartPole-v1`
├── 08-ProximalPolicyOptimization `CartPole-v1`
└── 09-MonteCarloTreeSearch `CartPole-v1`
```

## Reinforcement Learning Algorithms

### 01 Dynamic Programming - Iterative Policy Evaluation

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/4-1.png" width="100%">
    </td>
  </tr>
</table>

### 01 Dynamic Programming - Value Iteration

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/4-1.png" width="100%">
    </td>
  </tr>
</table>

### 02 Monte Carlo - Monte Carlo Prediction

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/5-1.png" width="100%">
    </td>
  </tr>
</table>

### 02 Monte Carlo - Monte Carlo Exploring Starts

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/6-1.png" width="100%">
    </td>
  </tr>
</table>

### 02 Monte Carlo - Monte Carlo On-Policy

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/5-1.png" width="100%">
    </td>
  </tr>
</table>

### 02 Monte Carlo - Monte Carlo Off-Policy

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/6-1.png" width="100%">
    </td>
  </tr>
</table>

### 03 Temporal Difference - Temporal Difference Prediction

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/5-1.png" width="100%">
    </td>
  </tr>
</table>

### 03 Temporal Difference - SARSA

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/6-1.png" width="100%">
    </td>
  </tr>
</table>

### 03 Temporal Difference - Q-Learning

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/5-1.png" width="100%">
    </td>
  </tr>
</table>

### 03 Temporal Difference - Double Q-Learning

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/6-1.png" width="100%">
    </td>
  </tr>
</table>

### 04 n-Step Bootstrapping - n-Step Prediction

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/6-1.png" width="100%">
    </td>
  </tr>
</table>

### 04 n-Step Bootstrapping - n-Step SARSA

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/5-1.png" width="100%">
    </td>
  </tr>
</table>

### 04 n-Step Bootstrapping - n-Step Tree Backup

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/6-1.png" width="100%">
    </td>
  </tr>
</table>

### 05 Function Approximation - Semi-Gradient SARSA

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/7-1.png" width="100%">
    </td>
  </tr>
</table>

### 06 Deep Q-Network

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/8-1.png" width="100%">
    </td>
  </tr>
</table>

### 07 Policy Gradient

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/9-1.png" width="100%">
    </td>
  </tr>
</table>

### 08 Proximal Policy Optimization: Clip

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/10-1.png" width="100%">
    </td>
  </tr>
</table>

### 09 Monte Carlo Tree Search

<table>
  <tr>
    <td style="width: 50%;">
      <img src="images/11-1.png" width="100%">
    </td>
  </tr>
</table>

## Requirements

```bash
pip install -r requirements.txt
```

## References

- Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto ([Web Link](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf))
- The Reinforcement Learning Course by Hugging Face ([Web Link](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction))
- Spinning Up in Deep RL by OpenAI ([Web Link](https://spinningup.openai.com/en/latest/))
- OpenAI Gymnasium API documentation ([Web Link](https://gymnasium.farama.org/index.html))
- Tensorflow Python API documentation ([Web Link](https://www.tensorflow.org/api_docs/python/tf/all_symbols))

## Usage

Each subdirectory contains its own set of implementations and explanatory notes. To explore the implementations and learn more about each concept, navigate to the respective subdirectory's README.md file.

Feel free to explore, experiment, and contribute to this open source project.
