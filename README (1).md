# Adaptive Video Streaming with Deep Reinforcement Learning

**Research internship at ArvanCloud (CDN and cloud provider), Tehran, summer 2026 (3 months)**

> The source code is kept private. This page summarises the problem, my approach and the results.
> I am happy to explain the implementation in detail during an interview.

## The problem

In DASH video streaming, each video is encoded at several bitrates and cut into short segments. Before downloading each segment, the player's **adaptive bitrate (ABR)** algorithm chooses which bitrate to fetch:

- too high, and the buffer empties and the video freezes (**rebuffering**);
- too low, and the viewer gets a poor image although the network could do better;
- switching too often makes playback feel unstable.

Classical ABR algorithms rely on hand-tuned rules based on throughput or buffer level. This project **learns an ABR policy with a Deep Q-Network (DQN)** and compares it with two classical baselines.

## Key results

Evaluated on 10 network traces of 200 segments each:

| Algorithm | Avg QoE | Avg bitrate | Rebuffering | Quality switches |
|---|---|---|---|---|
| Rate-based ABR | 94.66 | 2265 kbps | 8.18 s | 76.4 |
| Buffer-based ABR | −416.91 | 2268 kbps | 153.00 s | 99.1 |
| **DQN (mine)** | **296.29** | **2520 kbps** | **0.61 s** | **61.2** |

Compared with rate-based ABR, the DQN agent achieves:

- **+213 % QoE**
- **−93 % rebuffering** (8.18 s → 0.61 s)
- **+11 % average bitrate** (higher visual quality)
- **−20 % quality switches** (smoother playback)
- the best QoE on **all 10** evaluation traces

## Pipeline

```
Raw video ─► FFmpeg encoding (5 levels) ─► DASH packaging ─► Flask HTTP server
                                                                      │
                                                                      ▼
                                       Gym-like streaming environment (simulator)
                                                                      │
                          ┌───────────────────────┬───────────────────┴───────────┐
                          ▼                       ▼                               ▼
                   Rate-based ABR          Buffer-based ABR                 DQN agent
                          └───────────────► Pensieve-style QoE model ◄─────────────┘
```

## Method

**State** (3 features, normalised):
`[ throughput (Mbps), buffer level (s) / 10, last bitrate / max bitrate ]`

**Action** (5 discrete): the quality of the next segment, among 5 levels from 240p to 1080p ({400, 800, 1500, 3000, 6000} kbps).

**Reward** (per segment, Pensieve-style QoE):

```
r_t = log(b_t / b_min) − 4.3 · rebuffer_t − | log(b_t / b_min) − log(b_{t−1} / b_min) |
```

quality bonus − rebuffering penalty − smoothness penalty.

**DQN training details:**

- target network synchronised every 500 steps
- experience replay buffer (10,000 transitions)
- Huber loss and gradient clipping (max-norm 10)
- ε-greedy exploration decaying from 1.0 to 0.05
- Adam optimiser (learning rate 5e-4)
- 600 episodes of 200 segments, trained on the company GPU

**Network traces:** 10 real network bandwidth traces (mean 300 to 7000 kbps).

## Why the DQN agent wins

- It **anticipates bandwidth drops** instead of reacting after rebuffering has started, as rate-based ABR does.
- It **uses the buffer as a safety cushion**, filling it when the network is good so it can keep a higher bitrate during dips.
- It **balances quality, rebuffering and smoothness together**, while each heuristic mainly focuses on one signal.
- Buffer-based ABR performed poorly here because its fixed thresholds were not calibrated for these traces, a known limitation that motivated learning-based approaches like Pensieve.

## Limitations and next steps

- Evaluated in simulation on recorded traces; next steps are larger public datasets (FCC broadband, Norway 3G/4G) and tests with real viewers.
- Compare with stronger baselines (MPC, BOLA) and policy-gradient methods (A3C, PPO, as in Pensieve).
- Integrate the trained agent into a real dash.js player (for example through ONNX inference).

## Tech stack

Python · PyTorch · NumPy · FFmpeg · H.264 · MPEG-DASH · Flask · Matplotlib · Git

## References

- Mao et al., *Neural Adaptive Video Streaming with Pensieve*, SIGCOMM 2017
- Yin et al., *A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP* (MPC), SIGCOMM 2015
- Huang et al., *A Buffer-Based Approach to Rate Adaptation* (BBA), SIGCOMM 2014
- Mnih et al., *Human-level Control through Deep Reinforcement Learning*, Nature 2015

## Contact

Sevil Khojasteh Nejat · M2 Multimedia Networking, Université Paris-Saclay · [LinkedIn](https://www.linkedin.com/in/sevil-khojasteh-nejat)
