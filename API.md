# API

Everything below is exported by `using WiFiAssignment`. Start Julia with `-t 16` (or `-t auto`) so the subset
tables build in parallel; the thread count never changes a result.

```julia
using WiFiAssignment

world = make_world(seed=42, n=10, cpa=4)
assignment, score = optimize_assignment(world; seed=1)
results = compare_voice_strategies(world, assignment)
```

- [Worlds](#worlds): `SimConfig`, `make_world`, `World`
- [Scoring an assignment](#scoring-an-assignment): `client_rate`, `client_rates`, `total_score`
- [Choosing channels](#choosing-channels): `optimize_assignment`, `brute_force_fast`, `greedy_rate`,
  `simulated_annealing` + `AnnealConfig`, `naive_coloring`, `brute_force_naive`
- [Voice](#voice): `VoiceSimConfig`, `simulate_voice`, `compare_voice_strategies`, `VoiceResult`
- [Errors and guarantees](#errors-and-guarantees)

An **assignment** is a `Vector{Int}` with one channel per AP, each in `1:k`. Every search function returns its
channels numbered in order of first appearance (AP 1 is always on channel 1), so equal partitions compare equal.

## Worlds

### `SimConfig`

Radio and traffic settings, built with keywords: `SimConfig(k_channels=4, demand_mbps=3.0)`.

| Field | Default | Meaning |
|---|---:|---|
| `area` | 100.0 | side of the square floor, in metres |
| `noise_dbm` | -95.0 | noise floor |
| `k_channels` | 3 | channels available |
| `bw_mhz` | 20.0 | channel bandwidth; PHY rate is `bw · log2(1 + SINR)` |
| `min_rate` | 0.5 | rate floor in Mbps, so a starved client never scores `log(0)` |
| `shadow_sigma` | 4.0 | log-normal shadowing per link, in dB |
| `p0` | -30.0 | received power at 1 m, in dBm |
| `alpha` | 35.0 | path loss in dB per decade of distance (exponent 3.5) |
| `demand_mbps` | 40.0 | what each client wants; rates are capped here |
| `mac_efficiency` | 0.72 | share of the PHY rate left after MAC overhead |
| `max_phy_mbps` | 144.0 | PHY rate cap |
| `cca_dbm` | -82.0 | carrier-sense threshold: APs heard above it share airtime, below it they interfere |

### `make_world(; seed=nothing, cfg=SimConfig(), n=10, cpa=4) -> World`

Drops `n` APs (1 to 18) uniformly on the floor and `cpa` clients around each, 5 to 15 m from their AP, then draws
the shadowing for every link. The same `seed` always gives the same world.

Useful `World` fields: `n`, `cfg`, `client_ap` (the AP of each client; clients of AP `i` are
`(i-1)*cpa+1 : i*cpa`), `ap_activity` (airtime each AP needs on a clean channel, capped at 1), and the link tables
`AP_AP` and `C_AP` in dBm.

## Scoring an assignment

### `client_rate(W, c, assign)`, `client_rates(W, assign)`

The rate a client gets, in Mbps:

1. **SINR.** Signal from its own AP over noise plus every *hidden* co-channel AP (one it cannot carrier-sense),
   each weighted by how much of the time that AP is transmitting.
2. **PHY rate.** `bw · log2(1 + SINR)`, capped at `max_phy_mbps`.
3. **Airtime.** Scaled by `mac_efficiency`, then shared with the AP's other clients and with the busy time of
   every co-channel AP it *does* sense.
4. **Limits.** Clamped to `[min_rate, demand_mbps]`.

When APs are not saturated (light demand), how busy an AP is depends on the interference it suffers, which depends
on how busy the others are. That loop is solved as a fixed point before any rate is computed.

### `total_score(W, assign)`

Proportional fairness: the sum of `log(rate)` over all clients, in nats. Higher is better: one more nat means the
product of all client rates grew by a factor of e, so starving one client costs more than it gains elsewhere. This
is what every search maximises.

## Choosing channels

All of these share one table that scores every subset of APs placed on a common channel (2ⁿ entries), so after
the table is built a search is cheap. `k` defaults to `cfg.k_channels` and may be anything from 1 to it.

| Function | Returns | Use it for |
|---|---|---|
| `optimize_assignment(W; k, seed, voice)` | `(assign, score)` | the default; hit the exact optimum in all 270 benchmark worlds |
| `brute_force_fast(W; k)` | `(assign, score)` | the guaranteed optimum, any `k`, up to 18 APs |
| `greedy_rate(W; k, refine=true)` | `assign` | a fast starting point |
| `simulated_annealing(W; k, seed, cfg)` | `(assign, score, history)` | comparing against annealing |
| `naive_coloring(W; thresh=-75.0, k)` | `assign` | a graph-colouring baseline |
| `brute_force_naive(W)` | `(assign, score)` | a slow reference, at most a million assignments |

### `optimize_assignment(W; k=cfg.k_channels, seed=nothing, voice=nothing)`

Iterated local search. It starts from the greedy assignment, polishes it with single-AP moves and pairwise swaps
until nothing improves, then 200 times scrambles 2 to 4 APs and polishes again, keeping the result whenever it is
no worse. The whole thing restarts 8 times from random greedy orders and the best result wins.

With `voice=VoiceSimConfig(...)`, every voice flow joins the objective too, as `log(delivered codec rate / 64 kbps)`
under that config's policy, so the search trades data throughput against call quality.

### `brute_force_fast(W; k=cfg.k_channels)`

Exact dynamic program over subsets: the best way to split any set of APs into `j` channels is one channel plus the
best split of the rest into `j - 1`. With `k >= n` every AP simply gets its own channel. Cost grows like 3ⁿ, about
0.06 s at 18 APs and 3 channels with heavy demand.

### `greedy_rate(W; k, refine=true)`

Places APs one at a time, most-interfering first, each on the channel that raises the score most. `refine=false`
gives the plain one-pass version.

### `simulated_annealing(W; k, seed, cfg=AnnealConfig())`

Random single-AP moves, accepted by the Metropolis rule while the temperature cools. `history` is the score after
every step.

| `AnnealConfig` field | Default | Meaning |
|---|---:|---|
| `iters` | 3000 | number of moves |
| `T0` | `nothing` | starting temperature; `nothing` picks one where an average downhill move is accepted half the time |
| `cooling` | `nothing` | factor per step; `nothing` cools to 1% of `T0` over the run |
| `polish` | `true` | finish with the same local search as `optimize_assignment` |

## Voice

### `VoiceSimConfig`

| Field | Default | Meaning |
|---|---:|---|
| `duration_s` | 20.0 | simulated time |
| `frame_ms` | 20.0 | one packet per flow per frame |
| `deadline_ms` | 150.0 | a packet still queued after this is lost |
| `calls_per_ap` | 32 | two-way calls, so twice as many flows |
| `codec_kbps` | 64.0 | nominal codec rate |
| `min_codec_kbps` | 12.0 | the adaptive codec never goes below this |
| `packet_overhead_bytes` | 60 | headers per packet |
| `background_load` | 0.95 | data traffic, as a fraction of each AP's airtime |
| `retry_limit` | 2 | retransmissions before a collided packet is dropped |
| `adaptation_speed` | 0.35 | how far the codec moves toward its target each frame |
| `policy` | `:combined` | queueing policy, see below |

`VoiceSimConfig(cfg; policy=:fifo)` copies `cfg` with some fields changed, and rejects field names it does not have.

| Policy | Queue | Codec |
|---|---|---|
| `:fifo` | voice waits behind data | fixed |
| `:deadline` | voice first | fixed |
| `:adaptive` | voice waits behind data | shrinks to fit the airtime left |
| `:combined` | voice first | shrinks to fit the airtime left |

### `simulate_voice(W, assign; cfg=VoiceSimConfig()) -> VoiceResult`

Steps the network frame by frame. Each AP's airtime comes from the client rates above. Collisions come from
Bianchi's DCF model among the co-channel APs it senses, plus overlap with hidden ones, unless the frame beats them
by the 10 dB capture margin. Failed transmissions are retried up to `retry_limit` times.

| `VoiceResult` field | Meaning |
|---|---|
| `policy` | the policy that was simulated |
| `packets` | voice packets offered |
| `delivered_packets` | packets that arrived in time |
| `loss_fraction` | share lost, `deadline_loss_fraction + retry_loss_fraction` |
| `mean_delay_ms`, `p95_delay_ms` | queueing delay of delivered packets |
| `mean_codec_kbps` | average codec rate across APs and frames |
| `delivered_mbps` | voice payload delivered |

### `compare_voice_strategies(W, assign; cfg) -> Vector{VoiceResult}`

Runs all four policies on the same config, in the order `:fifo, :deadline, :adaptive, :combined`, whatever
`cfg.policy` says.

## Errors and guarantees

- Bad input (an AP count outside 1 to 18, a channel outside `1:k`, a wrong-length assignment, a negative load, an
  unknown policy, and so on) throws an `ArgumentError` that says what was wrong.
- Every fixed point (airtime, Bianchi, voice steady state) throws instead of returning an unconverged number.
- Same inputs and seed give the same output, on any number of threads.
- `seed=nothing` uses Julia's global RNG.
