# wifi_assignment

Which channel should each WiFi access point use, and what does that do to voice calls? This is a Julia model of
that problem for up to 18 APs. Each AP gets one of k channels. A client's rate depends on path loss and shadowing,
how much airtime its AP gets after carrier sense, interference from hidden APs, and how much the client actually
wants. An assignment is scored by proportional fairness, the sum of log rates over all clients.

The score splits into one term per channel group. That lets everything share a table over all 2ⁿ subsets of APs,
and a dynamic program over subsets gives the exact optimum for any number of channels, so I can check the
heuristics against the real answer instead of against each other.

On top of that there's a voice model: two-way calls with a 150 ms deadline, background traffic, retries, Bianchi's
DCF fixed point with 10 dB capture, and four queueing policies (FIFO or deadline priority, fixed or adaptive
codec). The optimizer can also aim for voice goodput instead of throughput.

The write-ups are in [`paper/results.pdf`](paper/results.pdf) (results) and [`paper/model.pdf`](paper/model.pdf)
(the model, the algorithms and how I checked them).

## Running it

Needs Julia 1.13. The package, tests and benchmarks are a Julia workspace with one shared manifest.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 16 --project=test test/runtests.jl

julia -t 16 --project=benchmarks benchmarks/stress.jl   # independent oracles, exhaustive k = 1..5
julia -t 16 --project=benchmarks benchmarks/dense.jl    # 18-AP worlds against a separate exact DP
julia -t 16 --project=benchmarks benchmarks/voice.jl    # voice scenarios and paired sign tests

julia -t 16 --project=benchmarks paper/results.jl       # ~2.5 min, writes paper/generated/
cd paper && latexmk -pdf results.tex model.tex
```

The thread count only changes speed. Results come out bit-identical either way.

```julia
using WiFiAssignment

world = make_world(seed=42, n=10, cpa=4, cfg=SimConfig(k_channels=4))
assignment, score = optimize_assignment(world; seed=1)
exact, best = brute_force_fast(world)

voice = VoiceSimConfig(calls_per_ap=32, background_load=0.8)
results = compare_voice_strategies(world, assignment; cfg=voice)
tuned, _ = optimize_assignment(world; voice=voice, policy=:combined)
```

## Code

`src/WiFiAssignment.jl` (~890 lines) is the whole package. It has the world model and the airtime fixed point,
the subset table (a threaded fast path for saturated worlds, a per-subset fixed point otherwise), graph colouring,
greedy with relocate/swap moves, simulated annealing, `optimize_assignment` (iterated local search, 8 restarts ×
200 kicks), the exact DP in `brute_force_fast`, and the voice simulator with its MAC model.

The fixed points throw an error if they don't converge rather than handing back a half-finished number.
`test/runtests.jl` has the unit tests. `benchmarks/stress.jl` and `benchmarks/dense.jl` recheck things against
code written separately: rates in dB, a Jacobi fixed point, brute-force search, a slot-level DCF simulation and a
closed-form queue.

## Results

From `paper/results.jl` on a Ryzen 7 7840HS, 16 threads.

### Assignment quality

270 worlds (10, 14 and 18 APs; 4 or 16 clients per AP), each compared with the exact optimum. The table is the
share of worlds where a method hit the optimum, by number of channels. "Light" means 3 Mbps demand, where APs
aren't saturated.

| Method | 2 | 3 | 4 | 6 | 3, light |
|---|---:|---:|---:|---:|---:|
| Graph colouring | 2 % | 2 % | 0 % | 0 % | 10 % |
| Greedy, single pass (original) | 2 % | 0 % | 3 % | 3 % | 37 % |
| Greedy + local search | 63 % | 38 % | 30 % | 43 % | 73 % |
| Annealing, calibrated T₀ + polish | 100 % | 100 % | 92 % | 80 % | 97 % |
| `optimize_assignment` | 100 % | 100 % | 100 % | 100 % | 100 % |

`optimize_assignment`'s worst gap was 1.5e-12 nats, i.e. rounding. The original greedy was 9.98 nats short in
the median and 89.59 at worst.

### Speed

Median time in ms, 16 clients per AP, 3 channels, heavy demand:

| APs | Original table | New table | Optimize | Exact |
|---:|---:|---:|---:|---:|
| 12 | 18.3 | 6.3 | 7.6 | 6.3 |
| 14 | 93.3 | 6.1 | 14.0 | 11.4 |
| 16 | 401.2 | 29.4 | 24.8 | 24.2 |
| 18 | 1799.7 | 73.8 | 72.4 | 97.7 |

The new table matches the old one to 4e-13. Light-demand worlds need the fixed point per subset and take up to
~1.8 s at 18 APs. Timings on my laptop jump around by up to 2× between runs, so read these as rough.

### MAC model

Checked against a slot-level DCF simulation, Bianchi's collision probability is off by at most 0.005 for up to 4
contenders (the most the voice scenarios hit) and 0.018 up to 10. The hidden-AP overlap formula is within 0.002.
The ALOHA formula I started with was off by as much as 0.218, which is why it's gone.

### Voice

20 worlds per scenario, 10 s simulated each:

| Scenario | Policy | Loss (median) | p95 delay | Codec |
|---|---|---:|---:|---:|
| Call centre | `:fifo` | 11.73 % | 140.0 ms | 64.0 kbps |
| | `:combined` | 1.19 % | 8.5 ms | 64.0 kbps |
| Overload | `:fifo` | 40.85 % | 140.0 ms | 64.0 kbps |
| | `:adaptive` | 20.85 % | 140.0 ms | 12.2 kbps |
| | `:combined` | 0.53 % | 37.6 ms | 59.6 kbps |

Shrinking the codec on its own doesn't save calls under overload: loss stays at 21 % and the codec drops to
12 kbps. Deadline priority plus adaptation gets loss under 1 % and keeps the codec near 60 kbps.

Paired over 50 worlds: deadline scheduling beat adaptation alone on loss in 49 (one tie) and on p95 delay in all
50, by 136.4 ms in the median. Under overload the combined policy beat fixed-rate priority in 50/50, and at
normal load it kept at least 99 % of the codec rate in 200/200 runs.

Pointing the optimizer at voice goodput instead of throughput improved goodput in up to 19 of 20 worlds per
scenario and never made it worse. The cost to data clients was at most 0.084 nats per client in the median.

## Limits

- The worlds, traffic and shadowing are synthetic and don't change during a run.
- The voice side is an aggregate queue model, not a packet-level 802.11 simulator. Partly busy APs count as
  fractional contenders in Bianchi, and an AP's own clients count as a single contender, so collisions in busy
  call cells are probably underestimated.
- Exact search stops at 18 APs.
- None of this has been checked against ns-3, real traces or hardware yet.
