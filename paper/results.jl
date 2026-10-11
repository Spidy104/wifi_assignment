using WiFiAssignment, Random, Statistics, Printf
const WA = WiFiAssignment
include(joinpath(@__DIR__, "..", "benchmarks", "voice.jl"))

const OUT = joinpath(@__DIR__, "generated")
const NUMBERS = String[]
const MAX_CONTENDERS = Ref(0)

number!(name, value) = push!(NUMBERS, "\\newcommand{\\$name}{$value}")
fmt(x, digits=2) = @sprintf("%.*f", digits, x)
pct(x, digits=1) = fmt(100x, digits) * "\\,\\%"
med(f, repeats=5) = (f(); f(); median([@elapsed(f()) for _ in 1:repeats]))

function table!(name, header, rows; align="l" * "r"^(length(header) - 1))
    open(joinpath(OUT, name * ".tex"), "w") do io
        println(io, "\\begin{tabular}{", align, "}\n\\toprule\n", join(header, " & "), " \\\\\n\\midrule")
        foreach(row -> println(io, join(row, " & "), " \\\\"), rows)
        println(io, "\\bottomrule\n\\end{tabular}")
    end
end

function series!(name, header, rows)
    open(joinpath(OUT, name * ".dat"), "w") do io
        println(io, join(header, " "))
        foreach(row -> println(io, join(row, " ")), rows)
    end
end

quality_config(n, k, demand) =
    SimConfig(k_channels=k, area=100.0 * sqrt(n / 18), shadow_sigma=8.0, min_rate=0.001, demand_mbps=demand)

const METHODS = [
    "Graph colouring" => (W, s) -> naive_coloring(W),
    "Greedy, single pass (original)" => (W, s) -> greedy_rate(W; refine=false),
    "Greedy with local search" => (W, s) -> greedy_rate(W),
    "Annealing, fixed \$T_0 = 5\$, no polish" =>
        (W, s) -> simulated_annealing(W; seed=s, cfg=AnnealConfig(iters=50_000, T0=5.0, polish=false))[1],
    "Annealing, calibrated \$T_0\$, polish" => (W, s) -> simulated_annealing(W; seed=s, cfg=AnnealConfig(iters=50_000))[1],
    "\\texttt{optimize\\_assignment}" => (W, s) -> optimize_assignment(W; seed=s)[1]]

function assignment_quality()
    regimes = [("2", 2, 40.0, 1:10), ("3", 3, 40.0, 1:10), ("4", 4, 40.0, 1:10), ("6", 6, 40.0, 1:10),
               ("3, light", 3, 3.0, 1:5)]
    foreach(((_, f),) -> f(make_world(n=6, cpa=2, seed=1, cfg=quality_config(6, 3, 40.0)), 1), METHODS)
    gaps = [Float64[] for _ in METHODS, _ in regimes]
    for (r, (_, k, demand, seeds)) in enumerate(regimes), n in (10, 14, 18), cpa in (4, 16), seed in seeds
        W = make_world(n=n, cpa=cpa, seed=seed, cfg=quality_config(n, k, demand))
        optimum = brute_force_fast(W)[2]
        for (m, (_, f)) in enumerate(METHODS)
            push!(gaps[m, r], max(optimum - total_score(W, f(W, seed)), 0.0))
        end
    end
    header = ["Method"; ["\\multicolumn{1}{c}{$label}" for (label, _) in regimes]]
    table!("optimal", header, [[name; [pct(count(<(1e-9), gaps[m, r]) / length(gaps[m, r]), 0) for r in eachindex(regimes)]]
                               for (m, (name, _)) in enumerate(METHODS)])
    table!("gaps", header, [[name; ["$(fmt(median(gaps[m, r]))) ($(fmt(maximum(gaps[m, r]))))" for r in eachindex(regimes)]]
                            for (m, (name, _)) in enumerate(METHODS)])
    number!("qualityWorlds", sum(length(gaps[1, r]) for r in eachindex(regimes)))
    number!("optimizeWorstGap", @sprintf("%.1e", maximum(maximum(gaps[end, r]) for r in eachindex(regimes))))
    number!("greedyPlainMedianGap", fmt(median(reduce(vcat, gaps[2, :]))))
    number!("greedyPlainWorstGap", fmt(maximum(reduce(vcat, gaps[2, :]))))
end

function original_subset_scores(W)
    count = 1 << W.n
    scores, interference, blockers = zeros(count), Vector{Float64}(undef, count), Vector{Float64}(undef, count)
    @inbounds for c in eachindex(W.client_ap)
        own = W.client_ap[c]
        signal = W.C_AP_mW[c, own]
        interference[1], blockers[1] = W.noise_mw, 0.0
        for mask in 1:(count - 1)
            bit, prev = trailing_zeros(mask) + 1, mask & (mask - 1)
            sensed = bit != own && W.AP_AP[own, bit] >= W.cfg.cca_dbm
            interference[mask + 1] = interference[prev + 1] +
                                     (bit == own || sensed ? 0.0 : W.C_AP_mW[c, bit] * W.ap_activity[bit])
            blockers[mask + 1] = blockers[prev + 1] + (sensed ? W.ap_activity[bit] : 0.0)
            if mask & (1 << (own - 1)) != 0
                phy = min(W.cfg.max_phy_mbps, W.rate_scale * log1p(signal / interference[mask + 1]))
                capacity = W.cfg.mac_efficiency * phy / (W.clients_per_ap * (1.0 + blockers[mask + 1]))
                scores[mask + 1] += log(min(W.cfg.demand_mbps, max(capacity, W.cfg.min_rate)))
            end
        end
    end
    return scores
end

function speed()
    rows, deviation = [], 0.0
    voice = VoiceSimConfig(calls_per_ap=64, background_load=0.75)
    for (regime, demand) in (("heavy", 40.0), ("light", 3.0)), n in (12, 14, 16, 18)
        W = make_world(n=n, cpa=16, seed=1, cfg=quality_config(n, 3, demand))
        saturated = all(>=(1.0), W.ap_activity)
        original = saturated ? med(() -> original_subset_scores(W), 3) : NaN
        saturated && (deviation = max(deviation, maximum(abs.(original_subset_scores(W) .- WA.subset_scores(W)))))
        ms(t) = isnan(t) ? "--" : fmt(1000t, 1)
        push!(rows, [regime, n, ms(original), ms(med(() -> WA.subset_scores(W), 3)),
                     ms(med(() -> optimize_assignment(W; seed=1), 3)), ms(med(() -> brute_force_fast(W), 3)),
                     ms(med(() -> optimize_assignment(W; seed=1, voice=voice), 3))])
    end
    table!("speed", ["Demand", "APs", "Original table", "New table", "Optimize", "Exact", "Voice-aware"], rows; align="lrrrrrr")
    number!("threads", Threads.nthreads())
    number!("originalTableDeviation", @sprintf("%.0e", deviation))
end

function simulate_dcf(n; slots=2_000_000, rng=Xoshiro(n))
    stage, counter = zeros(Int, n), [rand(rng, 0:(WA.CW_MIN - 1)) for _ in 1:n]
    attempts = collisions = 0
    for _ in 1:slots
        ready = count(==(0), counter)
        if ready == 0
            counter .-= 1
            continue
        end
        attempts += ready
        ready > 1 && (collisions += ready)
        for i in 1:n
            counter[i] == 0 || continue
            stage[i] = ready > 1 ? min(stage[i] + 1, WA.BACKOFF_STAGES) : 0
            counter[i] = rand(rng, 0:(WA.CW_MIN * 2^stage[i] - 1))
        end
    end
    return collisions / attempts
end

function simulate_hidden(u; trials=200_000, rng=Xoshiro(round(Int, 1000u)))
    overlaps = 0
    for _ in 1:trials
        t, start, hit = -200.0, rand(rng), false
        while t < 2.0
            hit |= t < start + 1.0 && t + 1.0 > start
            t += 1.0 + randexp(rng) * (1 - u) / u
        end
        overlaps += hit
    end
    return overlaps / trials
end

function mac_validation()
    bianchi = [(n, WA.dcf_collision(Float64(n)), simulate_dcf(n)) for n in 2:10]
    series!("bianchi", ["n", "model", "simulation"], [[n, fmt(m, 5), fmt(s, 5)] for (n, m, s) in bianchi])
    hidden = [(u, WA.hidden_collision(u), simulate_hidden(u), -expm1(-2u)) for u in 0.1:0.1:0.9]
    series!("hidden", ["u", "model", "simulation", "aloha"], [[fmt(u, 1), fmt(m, 5), fmt(s, 5), fmt(a, 5)] for (u, m, s, a) in hidden])
    number!("bianchiErrorScenario", fmt(maximum(abs(m - s) for (n, m, s) in bianchi if n <= MAX_CONTENDERS[]), 3))
    number!("bianchiErrorLarge", fmt(maximum(abs(m - s) for (n, m, s) in bianchi), 3))
    number!("hiddenError", fmt(maximum(abs(m - s) for (_, m, s, _) in hidden), 3))
    number!("alohaError", fmt(maximum(abs(a - s) for (_, _, s, a) in hidden), 3))
end

function estimate_accuracy()
    rows, worst_loss, worst_codec = [], 0.0, 0.0
    for sc in SCENARIOS
        cells = String[]
        for policy in POLICIES
            loss_err, codec_err = 0.0, 0.0
            for seed in 1:10
                W = make_world(seed=seed, cfg=SimConfig(k_channels=3, area=sc.area), n=sc.aps, cpa=sc.stations)
                for assign in (greedy_rate(W), rand(Xoshiro(seed), 1:3, sc.aps))
                    cfg = VoiceSimConfig(duration_s=20.0, calls_per_ap=sc.calls, background_load=sc.load, policy=policy)
                    bitrate, delivered = WA.voice_steady_state(WA.VoiceModel(W, cfg), assign, WA.activity(W, assign))
                    r = simulate_voice(W, assign; cfg=cfg)
                    loss_err = max(loss_err, abs(1 - mean(delivered) - r.loss_fraction))
                    codec_err = max(codec_err, abs(mean(bitrate) - r.mean_codec_kbps))
                end
            end
            worst_loss, worst_codec = max(worst_loss, loss_err), max(worst_codec, codec_err)
            push!(cells, "$(fmt(100loss_err)) / $(fmt(codec_err))")
        end
        push!(rows, [sc.name; cells])
    end
    table!("estimate", ["Scenario"; ["\\texttt{:$p}" for p in POLICIES]], rows)
    number!("estimateLossError", fmt(100worst_loss))
    number!("estimateCodecError", fmt(worst_codec))
end

function isolated_queue()
    worst_loss, worst_codec, cases = 0.0, 0.0, 0
    for cpa in (2, 8), load in (0.3, 0.9, 1.4), calls in (16, 256)
        W = make_world(n=1, cpa=cpa, seed=cpa, cfg=SimConfig(k_channels=1))
        cfg = VoiceSimConfig(duration_s=60.0, calls_per_ap=calls, background_load=load)
        raw = sum(client_rates(W, [1])) * 1000 * cfg.frame_ms
        overhead = 8 * cfg.packet_overhead_bytes
        offered(kbps) = 2calls * (kbps * cfg.frame_ms + overhead)
        fit(airtime) = clamp((max(airtime, 0) / 2calls - overhead) / cfg.frame_ms, cfg.min_codec_kbps, cfg.codec_kbps)
        codec = Dict(:fifo => cfg.codec_kbps, :deadline => cfg.codec_kbps, :adaptive => fit(raw - load * raw), :combined => fit(raw))
        background = Dict(:fifo => load * raw, :deadline => 0.0, :adaptive => load * raw, :combined => 0.0)
        for r in compare_voice_strategies(W, [1]; cfg=cfg)
            worst_codec = max(worst_codec, abs(r.mean_codec_kbps - codec[r.policy]))
            worst_loss = max(worst_loss, abs(r.loss_fraction - max(0.0, 1 - raw / (offered(codec[r.policy]) + background[r.policy]))))
            cases += 1
        end
    end
    number!("isolatedCases", cases)
    number!("isolatedLossError", fmt(100worst_loss))
    number!("isolatedCodecError", fmt(worst_codec))
end

function voice_objective_exactness()
    worst, cases = 0.0, 0
    for seed in 1:10, policy in POLICIES, (calls, load) in ((128, 0.8), (32, 1.05))
        W = make_world(n=7, cpa=4, seed=seed, cfg=SimConfig(k_channels=3, area=60.0))
        cfg = VoiceSimConfig(calls_per_ap=calls, background_load=load, policy=policy)
        model = WA.VoiceModel(W, cfg)
        objective(a) = total_score(W, a) + WA.voice_utility(model, a, WA.activity(W, a))
        best = maximum(objective(digits(code; base=3, pad=7) .+ 1) for code in 0:(3^7 - 1))
        found, score = optimize_assignment(W; seed=seed, voice=cfg)
        worst = max(worst, abs(score - best), best - objective(found))
        cases += 1
    end
    number!("voiceExactCases", cases)
    number!("voiceExactGap", @sprintf("%.0e", worst))
end

scenario_world(sc, seed) = make_world(seed=seed, cfg=SimConfig(k_channels=3, area=sc.area), n=sc.aps, cpa=sc.stations)
scenario_voice(sc) = VoiceSimConfig(duration_s=10.0, calls_per_ap=sc.calls, background_load=sc.load)

function voice_outcomes()
    rows, contenders = [], 0
    for sc in SCENARIOS
        results = Dict(p => VoiceResult[] for p in POLICIES)
        for seed in 1:20
            W = scenario_world(sc, seed)
            assign = optimize_assignment(W; seed=seed)[1]
            sensed, _ = WA.mac_tables(W)
            contenders = max(contenders, maximum(1 + count(j -> assign[j] == assign[ap] && sensed[ap, j], 1:W.n)
                                                 for ap in 1:W.n))
            for r in compare_voice_strategies(W, assign; cfg=scenario_voice(sc))
                push!(results[r.policy], r)
            end
        end
        for p in POLICIES
            rs = results[p]
            push!(rows, [p == :fifo ? sc.name : "", "\\texttt{:$p}", pct(median(r.loss_fraction for r in rs), 2),
                         pct(maximum(r.loss_fraction for r in rs), 2), fmt(median(r.p95_delay_ms for r in rs), 1),
                         fmt(median(r.mean_codec_kbps for r in rs), 1)])
        end
    end
    table!("voice", ["Scenario", "Policy", "Loss (median)", "Loss (worst)", "p95 delay (ms)", "Codec (kbps)"], rows; align="llrrrr")
    number!("maxContenders", contenders)
    MAX_CONTENDERS[] = contenders
end

function voice_aware()
    rows = []
    for sc in SCENARIOS, p in POLICIES
        before, after, better, worse, cost = Float64[], Float64[], 0, 0, Float64[]
        for seed in 1:20
            W, cfg = scenario_world(sc, seed), VoiceSimConfig(scenario_voice(sc); policy=p)
            plain = optimize_assignment(W; seed=seed)[1]
            aware = optimize_assignment(W; seed=seed, voice=cfg)[1]
            r0, r1 = simulate_voice(W, plain; cfg=cfg), simulate_voice(W, aware; cfg=cfg)
            push!(before, r0.loss_fraction); push!(after, r1.loss_fraction)
            better += r1.delivered_mbps > r0.delivered_mbps * (1 + 1e-9)
            worse += r1.delivered_mbps < r0.delivered_mbps * (1 - 1e-9)
            push!(cost, (total_score(W, plain) - total_score(W, aware)) / length(W.client_ap))
        end
        push!(rows, [p == :fifo ? sc.name : "", "\\texttt{:$p}", pct(median(before), 2), pct(median(after), 2),
                     "$better / $worse", fmt(median(cost), 3), fmt(maximum(cost), 3)])
    end
    table!("aware", ["Scenario", "Policy", "Loss before", "Loss after", "Goodput up/down", "Cost (median)", "Cost (max)"], rows;
           align="llrrrrr")
end

function hypothesis_numbers()
    h = hypotheses()
    for (name, values, scale) in (("One", h.scoring_gain, 100), ("OneB", h.delay_cut, 1), ("Two", h.combined_gain, 100))
        wins, ties, losses, p = sign_test(values)
        number!("h$(name)Worlds", length(values))
        number!("h$(name)Wins", wins)
        number!("h$(name)Ties", ties)
        number!("h$(name)Losses", losses)
        number!("h$(name)Median", fmt(scale * median(values)))
        number!("h$(name)P", @sprintf("%.1e", p))
    end
    number!("hThreeKept", count(h.full_rate))
    number!("hThreeTrials", length(h.full_rate))
    number!("hTwoCodecCut", fmt(median(h.bitrate_cost), 1))
end

function run_results()
    mkpath(OUT)
    for step in (assignment_quality, speed, voice_outcomes, mac_validation, estimate_accuracy, isolated_queue,
                 voice_objective_exactness, voice_aware, hypothesis_numbers)
        t = @elapsed step()
        @printf("%-26s %7.1f s\n", step, t)
    end
    write(joinpath(OUT, "numbers.tex"), join(NUMBERS, "\n") * "\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_results()
end
