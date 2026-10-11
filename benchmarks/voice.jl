using WiFiAssignment, Statistics, Printf

const POLICIES = (:fifo, :deadline, :adaptive, :combined)

const SCENARIOS = (
    (name="Branch office", aps=6, stations=12, area=70.0, calls=8, load=0.35),
    (name="Open-plan office", aps=10, stations=24, area=100.0, calls=24, load=0.65),
    (name="High-density call centre", aps=12, stations=32, area=80.0, calls=64, load=0.75),
    (name="Emergency overload stress", aps=14, stations=24, area=100.0, calls=256, load=0.85),
)

function scenario_results(scenario; seeds=1:20)
    losses = Matrix{Float64}(undef, length(seeds), length(POLICIES))
    delays = similar(losses)
    codecs = similar(losses)
    runtimes = Vector{Float64}(undef, length(seeds))
    for (row, seed) in enumerate(seeds)
        W = make_world(seed=seed, cfg=SimConfig(k_channels=3, area=scenario.area),
                       n=scenario.aps, cpa=scenario.stations)
        assignment = greedy_rate(W)
        cfg = VoiceSimConfig(duration_s=10.0, calls_per_ap=scenario.calls,
                             background_load=scenario.load)
        timed = @timed compare_voice_strategies(W, assignment; cfg=cfg)
        runtimes[row] = timed.time
        for (col, result) in enumerate(timed.value)
            losses[row, col] = result.loss_fraction
            delays[row, col] = result.p95_delay_ms
            codecs[row, col] = result.mean_codec_kbps
        end
    end
    return losses, delays, codecs, runtimes
end

function scenarios()
    println("Synthetic voice scenarios; 20 independently seeded geometries per scenario")
    for scenario in SCENARIOS
        losses, delays, codecs, runtimes = scenario_results(scenario)
        @printf("\n%s: %d APs, %d stations/AP, %d calls/AP, %.0f%% background load\n",
                scenario.name, scenario.aps, scenario.stations, scenario.calls,
                100 * scenario.load)
        @printf("%-10s %11s %11s %11s %11s\n",
                "Policy", "Loss med", "Loss worst", "P95 ms", "Codec kbps")
        for (col, policy) in enumerate(POLICIES)
            @printf("%-10s %10.2f%% %10.2f%% %11.1f %11.1f\n",
                    String(policy), 100 * median(losses[:, col]),
                    100 * maximum(losses[:, col]), median(delays[:, col]),
                    median(codecs[:, col]))
        end
        @printf("Median runtime for all policies: %.2f ms\n", 1000 * median(runtimes))
    end
end

function sign_test(values; atol=1e-12)
    wins = count(>(atol), values)
    losses = count(<(-atol), values)
    n = wins + losses
    p = n == 0 ? 1.0 : Float64(sum(binomial(big(n), k) for k in wins:n; init=big(0))) / 2.0^n
    return wins, length(values) - n, losses, p
end

function hypothesis_report(name, values; scale=100, unit="percentage points")
    wins, ties, losses, p = sign_test(values)
    @printf("%s\n  worlds %d, wins %d, ties %d, losses %d\n",
            name, length(values), wins, ties, losses)
    @printf("  median improvement %.2f %s, minimum %.2f, p %.3g\n",
            scale * median(values), unit, scale * minimum(values), p)
end

function hypotheses(; seeds=1:50)
    scoring_gain = Float64[]
    delay_cut = Float64[]
    combined_gain = Float64[]
    bitrate_cost = Float64[]
    full_rate = Bool[]
    for seed in seeds
        W = make_world(n=10, cpa=4, seed=seed)
        assignment = greedy_rate(W)
        seed_scoring = Float64[]
        seed_delay = Float64[]
        seed_combined = Float64[]
        seed_bitrate = Float64[]
        for load in (0.70, 0.85, 0.95, 1.05)
            cfg = VoiceSimConfig(duration_s=5.0, calls_per_ap=32, background_load=load)
            deadline = simulate_voice(W, assignment; policy=:deadline, cfg=cfg)
            adaptive = simulate_voice(W, assignment; policy=:adaptive, cfg=cfg)
            combined = simulate_voice(W, assignment; policy=:combined, cfg=cfg)
            push!(seed_scoring, adaptive.loss_fraction - deadline.loss_fraction)
            push!(seed_delay, adaptive.p95_delay_ms - deadline.p95_delay_ms)
            push!(full_rate, combined.mean_codec_kbps >= 0.99 * cfg.codec_kbps)
        end
        for calls in 128:128:1024
            cfg = VoiceSimConfig(duration_s=5.0, calls_per_ap=calls, background_load=0.8)
            deadline = simulate_voice(W, assignment; policy=:deadline, cfg=cfg)
            deadline.deadline_loss_fraction <= 0.01 && continue
            combined = simulate_voice(W, assignment; policy=:combined, cfg=cfg)
            push!(seed_combined, deadline.loss_fraction - combined.loss_fraction)
            push!(seed_bitrate, deadline.mean_codec_kbps - combined.mean_codec_kbps)
        end
        push!(scoring_gain, median(seed_scoring))
        push!(delay_cut, median(seed_delay))
        isempty(seed_combined) || push!(combined_gain, median(seed_combined))
        isempty(seed_bitrate) || push!(bitrate_cost, median(seed_bitrate))
    end
    hypothesis_report("H1: deadline scoring beats adaptation alone under background load",
                      scoring_gain)
    hypothesis_report("H1b: deadline scoring cuts p95 delay versus adaptation alone",
                      delay_cut; scale=1, unit="ms")
    hypothesis_report("H2: combined control beats fixed-rate scoring under voice overload",
                      combined_gain)
    @printf("  overloaded worlds %d, median codec reduction %.1f kbps\n",
            length(combined_gain), median(bitrate_cost))
    @printf("H3: combined control retained >=99%% codec rate in %d/%d normal trials\n",
            count(full_rate), length(full_rate))
    return (; scoring_gain, delay_cut, combined_gain, bitrate_cost, full_rate)
end

function main()
    scenarios()
    println()
    hypotheses()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
