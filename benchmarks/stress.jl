using WiFiAssignment, Test, Random, Statistics, Printf

function reference_phy(W::World, c::Int, assign, act)
    own = W.client_ap[c]
    interference = 10.0^(W.cfg.noise_dbm / 10.0)
    for ap in 1:W.n
        if ap != own && assign[ap] == assign[own] && W.AP_AP[own, ap] < W.cfg.cca_dbm
            interference += act[ap] * 10.0^(W.C_AP[c, ap] / 10.0)
        end
    end
    return min(W.cfg.max_phy_mbps, W.cfg.bw_mhz * log2(1 + 10.0^(W.C_AP[c, own] / 10.0) / interference))
end

# Jacobi iteration of the airtime fixed point; channel 0 marks an absent AP.
function reference_activity(W::World, assign)
    need(ap, channels, act) = min(1.0, sum(W.cfg.demand_mbps /
                                           (W.cfg.mac_efficiency * max(reference_phy(W, c, channels, act), W.cfg.min_rate))
                                           for c in eachindex(W.client_ap) if W.client_ap[c] == ap))
    act = [need(ap, collect(1:W.n), zeros(W.n)) for ap in 1:W.n]
    for _ in 1:100_000
        next = [assign[ap] == 0 ? act[ap] : need(ap, assign, act) for ap in 1:W.n]
        maximum(abs.(next .- act)) < 1e-14 && return next
        act = next
    end
    return act
end

function reference_rate(W::World, c::Int, assign, act=reference_activity(W, assign))
    own = W.client_ap[c]
    blockers = sum((act[ap] for ap in 1:W.n
                    if ap != own && assign[ap] == assign[own] && W.AP_AP[own, ap] >= W.cfg.cca_dbm); init=0.0)
    capacity = W.cfg.mac_efficiency * reference_phy(W, c, assign, act) / (W.clients_per_ap * (1 + blockers))
    return min(W.cfg.demand_mbps, max(capacity, W.cfg.min_rate))
end

function reference_score(W::World, assign)
    act = reference_activity(W, assign)
    return sum(log(reference_rate(W, c, assign, act)) for c in eachindex(W.client_ap))
end

function reference_subset_score(W::World, mask::Int)
    assign = [(mask >> (ap - 1)) & 1 for ap in 1:W.n]
    act = reference_activity(W, assign)
    return sum((log(reference_rate(W, c, assign, act)) for c in eachindex(W.client_ap)
                if assign[W.client_ap[c]] == 1); init=0.0)
end

function reference_exhaustive(W::World)
    k = W.cfg.k_channels
    assign = Vector{Int}(undef, W.n)
    best_score = -Inf
    best = similar(assign)
    for code in 0:(k^W.n - 1)
        value = code
        for ap in 1:W.n
            assign[ap] = value % k + 1
            value ÷= k
        end
        score = reference_score(W, assign)
        if score > best_score
            best_score = score
            copyto!(best, assign)
        end
    end
    return best, best_score
end

function check_assignment(W::World, assign)
    @test length(assign) == W.n
    @test all(ch -> 1 <= ch <= W.cfg.k_channels, assign)
    rates = client_rates(W, assign)
    @test length(rates) == length(W.client_ap)
    @test all(isfinite, rates)
    @test all(rate -> min(W.cfg.min_rate, W.cfg.demand_mbps) <= rate <= W.cfg.demand_mbps, rates)
    for c in eachindex(rates)
        @test isapprox(rates[c], reference_rate(W, c, assign); atol=1e-9, rtol=1e-11)
        @test isapprox(rates[c], client_rate(W, c, assign); atol=1e-12)
    end
    @test isapprox(total_score(W, assign), reference_score(W, assign); atol=1e-8, rtol=1e-11)
    return total_score(W, assign)
end

function run_checks()
    small_worlds = 0
    subset_worlds = 0
    larger_worlds = 0

    @testset "independent exhaustive oracle" begin
        for n in 1:6, k in 1:5, seed in 1:3
            cfg = SimConfig(k_channels=k, area=25.0 + 35seed,
                            noise_dbm=-120.0 + 12seed,
                            bw_mhz=5.0 + 10seed,
                            shadow_sigma=seed == 1 ? 0.0 : 3.5seed,
                            min_rate=seed == 3 ? 3.0 : 0.1)
            W = make_world(n=n, cpa=1 + (n + seed) % 3, seed=seed, cfg=cfg)
            reference_assign, reference_best = reference_exhaustive(W)
            naive_assign, naive_best = brute_force_naive(W)
            @test isapprox(reference_best, naive_best; atol=1e-8, rtol=1e-11)
            @test isapprox(reference_best, check_assignment(W, naive_assign); atol=1e-8, rtol=1e-11)
            @test isapprox(reference_best, check_assignment(W, reference_assign); atol=1e-8, rtol=1e-11)
            fast_assign, fast_best = brute_force_fast(W)
            @test isapprox(reference_best, fast_best; atol=1e-8, rtol=1e-11)
            @test isapprox(reference_best, check_assignment(W, fast_assign); atol=1e-8, rtol=1e-11)
            small_worlds += 1
        end
    end

    @testset "all subset masks and channel relabelings" begin
        for n in 1:10, seed in 1:4
            W = make_world(n=n, cpa=1 + seed % 3, seed=100 + seed)
            table = WiFiAssignment.subset_scores(W)
            @test length(table) == 1 << n
            for mask in 0:((1 << n) - 1)
                @test isapprox(table[mask + 1], reference_subset_score(W, mask);
                               atol=1e-8, rtol=1e-11)
            end
            assign = rand(Xoshiro(seed), 1:W.cfg.k_channels, n)
            renamed = randperm(Xoshiro(seed + 100), W.cfg.k_channels)[assign]
            @test isapprox(total_score(W, assign), total_score(W, renamed); atol=1e-10)
            @test all(isapprox.(client_rates(W, assign), client_rates(W, renamed); atol=1e-10))
            subset_worlds += 1
        end
    end

    @testset "long annealing and varied channel conditions" begin
        for n in (1, 2, 4, 8, 12, 14, 16, 18), k in (1, 2, 3, 5), seed in 1:3
            cfg = SimConfig(k_channels=k, area=seed == 1 ? 2.0 : 100.0 + 40seed,
                            noise_dbm=(-125.0, -90.0, -70.0)[seed],
                            bw_mhz=(1.0, 20.0, 40.0)[seed],
                            shadow_sigma=(0.0, 4.0, 12.0)[seed],
                            min_rate=seed == 1 ? 200.0 : 0.01)
            W = make_world(n=n, cpa=n >= 16 ? 1 : 1 + seed % 3, cfg=cfg, seed=500 + seed)
            coloring = naive_coloring(W; thresh=(-110.0, -80.0, -40.0)[seed])
            greedy = greedy_rate(W)
            annealed, annealed_score, history = simulated_annealing(
                W; seed=1000 + seed, iters=12_000, T0=5.0, cooling=0.9995)
            check_assignment(W, coloring)
            check_assignment(W, greedy)
            @test isapprox(check_assignment(W, annealed), annealed_score; atol=1e-8)
            @test length(history) == 12_000
            @test annealed_score >= maximum(history) - 1e-8
            exact_assign, exact_score = brute_force_fast(W)
            @test isapprox(check_assignment(W, exact_assign), exact_score; atol=1e-8)
            @test exact_score >= annealed_score - 1e-8
            @test exact_score >= total_score(W, greedy) - 1e-8
            @test isapprox(optimize_assignment(W; seed=seed)[2], exact_score; atol=1e-8)
            larger_worlds += 1
        end
    end

    @testset "identities, determinism, and invalid inputs" begin
        W = make_world(n=7, cpa=2, seed=900)
        same = make_world(n=7, cpa=2, seed=900)
        @test isequal(W.AP_AP, same.AP_AP)
        @test W.C_AP == same.C_AP
        @test W.client_pos == same.client_pos
        @test all(isnan(W.AP_AP[i, i]) for i in 1:W.n)
        @test all(W.AP_AP[i, j] == W.AP_AP[j, i] for i in 1:W.n for j in 1:W.n if i != j)

        shifted = make_world(n=7, cpa=2, seed=900,
                             cfg=SimConfig(p0=-20.0, noise_dbm=-85.0, cca_dbm=-72.0))
        assign = [1, 2, 3, 2, 1, 3, 1]
        @test all(isapprox.(client_rates(W, assign), client_rates(shifted, assign);
                            atol=1e-9, rtol=1e-11))
        @test isapprox(total_score(W, assign), total_score(shifted, assign); atol=1e-9)
        @test simulated_annealing(W; seed=88, iters=20_000)[1] ==
              simulated_annealing(W; seed=88, iters=20_000)[1]
        @test_throws ArgumentError total_score(W, [1])
        @test_throws ArgumentError total_score(W, zeros(Int, W.n))
        @test_throws ArgumentError total_score(W, fill(4, W.n))
        @test_throws BoundsError client_rate(W, 0, assign)
        @test_throws BoundsError client_rate(W, length(W.client_ap) + 1, assign)
        @test_throws ArgumentError greedy_rate(W; k=0)
        @test_throws ArgumentError simulated_annealing(W; k=4)
        @test_throws ArgumentError simulated_annealing(W; iters=-1)
        @test_throws ArgumentError simulated_annealing(W; T0=0)
        @test_throws ArgumentError simulated_annealing(W; cooling=1.1)
        @test_throws ArgumentError brute_force_naive(make_world(n=13, cpa=1, seed=1))
        @test_throws ArgumentError make_world(n=19)
        @test_throws ArgumentError make_world(cpa=0)
        @test_throws ArgumentError make_world(cfg=SimConfig(k_channels=0))
        @test_throws ArgumentError make_world(cfg=SimConfig(shadow_sigma=-1.0))
        @test_throws ArgumentError make_world(cfg=SimConfig(min_rate=0.0))
        @test_throws ArgumentError make_world(cfg=SimConfig(bw_mhz=0.0))
        @test_throws ArgumentError make_world(cfg=SimConfig(area=0.0))
        @test_throws ArgumentError make_world(cfg=SimConfig(noise_dbm=NaN))
        @test_throws ArgumentError make_world(cfg=SimConfig(demand_mbps=0.0))
        @test_throws ArgumentError make_world(cfg=SimConfig(mac_efficiency=1.1))
        @test_throws ArgumentError make_world(cfg=SimConfig(max_phy_mbps=0.0))
        @test_throws ArgumentError make_world(cfg=SimConfig(cca_dbm=NaN))
    end

    @testset "voice queues against the analytic steady state of an isolated AP" begin
        for cpa in (2, 8), load in (0.3, 0.9, 1.4), calls in (16, 256)
            W = make_world(n=1, cpa=cpa, seed=cpa, cfg=SimConfig(k_channels=1))
            cfg = VoiceSimConfig(duration_s=60.0, calls_per_ap=calls, background_load=load)
            raw = sum(client_rates(W, [1])) * 1000 * cfg.frame_ms
            overhead = 8 * cfg.packet_overhead_bytes
            offered(kbps) = 2calls * (kbps * cfg.frame_ms + overhead)
            fit(airtime) = clamp((max(airtime, 0) / 2calls - overhead) / cfg.frame_ms,
                                 cfg.min_codec_kbps, cfg.codec_kbps)
            shortfall(bits) = max(0.0, 1 - raw / bits)
            codec = Dict(:fifo => cfg.codec_kbps, :deadline => cfg.codec_kbps,
                         :adaptive => fit(raw - load * raw), :combined => fit(raw))
            background = Dict(:fifo => load * raw, :deadline => 0.0, :adaptive => load * raw, :combined => 0.0)
            for r in compare_voice_strategies(W, [1]; cfg=cfg)
                @test isapprox(r.mean_codec_kbps, codec[r.policy]; atol=0.5)
                @test isapprox(r.loss_fraction, shortfall(offered(codec[r.policy]) + background[r.policy]); atol=0.005)
                @test r.retry_loss_fraction == 0.0
            end
        end
    end

    @testset "MAC collision model against slot-level simulation" begin
        rng = Xoshiro(11)
        for n in 2:4
            stage = zeros(Int, n)
            counter = [rand(rng, 0:(WiFiAssignment.CW_MIN - 1)) for _ in 1:n]
            attempts = collisions = 0
            for _ in 1:1_000_000
                ready = count(==(0), counter)
                if ready == 0
                    counter .-= 1
                    continue
                end
                attempts += ready
                ready > 1 && (collisions += ready)
                for i in 1:n
                    counter[i] == 0 || continue
                    stage[i] = ready > 1 ? min(stage[i] + 1, WiFiAssignment.BACKOFF_STAGES) : 0
                    counter[i] = rand(rng, 0:(WiFiAssignment.CW_MIN * 2^stage[i] - 1))
                end
            end
            @test isapprox(WiFiAssignment.dcf_collision(Float64(n)), collisions / attempts; atol=0.01)
        end
        for u in (0.1, 0.5, 0.9)
            overlaps = 0
            for _ in 1:100_000
                t, start, hit = -200.0, rand(rng), false
                while t < 2.0
                    hit |= t < start + 1.0 && t + 1.0 > start
                    t += 1.0 + randexp(rng) * (1 - u) / u
                end
                overlaps += hit
            end
            @test isapprox(WiFiAssignment.hidden_collision(u), overlaps / 100_000; atol=0.01)
        end
    end

    @testset "voice steady-state estimate against the simulator" begin
        for (n, cpa, area, calls, load) in ((12, 32, 80.0, 64, 0.75), (8, 4, 100.0, 256, 0.95),
                                            (14, 24, 100.0, 256, 0.85)),
            seed in 1:3, policy in (:fifo, :deadline, :adaptive, :combined)
            W = make_world(seed=seed, cfg=SimConfig(k_channels=3, area=area), n=n, cpa=cpa)
            assign = rand(Xoshiro(seed), 1:3, n)
            cfg = VoiceSimConfig(duration_s=20.0, calls_per_ap=calls, background_load=load)
            bitrate, delivered = WiFiAssignment.voice_steady_state(W, assign, WiFiAssignment.activity(W, assign), cfg, policy)
            r = simulate_voice(W, assign; policy=policy, cfg=cfg)
            @test isapprox(1 - mean(delivered), r.loss_fraction; atol=0.006)
            @test isapprox(mean(bitrate), r.mean_codec_kbps; atol=0.15)
        end
    end

    @testset "voice packet accounting on random worlds" begin
        rng = Xoshiro(7)
        for trial in 1:40
            n = rand(rng, 2:8)
            W = make_world(n=n, cpa=rand(rng, 1:6), seed=trial)
            cfg = VoiceSimConfig(duration_s=1.0, calls_per_ap=rand(rng, (4, 64, 512)),
                                 background_load=1.5rand(rng))
            for r in compare_voice_strategies(W, rand(rng, 1:3, n); cfg=cfg)
                @test isapprox(r.delivered_packets + r.packets * (r.deadline_loss_fraction + r.retry_loss_fraction),
                               r.packets; rtol=1e-9)
                @test 0 <= r.loss_fraction <= 1 && r.p95_delay_ms <= cfg.deadline_ms
            end
        end
    end

    println("Checked $small_worlds independently exhaustive worlds, $subset_worlds complete subset tables, and $larger_worlds long annealing worlds")
end

function measure(f, repeats::Int)
    f()
    f()
    GC.gc()
    times = Vector{Float64}(undef, repeats)
    bytes = Vector{Int}(undef, repeats)
    for i in 1:repeats
        result = @timed f()
        times[i] = result.time
        bytes[i] = result.bytes
    end
    return median(times) * 1000, minimum(bytes)
end

function hot_allocations(W::World, assign)
    act = WiFiAssignment.activity(W, assign)
    WiFiAssignment.rate_unchecked(W, 1, assign, act)
    WiFiAssignment.score_unchecked(W, assign)
    return @allocated(WiFiAssignment.rate_unchecked(W, 1, assign, act)),
           @allocated(WiFiAssignment.score_unchecked(W, assign))
end

function run_benchmarks()
    println("\nJulia ", VERSION, " / ", Sys.MACHINE)
    @printf("%-5s %-16s %12s %14s\n", "APs", "Operation", "Median ms", "Min alloc KiB")
    for n in (6, 8, 10, 12, 14, 16, 18)
        W = make_world(n=n, cpa=n >= 16 ? 2 : 4, seed=2026)
        greedy = greedy_rate(W)
        hot = hot_allocations(W, greedy)
        @test hot[1] == 0 && hot[2] < 1024
        cases = (("subset scores", () -> WiFiAssignment.subset_scores(W)),
                 ("greedy", () -> greedy_rate(W)),
                 ("anneal 30k", () -> simulated_annealing(W; seed=1, iters=30_000)))
        for (label, f) in cases
            milliseconds, bytes = measure(f, n >= 16 ? 3 : 5)
            @printf("%-5d %-16s %12.3f %14.1f\n", n, label, milliseconds, bytes / 1024)
        end
        milliseconds, bytes = measure(() -> brute_force_fast(W), n >= 14 ? 3 : 5)
        @printf("%-5d %-16s %12.3f %14.1f\n", n, "exact", milliseconds, bytes / 1024)
        if n <= 10
            milliseconds, bytes = measure(() -> brute_force_naive(W), 3)
            @printf("%-5d %-16s %12.3f %14.1f\n", n, "naive exact", milliseconds, bytes / 1024)
        end
        println()
    end

    W = make_world(n=10, cpa=4, seed=2026)
    simulated_annealing(W; seed=7, iters=0)
    simulated_annealing(W; seed=7, iters=100_000)
    baseline = @allocated simulated_annealing(W; seed=7, iters=0)
    extended = @allocated simulated_annealing(W; seed=7, iters=100_000)
    delta = extended - baseline
    @printf("SA extra allocation over 100,000 steps: %d bytes (history alone: 800,000 bytes)\n", delta)
end

function main()
    run_checks()
    run_benchmarks()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
