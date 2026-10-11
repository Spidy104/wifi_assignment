using WiFiAssignment, Test, Random, Statistics, Printf

function independent_rate(W::World, client::Int, assignment)
    own = W.client_ap[client]
    signal = 10.0^(W.C_AP[client, own] / 10.0)
    interference = 10.0^(W.cfg.noise_dbm / 10.0)
    blockers = 0.0
    for ap in 1:W.n
        if ap != own && assignment[ap] == assignment[own]
            if W.AP_AP[own, ap] >= W.cfg.cca_dbm
                blockers += W.ap_activity[ap]
            else
                interference += W.ap_activity[ap] * 10.0^(W.C_AP[client, ap] / 10.0)
            end
        end
    end
    phy = min(W.cfg.max_phy_mbps, W.cfg.bw_mhz * log2(1 + signal / interference))
    capacity = W.cfg.mac_efficiency * phy / (W.clients_per_ap * (1 + blockers))
    return min(W.cfg.demand_mbps, max(capacity, W.cfg.min_rate))
end

function independent_score(W::World, assignment)
    score = 0.0
    for client in eachindex(W.client_ap)
        score += log(independent_rate(W, client, assignment))
    end
    return score
end

function exact_partition_dp(table, channels)
    layer = copy(table)
    for _ in 2:channels
        next = similar(layer)
        @inbounds for mask in 0:(length(table) - 1)
            best = -Inf
            subset = mask
            while true
                best = max(best, table[subset + 1] + layer[(mask ⊻ subset) + 1])
                subset == 0 && break
                subset = (subset - 1) & mask
            end
            next[mask + 1] = best
        end
        layer = next
    end
    return layer[end]
end

function dense_world(n::Int, clients_per_ap::Int, channels::Int, seed::Int, area::Float64)
    cfg = SimConfig(k_channels=channels, area=area, noise_dbm=-95.0,
                    bw_mhz=20.0, min_rate=0.001, p0=-30.0,
                    alpha=35.0, shadow_sigma=8.0)
    W = make_world(n=n, cpa=clients_per_ap, cfg=cfg, seed=seed)
    all(>=(1.0), W.ap_activity) || error("the independent oracle assumes saturated APs")
    return W
end

function check_dense_assignment(W::World, assignment)
    @test length(assignment) == W.n
    @test all(ch -> 1 <= ch <= W.cfg.k_channels, assignment)
    rates = client_rates(W, assignment)
    @test length(rates) == length(W.client_ap)
    @test all(isfinite, rates)
    @test all(rate -> min(W.cfg.min_rate, W.cfg.demand_mbps) <= rate <= W.cfg.demand_mbps, rates)
    for client in eachindex(rates)
        @test isapprox(rates[client], independent_rate(W, client, assignment);
                       rtol=1e-11, atol=1e-9)
    end
    score = total_score(W, assignment)
    @test isapprox(score, independent_score(W, assignment); rtol=1e-11, atol=1e-8)
    return score
end

function check_dense_correctness()
    @testset "dense exact optimum by independent dynamic programming" begin
        for n in (12, 14), clients_per_ap in (8, 16), seed in (17, 42)
            W = dense_world(n, clients_per_ap, 3, seed, 80.0)
            table = WiFiAssignment.subset_scores(W)
            dp = exact_partition_dp(table, 3)
            assignment, partition = brute_force_fast(W)
            @test isapprox(dp, partition; atol=1e-8, rtol=1e-11)
            @test isapprox(dp, check_dense_assignment(W, assignment); atol=1e-8, rtol=1e-11)
            greedy = greedy_rate(W)
            annealed, sa_score, history = simulated_annealing(W; seed=seed, iters=250_000)
            @test check_dense_assignment(W, greedy) <= dp + 1e-8
            @test isapprox(check_dense_assignment(W, annealed), sa_score; atol=1e-8)
            @test sa_score <= dp + 1e-8
            @test sa_score >= maximum(history) - 1e-8
        end
    end

    @testset "optimize_assignment reaches the independent optimum" begin
        for n in (12, 14, 16), clients_per_ap in (4, 16), seed in 1:5
            W = dense_world(n, clients_per_ap, 3, seed, 100.0 * sqrt(n / 18))
            optimum = exact_partition_dp(WiFiAssignment.subset_scores(W), 3)
            best, score = optimize_assignment(W; seed=seed)
            @test isapprox(score, optimum; atol=1e-8)
            @test isapprox(check_dense_assignment(W, best), score; atol=1e-8)
        end
    end

    @testset "exact search and optimizer for 2, 4 and 5 channels" begin
        for channels in (2, 4, 5), n in (12, 14), clients_per_ap in (4, 16), seed in 1:3
            W = dense_world(n, clients_per_ap, channels, seed, 100.0 * sqrt(n / 18))
            optimum = exact_partition_dp(WiFiAssignment.subset_scores(W), channels)
            exact, exact_score = brute_force_fast(W)
            @test isapprox(exact_score, optimum; atol=1e-8)
            @test isapprox(check_dense_assignment(W, exact), optimum; atol=1e-8)
            best, score = optimize_assignment(W; seed=seed)
            @test isapprox(score, optimum; atol=1e-8)
            @test isapprox(check_dense_assignment(W, best), score; atol=1e-8)
        end
    end

    @testset "18 APs, heavy clients, and stronger transmitters" begin
        for (clients_per_ap, channels, seed, area) in ((16, 3, 1, 8.0), (24, 6, 2, 90.0),
                                                       (32, 12, 3, 120.0), (48, 18, 4, 160.0))
            W = dense_world(18, clients_per_ap, channels, seed, area)
            rng = Xoshiro(seed)
            for _ in 1:12
                check_dense_assignment(W, rand(rng, 1:channels, W.n))
            end
            same = ones(Int, W.n)
            distinct = collect(1:W.n)
            check_dense_assignment(W, same)
            exact, best_possible = brute_force_fast(W)
            @test isapprox(check_dense_assignment(W, exact), best_possible; atol=1e-8)
            @test isapprox(optimize_assignment(W; seed=seed)[2], best_possible; atol=1e-8)
            greedy = greedy_rate(W)
            greedy_score = check_dense_assignment(W, greedy)
            annealed, sa_score, history = simulated_annealing(
                W; seed=seed, iters=1_000_000, T0=20.0, cooling=0.999995)
            @test isapprox(sa_score, check_dense_assignment(W, annealed); atol=1e-8)
            @test sa_score >= maximum(history) - 1e-8
            @test length(history) == 1_000_000
            @test greedy_score <= best_possible + 1e-8
            @test sa_score <= best_possible + 1e-8

            stronger = make_world(n=18, cpa=clients_per_ap, seed=seed,
                                  cfg=SimConfig(k_channels=channels, area=area,
                                                noise_dbm=-95.0, bw_mhz=20.0,
                                                min_rate=0.001, p0=-20.0,
                                                alpha=35.0, shadow_sigma=8.0))
            @test all(stronger.C_AP[c, stronger.client_ap[c]] > W.C_AP[c, W.client_ap[c]]
                      for c in eachindex(W.client_ap))
            check_dense_assignment(stronger, greedy)
            if channels == W.n
                @test all(client_rates(W, distinct) .>= client_rates(W, same) .- 1e-8)
            end
            @printf("18 APs, %4d clients, %2d channels, area %4.0f: greedy %.2f, annealed %.2f, optimum %.2f\n",
                    length(W.client_ap), channels, area, greedy_score, sa_score, best_possible)
        end
    end

    @testset "independent 18 AP exact ceiling" begin
        W = dense_world(18, 16, 3, 5, 100.0)
        table = WiFiAssignment.subset_scores(W)
        timed = @timed exact_partition_dp(table, 3)
        optimum = timed.value
        plain = check_dense_assignment(W, greedy_rate(W; refine=false))
        refined = check_dense_assignment(W, greedy_rate(W))
        annealed, score, _ = simulated_annealing(W; seed=5, iters=1_000_000)
        best, best_score = optimize_assignment(W; seed=5)
        @test isapprox(score, check_dense_assignment(W, annealed); atol=1e-8)
        @test isapprox(best_score, check_dense_assignment(W, best); atol=1e-8)
        @test isapprox(best_score, optimum; atol=1e-8)
        @test plain <= refined <= optimum + 1e-8
        @test score <= optimum + 1e-8
        @test optimum - plain > 0.1
        @test optimum - score < 2.0
        @printf("Independent 18 AP DP optimum: %.3f; greedy gap: %.3f; SA gap: %.3f; DP: %.2f s\n",
                optimum, optimum - plain, optimum - score, timed.time)
    end
end

function sample(f, repetitions::Int)
    f()
    GC.gc()
    times = Vector{Float64}(undef, repetitions)
    bytes = Vector{Int}(undef, repetitions)
    for i in 1:repetitions
        result = @timed f()
        times[i] = result.time
        bytes[i] = result.bytes
    end
    return median(times), minimum(bytes)
end

function allocation_difference(W::World)
    simulated_annealing(W; seed=44, iters=0)
    simulated_annealing(W; seed=44, iters=1_000_000)
    baseline = @allocated simulated_annealing(W; seed=44, iters=0)
    long = @allocated simulated_annealing(W; seed=44, iters=1_000_000)
    return long - baseline
end

function benchmark_density()
    println("\nDense scaling on Julia ", VERSION, " / ", Sys.MACHINE)
    @printf("%4s %8s %4s %12s %12s %13s %11s\n",
            "APs", "Clients", "K", "Subset s", "Greedy s", "SA 1M s", "SA MiB")
    for (n, cpa, k, seed, area) in ((12, 8, 3, 21, 50.0),
                                    (14, 16, 3, 22, 80.0),
                                    (16, 24, 6, 23, 100.0),
                                    (18, 16, 3, 24, 100.0),
                                    (18, 32, 6, 25, 120.0),
                                    (18, 48, 12, 26, 160.0),
                                    (18, 64, 18, 27, 160.0))
        W = dense_world(n, cpa, k, seed, area)
        count = n * cpa
        repeats = count > 500 ? 2 : 3
        subset_time, _ = sample(() -> WiFiAssignment.subset_scores(W), repeats)
        greedy_time, _ = sample(() -> greedy_rate(W), repeats)
        anneal_time, anneal_bytes = sample(
            () -> simulated_annealing(W; seed=42, iters=1_000_000), repeats)
        @printf("%4d %8d %4d %12.3f %12.3f %13.3f %11.2f\n",
                n, count, k, subset_time, greedy_time, anneal_time,
                anneal_bytes / 1024^2)
    end

    W = dense_world(18, 32, 6, 73, 120.0)
    extra = allocation_difference(W)
    @printf("Extra allocation for 1,000,000 SA steps: %d bytes; history: 8,000,000 bytes\n", extra)
end

function main()
    check_dense_correctness()
    benchmark_density()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
