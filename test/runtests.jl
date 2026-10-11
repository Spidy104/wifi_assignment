using Test, WiFiAssignment

@testset "subset scores and exact search" begin
    for n in 1:8, seed in 1:3
        W = make_world(n=n, cpa=2, seed=seed)
        fast, exact_score = brute_force_fast(W)
        naive, naive_score = brute_force_naive(W)
        @test isapprox(exact_score, naive_score; atol=1e-10)
        @test isapprox(total_score(W, fast), exact_score; atol=1e-10)
        @test isapprox(total_score(W, naive), naive_score; atol=1e-10)
        @test all(1 .<= fast .<= W.cfg.k_channels)
        @test total_score(W, greedy_rate(W)) <= exact_score + 1e-10
        @test total_score(W, naive_coloring(W)) <= exact_score + 1e-10
    end
end

@testset "annealing and validation" begin
    W = make_world(n=8, cpa=2, seed=11)
    best, score, history = simulated_annealing(W; seed=11, iters=300)
    @test length(history) == 300
    @test isapprox(total_score(W, best), score; atol=1e-10)
    @test score >= maximum(history) - 1e-8
    @test simulated_annealing(W; seed=11, iters=300)[1] == best
    @test_throws ArgumentError total_score(W, zeros(Int, W.n))
    @test_throws ArgumentError client_rates(W, [1])
    @test_throws ArgumentError simulated_annealing(W; cooling=0)
    @test_throws ArgumentError make_world(n=0)

    one_channel = make_world(cfg=SimConfig(k_channels=1), n=4, cpa=2, seed=2)
    single, single_score, single_history = simulated_annealing(one_channel; iters=8)
    @test single == ones(Int, 4)
    @test isapprox(total_score(one_channel, single), single_score; atol=1e-10)
    @test all(==(single_score), single_history)
end

@testset "local search and optimizer" begin
    for n in 2:8, seed in 1:3
        W = make_world(n=n, cpa=2, seed=seed)
        exact_score = brute_force_fast(W)[2]
        @test total_score(W, greedy_rate(W)) >= total_score(W, greedy_rate(W; refine=false)) - 1e-10

        best, score = optimize_assignment(W; seed=seed)
        @test isapprox(score, exact_score; atol=1e-10)
        @test isapprox(total_score(W, best), score; atol=1e-10)
        @test optimize_assignment(W; seed=seed)[1] == best

        polished = simulated_annealing(W; seed=seed, iters=200)[2]
        unpolished = simulated_annealing(W; seed=seed, iters=200, polish=false)[2]
        @test polished >= unpolished - 1e-10
    end
    W = make_world(n=6, cpa=2, seed=4, cfg=SimConfig(k_channels=5))
    for k in 1:5
        best = maximum(total_score(W, collect(a)) for a in Iterators.product(ntuple(_ -> 1:k, 6)...))
        exact, exact_score = brute_force_fast(W; k=k)
        found, found_score = optimize_assignment(W; k=k, seed=1)
        @test exact_score ≈ best && total_score(W, exact) ≈ best
        @test found_score ≈ best && total_score(W, found) ≈ best
        @test all(<=(k), exact) && all(<=(k), found) && all(<=(k), greedy_rate(W; k=k)) &&
              all(<=(k), simulated_annealing(W; k=k, iters=100)[1])
    end
    W = make_world(n=1, cpa=2, seed=1)
    @test optimize_assignment(W)[2] ≈ brute_force_fast(W)[2]
    @test_throws ArgumentError optimize_assignment(W; restarts=0)
    @test_throws ArgumentError optimize_assignment(W; k=0)
end
@testset "airtime, carrier sense, and hidden transmitters" begin
    common = (k_channels=2, area=1.0, shadow_sigma=0.0, noise_dbm=-95.0,
              bw_mhz=20.0, p0=-30.0, alpha=35.0, demand_mbps=200.0,
              mac_efficiency=1.0, max_phy_mbps=100.0, min_rate=0.01)
    sensed = make_world(n=2, cpa=1, seed=19, cfg=SimConfig(; common..., cca_dbm=-100.0))
    hidden = make_world(n=2, cpa=1, seed=19, cfg=SimConfig(; common..., cca_dbm=100.0))
    same = [1, 1]
    separate = [1, 2]

    @test sensed.ap_activity == [1.0, 1.0]
    @test client_rate(sensed, 1, separate) ≈ 100.0
    @test client_rate(sensed, 1, same) ≈ 50.0
    @test client_rate(hidden, 1, same) < client_rate(sensed, 1, same)
    @test client_rate(hidden, 1, separate) ≈ client_rate(sensed, 1, separate)
    @test total_score(sensed, separate) > total_score(sensed, same)
    @test total_score(sensed, same) > total_score(hidden, same)

    shared = make_world(n=2, cpa=4, seed=19,
                        cfg=SimConfig(; common..., cca_dbm=-100.0, demand_mbps=20.0))
    @test shared.ap_activity ≈ [0.8, 0.8]
    @test client_rate(shared, 1, same) ≈ 100.0 / (4 * 1.8)
    @test isapprox(WiFiAssignment.subset_scores(shared)[4], total_score(shared, same); atol=1e-10)

    contended = make_world(n=2, cpa=8, seed=19,
                           cfg=SimConfig(; common..., cca_dbm=100.0, demand_mbps=5.0))
    act = WiFiAssignment.activity(contended, same)
    @test all(act .> contended.ap_activity)
    @test WiFiAssignment.activity(contended, separate) == contended.ap_activity
    @test client_rate(contended, 1, same) <
          WiFiAssignment.rate_unchecked(contended, 1, same, contended.ap_activity)
    @test isapprox(WiFiAssignment.subset_scores(contended)[4], total_score(contended, same); atol=1e-10)

    crowded = make_world(n=2, cpa=8, seed=19, cfg=SimConfig(; common..., cca_dbm=-100.0))
    @test client_rate(crowded, 1, separate) ≈ 12.5
    @test all(crowded.ap_activity .== 1.0)

    light = make_world(n=2, cpa=1, seed=19,
                       cfg=SimConfig(; common..., cca_dbm=-100.0, demand_mbps=2.0))
    @test client_rate(light, 1, separate) ≈ 2.0
    @test light.ap_activity[1] < 0.1

    floor = make_world(n=2, cpa=1, seed=19,
                       cfg=SimConfig(; common..., cca_dbm=-100.0,
                                     demand_mbps=1.0, min_rate=200.0))
    @test client_rate(floor, 1, separate) == 1.0
end

@testset "two-way voice policies" begin
    W = make_world(n=8, cpa=4, seed=7)
    assignment = greedy_rate(W)
    light = compare_voice_strategies(
        W, assignment; cfg=VoiceSimConfig(duration_s=2.0, background_load=0.2))
    congested = compare_voice_strategies(
        W, assignment; cfg=VoiceSimConfig(duration_s=2.0, background_load=0.95))
    voice_heavy = compare_voice_strategies(
        W, assignment; cfg=VoiceSimConfig(duration_s=2.0, background_load=0.8,
                                           calls_per_ap=512))

    @test all(r -> r.packets == 2 * 32 * 8 * 100, light)
    @test all(r -> 0 <= r.loss_fraction <= 1, congested)
    @test all(r -> isapprox(r.delivered_packets +
                            r.packets * (r.deadline_loss_fraction + r.retry_loss_fraction),
                            r.packets; atol=1e-7), congested)
    @test light[2].p95_delay_ms <= light[1].p95_delay_ms
    @test congested[2].p95_delay_ms < congested[1].p95_delay_ms
    @test voice_heavy[2].deadline_loss_fraction < voice_heavy[1].deadline_loss_fraction
    @test voice_heavy[4].loss_fraction < voice_heavy[2].loss_fraction
    @test voice_heavy[4].mean_codec_kbps < voice_heavy[2].mean_codec_kbps
    @test voice_heavy[3].mean_codec_kbps < voice_heavy[4].mean_codec_kbps
    @test all(r -> r.p95_delay_ms <= 150.0, (light..., congested..., voice_heavy...))

    calls_256(seconds, load, policy) = simulate_voice(
        W, assignment; policy=policy,
        cfg=VoiceSimConfig(duration_s=seconds, background_load=load, calls_per_ap=256))
    @test light[3].mean_codec_kbps >= 0.99 * 64.0
    @test calls_256(2.0, 0.95, :adaptive).mean_codec_kbps <
          calls_256(2.0, 0.2, :adaptive).mean_codec_kbps
    @test abs(calls_256(2.0, 0.8, :fifo).loss_fraction -
              calls_256(8.0, 0.8, :fifo).loss_fraction) < 0.05

    isolated = make_world(n=2, cpa=1, seed=3, cfg=SimConfig(k_channels=2))
    no_collision = simulate_voice(
        isolated, [1, 2]; policy=:deadline,
        cfg=VoiceSimConfig(duration_s=0.02, calls_per_ap=1, background_load=0.0))
    @test no_collision.retry_loss_fraction == 0.0
    @test isapprox(no_collision.delivered_packets, no_collision.packets; atol=1e-10)

    @test WiFiAssignment.dcf_collision(1.0) == 0.0
    @test isapprox(WiFiAssignment.dcf_collision(10.0), 0.3844; atol=1e-4)
    @test issorted(WiFiAssignment.dcf_collision.(1.0:0.5:20.0))
    @test WiFiAssignment.hidden_collision(0.0) == 0.0
    @test WiFiAssignment.hidden_collision(1.0) == 1.0
    @test all(u -> WiFiAssignment.hidden_collision(u) >= u, 0.0:0.05:1.0)
    for (area, collides) in ((1.0, true), (10_000.0, false))
        pair = make_world(n=2, cpa=4, seed=3,
                          cfg=SimConfig(k_channels=1, area=area, cca_dbm=100.0, shadow_sigma=0.0))
        result = simulate_voice(pair, [1, 1]; policy=:deadline,
                                cfg=VoiceSimConfig(duration_s=0.2, calls_per_ap=1, background_load=0.5))
        @test (result.retry_loss_fraction > 0) == collides
    end
    @test_throws ArgumentError simulate_voice(W, assignment; policy=:unknown)
    @test_throws ArgumentError simulate_voice(
        W, assignment; cfg=VoiceSimConfig(min_codec_kbps=65.0))
end

@testset "voice-aware assignment" begin
    cfg = VoiceSimConfig(calls_per_ap=128, background_load=0.8)
    for seed in 1:3, policy in (:fifo, :deadline, :adaptive, :combined)
        W = make_world(n=6, cpa=4, seed=seed, cfg=SimConfig(area=60.0))
        voice(a) = WiFiAssignment.voice_utility(W, a, WiFiAssignment.activity(W, a), cfg, policy)
        table = WiFiAssignment.voice_subset_scores(W, cfg, policy)
        assignments = [digits(code; base=3, pad=6) .+ 1 for code in 0:(3^6 - 1)]
        @test all(assignments) do a
            masks = WiFiAssignment.channel_masks(a, 3)
            isapprox(sum(table[m + 1] for m in masks), voice(a); atol=1e-9)
        end
        best = maximum(a -> total_score(W, a) + voice(a), assignments)
        found, score = optimize_assignment(W; seed=seed, voice=cfg, policy=policy)
        @test isapprox(score, best; atol=1e-9)
        @test isapprox(total_score(W, found) + voice(found), best; atol=1e-9)
    end
    W = make_world(n=4, cpa=2, seed=1)
    @test_throws ArgumentError optimize_assignment(W; voice=cfg, policy=:unknown)
    @test_throws ArgumentError optimize_assignment(W; voice=VoiceSimConfig(min_codec_kbps=65.0))
end
