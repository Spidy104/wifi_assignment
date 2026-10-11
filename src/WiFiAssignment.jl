module WiFiAssignment

using Random
using Base: @kwdef

export SimConfig, World, make_world, client_rate, client_rates, total_score,
       naive_coloring, greedy_rate, AnnealConfig, simulated_annealing, optimize_assignment,
       brute_force_fast, brute_force_naive, VoiceSimConfig, VoiceResult, simulate_voice,
       compare_voice_strategies

@kwdef struct SimConfig
    area::Float64 = 100.0
    noise_dbm::Float64 = -95.0
    k_channels::Int = 3
    bw_mhz::Float64 = 20.0
    min_rate::Float64 = 0.5
    shadow_sigma::Float64 = 4.0
    p0::Float64 = -30.0
    alpha::Float64 = 35.0
    demand_mbps::Float64 = 40.0
    mac_efficiency::Float64 = 0.72
    max_phy_mbps::Float64 = 144.0
    cca_dbm::Float64 = -82.0
end

const DEFAULT_CONFIG = SimConfig()

struct World
    cfg::SimConfig
    n::Int
    ap_pos::Matrix{Float64}
    client_pos::Matrix{Float64}
    client_ap::Vector{Int}
    AP_AP::Matrix{Float64}
    C_AP::Matrix{Float64}
    C_AP_mW::Matrix{Float64}
    noise_mw::Float64
    rate_scale::Float64
    clients_per_ap::Int
    ap_activity::Vector{Float64}
end

const POLICIES = (:fifo, :deadline, :adaptive, :combined)

@kwdef struct VoiceSimConfig
    duration_s::Float64 = 20.0
    frame_ms::Float64 = 20.0
    deadline_ms::Float64 = 150.0
    calls_per_ap::Int = 32
    codec_kbps::Float64 = 64.0
    min_codec_kbps::Float64 = 12.0
    packet_overhead_bytes::Int = 60
    background_load::Float64 = 0.95
    retry_limit::Int = 2
    adaptation_speed::Float64 = 0.35
    policy::Symbol = :combined
end

function VoiceSimConfig(cfg::VoiceSimConfig; changes...)
    for name in keys(changes)
        hasfield(VoiceSimConfig, name) || throw(ArgumentError("VoiceSimConfig has no field $name"))
    end
    return VoiceSimConfig(; (f => getfield(cfg, f) for f in fieldnames(VoiceSimConfig))..., changes...)
end

@kwdef struct AnnealConfig
    iters::Int = 3000
    T0::Union{Nothing,Float64} = nothing
    cooling::Union{Nothing,Float64} = nothing
    polish::Bool = true
end

struct VoiceResult
    policy::Symbol
    packets::Int
    delivered_packets::Float64
    loss_fraction::Float64
    deadline_loss_fraction::Float64
    retry_loss_fraction::Float64
    mean_delay_ms::Float64
    p95_delay_ms::Float64
    mean_codec_kbps::Float64
    delivered_mbps::Float64
end

const CW_MIN = 16
const BACKOFF_STAGES = 6
const CAPTURE_DB = 10.0
const RESTARTS = 8
const KICKS = 200

function dcf_collision(n::Float64)
    n <= 1.0 && return 0.0
    lo, hi, p = 0.0, 1.0, 0.5
    for _ in 1:100
        sum_powers, slope_powers, power = 1.0, 0.0, 1.0
        for i in 1:(BACKOFF_STAGES - 1)
            slope_powers += 2i * power
            power *= 2p
            sum_powers += power
        end
        denominator = 1 + CW_MIN + p * CW_MIN * sum_powers
        tau = 2 / denominator
        dtau = -2 * CW_MIN * (sum_powers + p * slope_powers) / denominator^2
        idle = (1 - tau)^(n - 2)
        f = p - 1 + idle * (1 - tau)
        f > 0 ? (hi = p) : (lo = p)
        step = p - f / (1 - (n - 1) * idle * dtau)
        next = lo < step < hi ? step : (lo + hi) / 2
        abs(next - p) < 1e-15 && return next
        p = next
    end
    error("Bianchi collision fixed point did not converge for $n contenders")
end

hidden_collision(u::Float64) = u + (1 - u) * -expm1(-u / (1 - u))

struct QueuedWork
    remaining_bits::Float64
    packet_bits::Float64
    payload_bits::Float64
    arrival_ms::Float64
    deadline_ms::Float64
end

function Base.show(io::IO, W::World)
    print(io, "World($(W.n) APs, $(length(W.client_ap)) clients, K=$(W.cfg.k_channels))")
end

function validate_config(cfg::SimConfig, n::Int, cpa::Int)
    1 <= n <= 18 || throw(ArgumentError("the number of APs must be between 1 and 18"))
    cpa > 0 || throw(ArgumentError("clients per AP must be positive"))
    cfg.k_channels > 0 || throw(ArgumentError("the channel count must be positive"))
    isfinite(cfg.area) && cfg.area > 0 || throw(ArgumentError("area must be positive and finite"))
    isfinite(cfg.noise_dbm) || throw(ArgumentError("noise floor must be finite"))
    isfinite(cfg.bw_mhz) && cfg.bw_mhz > 0 || throw(ArgumentError("bandwidth must be positive and finite"))
    isfinite(cfg.min_rate) && cfg.min_rate > 0 || throw(ArgumentError("minimum rate must be positive and finite"))
    isfinite(cfg.shadow_sigma) && cfg.shadow_sigma >= 0 || throw(ArgumentError("shadow deviation must be nonnegative and finite"))
    isfinite(cfg.p0) || throw(ArgumentError("reference power must be finite"))
    isfinite(cfg.alpha) && cfg.alpha > 0 || throw(ArgumentError("path-loss coefficient must be positive and finite"))
    isfinite(cfg.demand_mbps) && cfg.demand_mbps > 0 || throw(ArgumentError("client demand must be positive and finite"))
    isfinite(cfg.mac_efficiency) && 0 < cfg.mac_efficiency <= 1 || throw(ArgumentError("MAC efficiency must be within (0, 1]"))
    isfinite(cfg.max_phy_mbps) && cfg.max_phy_mbps > 0 || throw(ArgumentError("PHY cap must be positive and finite"))
    isfinite(cfg.cca_dbm) || throw(ArgumentError("carrier-sense threshold must be finite"))
end

function make_world(; seed::Union{Nothing,Integer}=nothing, cfg::SimConfig=DEFAULT_CONFIG,
                    n::Integer=10, cpa::Integer=4)
    n_aps = Int(n)
    clients_per_ap = Int(cpa)
    validate_config(cfg, n_aps, clients_per_ap)
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    clients = n_aps * clients_per_ap
    ap_pos = rand(rng, n_aps, 2) .* cfg.area
    client_ap = repeat(1:n_aps; inner=clients_per_ap)
    client_pos = Matrix{Float64}(undef, clients, 2)

    for c in 1:clients
        own = client_ap[c]
        radius = 5.0 + 10.0 * rand(rng)
        angle = 2π * rand(rng)
        client_pos[c, 1] = clamp(ap_pos[own, 1] + radius * cos(angle), 0.0, cfg.area)
        client_pos[c, 2] = clamp(ap_pos[own, 2] + radius * sin(angle), 0.0, cfg.area)
    end

    ap_ap = fill(NaN, n_aps, n_aps)
    for i in 1:n_aps, j in (i + 1):n_aps
        d = hypot(ap_pos[i, 1] - ap_pos[j, 1], ap_pos[i, 2] - ap_pos[j, 2])
        value = cfg.p0 - cfg.alpha * log10(max(d, 1.0)) + cfg.shadow_sigma * randn(rng)
        ap_ap[i, j] = ap_ap[j, i] = value
    end

    client_ap_dbm = Matrix{Float64}(undef, clients, n_aps)
    client_ap_mw = similar(client_ap_dbm)
    for j in 1:n_aps, c in 1:clients
        d = hypot(client_pos[c, 1] - ap_pos[j, 1], client_pos[c, 2] - ap_pos[j, 2])
        dbm = cfg.p0 - cfg.alpha * log10(max(d, 1.0)) + cfg.shadow_sigma * randn(rng)
        client_ap_dbm[c, j] = dbm
        client_ap_mw[c, j] = 10.0^(dbm / 10.0)
    end

    noise = 10.0^(cfg.noise_dbm / 10.0)
    scale = cfg.bw_mhz / log(2.0)
    activity = zeros(n_aps)
    for c in 1:clients
        own = client_ap[c]
        phy = min(cfg.max_phy_mbps, scale * log1p(client_ap_mw[c, own] / noise))
        activity[own] += cfg.demand_mbps / (cfg.mac_efficiency * max(phy, cfg.min_rate))
    end
    for ap in eachindex(activity)
        activity[ap] = min(activity[ap], 1.0)
    end

    return World(cfg, n_aps, ap_pos, client_pos, client_ap, ap_ap,
                 client_ap_dbm, client_ap_mw, noise, scale, clients_per_ap, activity)
end

function validate_assignment(W::World, assign::AbstractVector{<:Integer})
    length(assign) == W.n || throw(ArgumentError("assignment must contain one channel per AP"))
    for ch in assign
        1 <= ch <= W.cfg.k_channels || throw(ArgumentError("assignment contains an invalid channel"))
    end
end

function validate_k(W::World, k::Integer)
    1 <= k <= W.cfg.k_channels || throw(ArgumentError("channel count must be within 1:$(W.cfg.k_channels)"))
end

function activity!(act::AbstractVector{Float64}, W::World, assign::AbstractVector{<:Integer})
    copyto!(act, W.ap_activity)
    for _ in 1:1000
        change = 0.0
        @inbounds for ap in 1:W.n
            (assign[ap] == 0 || act[ap] >= 1.0) && continue
            need = 0.0
            for c in ((ap - 1) * W.clients_per_ap + 1):(ap * W.clients_per_ap)
                interference = W.noise_mw
                for j in 1:W.n
                    if j != ap && assign[j] == assign[ap] && W.AP_AP[ap, j] < W.cfg.cca_dbm
                        interference += W.C_AP_mW[c, j] * act[j]
                    end
                end
                phy = min(W.cfg.max_phy_mbps, W.rate_scale * log1p(W.C_AP_mW[c, ap] / interference))
                need += W.cfg.demand_mbps / (W.cfg.mac_efficiency * max(phy, W.cfg.min_rate))
            end
            need = min(need, 1.0)
            change = max(change, need - act[ap])
            act[ap] = need
        end
        change <= 1e-12 && return act
    end
    error("airtime fixed point did not converge")
end

activity(W::World, assign::AbstractVector{<:Integer}) =
    all(>=(1.0), W.ap_activity) ? W.ap_activity : activity!(similar(W.ap_activity), W, assign)

@inline function rate_unchecked(W::World, c::Int, assign::AbstractVector{<:Integer},
                                act::AbstractVector{Float64})
    own = @inbounds W.client_ap[c]
    ch = @inbounds assign[own]
    signal = @inbounds W.C_AP_mW[c, own]
    interference = W.noise_mw
    blockers = 0.0
    @inbounds for j in 1:W.n
        if j != own && assign[j] == ch
            if W.AP_AP[own, j] >= W.cfg.cca_dbm
                blockers += act[j]
            else
                interference += W.C_AP_mW[c, j] * act[j]
            end
        end
    end
    phy = min(W.cfg.max_phy_mbps, W.rate_scale * log1p(signal / interference))
    capacity = W.cfg.mac_efficiency * phy / (W.clients_per_ap * (1.0 + blockers))
    return min(W.cfg.demand_mbps, max(capacity, W.cfg.min_rate))
end

function client_rate(W::World, c::Integer, assign::AbstractVector{<:Integer})
    1 <= c <= length(W.client_ap) || throw(BoundsError(W.client_ap, c))
    validate_assignment(W, assign)
    return rate_unchecked(W, Int(c), assign, activity(W, assign))
end

function client_rates(W::World, assign::AbstractVector{<:Integer})
    validate_assignment(W, assign)
    act = activity(W, assign)
    return [rate_unchecked(W, c, assign, act) for c in eachindex(W.client_ap)]
end

function validate_voice_config(cfg::VoiceSimConfig)
    isfinite(cfg.duration_s) && cfg.duration_s > 0 || throw(ArgumentError("voice duration must be positive and finite"))
    isfinite(cfg.frame_ms) && cfg.frame_ms > 0 || throw(ArgumentError("voice frame duration must be positive and finite"))
    isfinite(cfg.deadline_ms) && cfg.deadline_ms >= cfg.frame_ms || throw(ArgumentError("voice deadline must be at least one frame"))
    cfg.calls_per_ap > 0 || throw(ArgumentError("calls per AP must be positive"))
    isfinite(cfg.codec_kbps) && cfg.codec_kbps > 0 || throw(ArgumentError("codec rate must be positive and finite"))
    isfinite(cfg.min_codec_kbps) && 0 < cfg.min_codec_kbps <= cfg.codec_kbps || throw(ArgumentError("minimum codec rate must be within (0, codec rate]"))
    cfg.packet_overhead_bytes >= 0 || throw(ArgumentError("packet overhead must be nonnegative"))
    isfinite(cfg.background_load) && cfg.background_load >= 0 || throw(ArgumentError("background load must be nonnegative and finite"))
    cfg.retry_limit >= 0 || throw(ArgumentError("retry limit must be nonnegative"))
    isfinite(cfg.adaptation_speed) && 0 < cfg.adaptation_speed <= 1 || throw(ArgumentError("adaptation speed must be within (0, 1]"))
    cfg.policy in POLICIES || throw(ArgumentError("voice policy must be :fifo, :deadline, :adaptive, or :combined"))
end

prioritized(cfg::VoiceSimConfig) = cfg.policy in (:deadline, :combined)
adaptive(cfg::VoiceSimConfig) = cfg.policy in (:adaptive, :combined)
packet_bits(cfg::VoiceSimConfig, kbps::Float64) = kbps * cfg.frame_ms + 8.0 * cfg.packet_overhead_bytes

function voice_ap_capacity(W::World, assign::AbstractVector{<:Integer})
    capacity = zeros(W.n)
    act = activity(W, assign)
    @inbounds for c in eachindex(W.client_ap)
        capacity[W.client_ap[c]] += rate_unchecked(W, c, assign, act)
    end
    return capacity
end

function mac_tables(W::World)
    capture = 10.0^(CAPTURE_DB / 10.0)
    clients(i) = ((i - 1) * W.clients_per_ap + 1):(i * W.clients_per_ap)
    sensed = [i != j && W.AP_AP[i, j] >= W.cfg.cca_dbm for i in 1:W.n, j in 1:W.n]
    share = [count(c -> W.C_AP_mW[c, i] < capture * W.C_AP_mW[c, j], clients(i)) / W.clients_per_ap
             for i in 1:W.n, j in 1:W.n]
    return sensed, share
end

function mac_exposure(assign, ap, utilization, (sensed, share))
    sensed_load, clear = 0.0, 1.0
    @inbounds for j in eachindex(assign)
        (j == ap || assign[j] != assign[ap]) && continue
        if sensed[ap, j]
            sensed_load += utilization[j]
        else
            clear *= 1.0 - share[ap, j] * hidden_collision(utilization[j])
        end
    end
    return 1.0 + sensed_load, clear
end

struct VoiceModel
    W::World
    cfg::VoiceSimConfig
    mac::Tuple{Matrix{Bool},Matrix{Float64}}
end

VoiceModel(W::World, cfg::VoiceSimConfig) = VoiceModel(W, cfg, mac_tables(W))

struct VoiceWork
    raw::Vector{Float64}
    bitrate::Vector{Float64}
    utilization::Vector{Float64}
    service::Vector{Float64}
    residual::Vector{Float64}
    contenders::Vector{Float64}
    contention::Vector{Float64}
end

VoiceWork(n::Int) = VoiceWork((zeros(n) for _ in 1:7)...)

function voice_steady_state!(work::VoiceWork, model::VoiceModel, assign, act)
    (; W, cfg, mac) = model
    (; raw, bitrate, utilization, service, residual, contenders, contention) = work
    fill!(raw, 0.0)
    for c in eachindex(W.client_ap)
        ap = W.client_ap[c]
        assign[ap] != 0 && (raw[ap] += rate_unchecked(W, c, assign, act) * 1000.0 * cfg.frame_ms)
    end
    flows = 2 * cfg.calls_per_ap
    background = prioritized(cfg) ? 0.0 : cfg.background_load
    fill!(bitrate, cfg.codec_kbps)
    fill!(contenders, NaN)
    damping, previous = 1.0, Inf
    for _ in 1:10_000
        for ap in 1:W.n
            assign[ap] == 0 && continue
            utilization[ap] = min(1.0, cfg.background_load + flows * packet_bits(cfg, bitrate[ap]) / raw[ap])
        end
        change = 0.0
        for ap in 1:W.n
            assign[ap] == 0 && continue
            n_contenders, clear = mac_exposure(assign, ap, utilization, mac)
            if n_contenders != contenders[ap]
                contenders[ap] = n_contenders
                contention[ap] = dcf_collision(n_contenders)
            end
            collision = 1.0 - (1.0 - contention[ap]) * clear
            service[ap] = raw[ap] / sum(collision^i for i in 0:cfg.retry_limit)
            residual[ap] = collision^(cfg.retry_limit + 1)
            if adaptive(cfg)
                target = clamp((max(service[ap] - background * raw[ap], 0.0) / flows -
                                8.0 * cfg.packet_overhead_bytes) / cfg.frame_ms,
                               cfg.min_codec_kbps, cfg.codec_kbps)
                change = max(change, abs(target - bitrate[ap]))
                bitrate[ap] += damping * (target - bitrate[ap])
            end
        end
        change < 1e-12 && return work
        change >= previous && (damping /= 2)
        previous = change
    end
    error("voice steady state did not converge")
end

function delivered_fraction(work::VoiceWork, model::VoiceModel, ap::Int)
    cfg = model.cfg
    background = prioritized(cfg) ? 0.0 : cfg.background_load * work.raw[ap]
    offered = 2 * cfg.calls_per_ap * packet_bits(cfg, work.bitrate[ap]) + background
    return min(1.0, work.service[ap] / offered) * (1.0 - work.residual[ap])
end

function voice_steady_state(model::VoiceModel, assign, act)
    work = voice_steady_state!(VoiceWork(model.W.n), model, assign, act)
    members = filter(ap -> assign[ap] != 0, 1:model.W.n)
    return work.bitrate[members], [delivered_fraction(work, model, ap) for ap in members]
end

function voice_utility(model::VoiceModel, work::VoiceWork, assign, act)
    voice_steady_state!(work, model, assign, act)
    total = 0.0
    for ap in 1:model.W.n
        assign[ap] == 0 && continue
        quality = work.bitrate[ap] / model.cfg.codec_kbps * delivered_fraction(work, model, ap)
        total += log(max(quality, floatmin(Float64)))
    end
    return 2 * model.cfg.calls_per_ap * total
end

voice_utility(model::VoiceModel, assign, act) = voice_utility(model, VoiceWork(model.W.n), assign, act)

function subset_chunks(n::Int)
    masks = 1:((1 << n) - 1)
    return collect(Iterators.partition(masks, max(64, cld(length(masks), 8 * Threads.nthreads()))))
end

function mask_assign!(assign::Vector{Int}, mask::Int)
    for ap in eachindex(assign)
        assign[ap] = (mask >> (ap - 1)) & 1
    end
    return assign
end

function voice_subset_scores(W::World, cfg::VoiceSimConfig)
    model = VoiceModel(W, cfg)
    scores = zeros(1 << W.n)
    Threads.@threads :greedy for chunk in subset_chunks(W.n)
        assign, act, work = zeros(Int, W.n), similar(W.ap_activity), VoiceWork(W.n)
        for mask in chunk
            activity!(act, W, mask_assign!(assign, mask))
            scores[mask + 1] = voice_utility(model, work, assign, act)
        end
    end
    return scores
end

function weighted_quantile(values::Vector{Float64}, weights::Vector{Float64}, p::Float64)
    isempty(values) && return Inf
    order = sortperm(values)
    target = p * sum(weights)
    cumulative = 0.0
    for i in order
        cumulative += weights[i]
        cumulative >= target && return values[i]
    end
    return values[order[end]]
end

function simulate_voice(W::World, assign::AbstractVector{<:Integer}; cfg::VoiceSimConfig=VoiceSimConfig())
    validate_assignment(W, assign)
    validate_voice_config(cfg)
    frames = max(1, round(Int, 1000 * cfg.duration_s / cfg.frame_ms))
    drain_frames = ceil(Int, cfg.deadline_ms / cfg.frame_ms)
    packets_per_frame = 2 * cfg.calls_per_ap
    total_packets = frames * packets_per_frame * W.n
    capacity = voice_ap_capacity(W, assign)
    mac = mac_tables(W)
    utilization = zeros(W.n)
    contenders = fill(NaN, W.n)
    contention = zeros(W.n)
    queues = [QueuedWork[] for _ in 1:W.n]
    bitrate = fill(cfg.codec_kbps, W.n)
    delays = Float64[]
    delay_weights = Float64[]
    sizehint!(delays, 2 * frames * W.n)
    sizehint!(delay_weights, 2 * frames * W.n)
    delivered = 0.0
    deadline_lost = 0.0
    retry_lost = 0.0
    payload_delivered = 0.0
    codec_sum = 0.0
    overhead_bits = 8.0 * cfg.packet_overhead_bytes
    service = zeros(W.n)
    residual_loss = zeros(W.n)

    @inbounds for step in 1:(frames + drain_frames)
        start_ms = (step - 1) * cfg.frame_ms
        end_ms = step * cfg.frame_ms
        expiry_ms = start_ms + cfg.deadline_ms
        if step <= frames
            for ap in 1:W.n
                voice_bits = packets_per_frame * packet_bits(cfg, bitrate[ap])
                utilization[ap] = min(1.0, cfg.background_load + voice_bits / (capacity[ap] * 1000.0 * cfg.frame_ms))
            end
        end

        for ap in 1:W.n
            raw_service = capacity[ap] * 1000.0 * cfg.frame_ms
            queue = queues[ap]
            if step <= frames
                n_contenders, clear = mac_exposure(assign, ap, utilization, mac)
                if n_contenders != contenders[ap]
                    contenders[ap] = n_contenders
                    contention[ap] = dcf_collision(n_contenders)
                end
                collision = 1.0 - (1.0 - contention[ap]) * clear
                attempts = sum(collision^i for i in 0:cfg.retry_limit)
                service[ap] = raw_service / attempts
                residual_loss[ap] = collision^(cfg.retry_limit + 1)
                background = prioritized(cfg) ? 0.0 : cfg.background_load * raw_service

                if adaptive(cfg)
                    available = max(service[ap] - background, 0.0)
                    target = clamp((available / packets_per_frame - overhead_bits) / cfg.frame_ms,
                                   cfg.min_codec_kbps, cfg.codec_kbps)
                    bitrate[ap] += cfg.adaptation_speed * (target - bitrate[ap])
                end

                stream_bits = packets_per_frame * packet_bits(cfg, bitrate[ap]) + background
                push!(queue, QueuedWork(stream_bits, stream_bits / packets_per_frame,
                                        bitrate[ap] * cfg.frame_ms, start_ms, expiry_ms))
                codec_sum += bitrate[ap]
            end

            while !isempty(queue) && first(queue).deadline_ms < end_ms
                work = popfirst!(queue)
                deadline_lost += work.remaining_bits / work.packet_bits
            end

            budget = service[ap]
            while budget > 0 && !isempty(queue)
                work = first(queue)
                served = min(budget, work.remaining_bits)
                remaining = work.remaining_bits - served
                budget -= served
                packets = served / work.packet_bits
                successful = packets * (1.0 - residual_loss[ap])
                delivered += successful
                retry_lost += packets - successful
                payload_delivered += successful * work.payload_bits
                push!(delays, start_ms + cfg.frame_ms * (1.0 - budget / service[ap]) - work.arrival_ms)
                push!(delay_weights, successful)
                if remaining <= eps(Float64)
                    popfirst!(queue)
                else
                    queue[1] = QueuedWork(remaining, work.packet_bits, work.payload_bits,
                                          work.arrival_ms, work.deadline_ms)
                end
            end
        end
    end

    loss = (deadline_lost + retry_lost) / total_packets
    mean_delay = isempty(delays) ? Inf :
                 sum(delays[i] * delay_weights[i] for i in eachindex(delays)) / sum(delay_weights)
    p95_delay = weighted_quantile(delays, delay_weights, 0.95)
    mean_codec = codec_sum / (frames * W.n)
    simulated_s = frames * cfg.frame_ms / 1000.0
    delivered_mbps = payload_delivered / (simulated_s * 1.0e6)
    return VoiceResult(cfg.policy, total_packets, delivered, loss,
                       deadline_lost / total_packets, retry_lost / total_packets,
                       mean_delay, p95_delay, mean_codec, delivered_mbps)
end

compare_voice_strategies(W::World, assign::AbstractVector{<:Integer}; cfg::VoiceSimConfig=VoiceSimConfig()) =
    [simulate_voice(W, assign; cfg=VoiceSimConfig(cfg; policy)) for policy in POLICIES]

function score_unchecked(W::World, assign::AbstractVector{<:Integer})
    act = activity(W, assign)
    score = 0.0
    for c in eachindex(W.client_ap)
        score += log(rate_unchecked(W, c, assign, act))
    end
    return score
end

function contended_subset_scores(W::World)
    scores = zeros(1 << W.n)
    Threads.@threads :greedy for chunk in subset_chunks(W.n)
        assign, act = zeros(Int, W.n), similar(W.ap_activity)
        for mask in chunk
            activity!(act, W, mask_assign!(assign, mask))
            score = 0.0
            for c in eachindex(W.client_ap)
                assign[W.client_ap[c]] == 1 && (score += log(rate_unchecked(W, c, assign, act)))
            end
            scores[mask + 1] = score
        end
    end
    return scores
end

function total_score(W::World, assign::AbstractVector{<:Integer})
    validate_assignment(W, assign)
    return score_unchecked(W, assign)
end

function subset_sums(w::AbstractVector{Float64})
    sums = zeros(1 << length(w))
    for mask in 1:(length(sums) - 1)
        sums[mask + 1] = sums[(mask & (mask - 1)) + 1] + w[trailing_zeros(mask) + 1]
    end
    return sums
end

function subset_scores(W::World)
    all(>=(1.0), W.ap_activity) || return contended_subset_scores(W)
    n = W.n
    low = n >> 1
    lo_size = 1 << low
    scores = zeros(1 << n)
    clients = [findall(==(ap), W.client_ap) for ap in 1:n]
    extreme = max(abs(log(min(W.cfg.min_rate, W.cfg.demand_mbps))), abs(log(W.cfg.demand_mbps)), 1.0)
    group_size = clamp(floor(Int, 600 / extreme), 1, 16)
    sinr_cap = expm1(W.cfg.max_phy_mbps / W.rate_scale)
    sensed = [i != j && W.AP_AP[i, j] >= W.cfg.cca_dbm for i in 1:n, j in 1:n]
    blocked_tables = map(1:n) do own
        blocked = [sensed[own, j] ? W.ap_activity[j] : 0.0 for j in 1:n]
        (subset_sums(blocked[1:low]), subset_sums(blocked[(low + 1):n]))
    end
    hidden_tables = map(eachindex(W.client_ap)) do c
        own = W.client_ap[c]
        hidden = [j == own || sensed[own, j] ? 0.0 : W.C_AP_mW[c, j] * W.ap_activity[j] for j in 1:n]
        (subset_sums(hidden[1:low]), subset_sums(hidden[(low + 1):n]))
    end

    Threads.@threads :greedy for hi in 0:((1 << (n - low)) - 1)
        row = view(scores, (hi * lo_size + 1):((hi + 1) * lo_size))
        product = Vector{Float64}(undef, lo_size)
        @inbounds for own in 1:n
            own_lo = own <= low ? 1 << (own - 1) : 0
            own_hi = own > low ? 1 << (own - low - 1) : 0
            hi & own_hi == own_hi || continue
            blocked_lo, blocked_hi = blocked_tables[own]
            blocked = blocked_hi[hi + 1]
            for group in Iterators.partition(clients[own], group_size)
                fill!(product, 1.0)
                for c in group
                    hidden_lo, hidden_hi = hidden_tables[c]
                    base = W.noise_mw + hidden_hi[hi + 1]
                    signal = W.C_AP_mW[c, own]
                    for lo in 0:(lo_size - 1)
                        lo & own_lo == own_lo || continue
                        sinr = signal / (base + hidden_lo[lo + 1])
                        phy = sinr < sinr_cap ? W.rate_scale * log1p(sinr) : W.cfg.max_phy_mbps
                        capacity = W.cfg.mac_efficiency * phy /
                                   (W.clients_per_ap * (1.0 + (blocked + blocked_lo[lo + 1])))
                        product[lo + 1] *= min(W.cfg.demand_mbps, max(capacity, W.cfg.min_rate))
                    end
                end
                for lo in 0:(lo_size - 1)
                    lo & own_lo == own_lo && (row[lo + 1] += log(product[lo + 1]))
                end
            end
        end
    end
    return scores
end

function naive_coloring(W::World; thresh::Real=-75.0, k::Integer=W.cfg.k_channels)
    validate_k(W, k)
    n = W.n
    adj = BitMatrix(undef, n, n)
    for j in 1:n, i in 1:n
        adj[i, j] = i != j && W.AP_AP[i, j] > thresh
    end
    degree = vec(sum(adj; dims=2))
    order = sortperm(degree; rev=true)
    colors = zeros(Int, n)
    conflicts = zeros(Int, k)

    for v in order
        fill!(conflicts, 0)
        for u in 1:n
            if adj[v, u] && colors[u] != 0
                conflicts[colors[u]] += 1
            end
        end
        colors[v] = argmin(conflicts)
    end
    return colors
end

function channel_masks(assign, k)
    masks = zeros(Int, k)
    for (ap, ch) in enumerate(assign)
        masks[ch] |= 1 << (ap - 1)
    end
    return masks
end

struct Partition
    assign::Vector{Int}
    masks::Vector{Int}
end

Partition(assign::Vector{Int}, k::Integer) = Partition(assign, channel_masks(assign, k))

function Base.copyto!(dst::Partition, src::Partition)
    copyto!(dst.assign, src.assign)
    copyto!(dst.masks, src.masks)
    return dst
end

masks_score(scores, part::Partition) = sum(mask -> scores[mask + 1], part.masks)

@inline function move_delta(scores, part::Partition, ap, to)
    masks, bit, from = part.masks, 1 << (ap - 1), part.assign[ap]
    return scores[(masks[from] & ~bit) + 1] + scores[(masks[to] | bit) + 1] -
           scores[masks[from] + 1] - scores[masks[to] + 1]
end

@inline function swap_delta(scores, part::Partition, a, b)
    masks, from, to = part.masks, part.assign[a], part.assign[b]
    bit_a, bit_b = 1 << (a - 1), 1 << (b - 1)
    return scores[((masks[from] & ~bit_a) | bit_b) + 1] +
           scores[((masks[to] & ~bit_b) | bit_a) + 1] -
           scores[masks[from] + 1] - scores[masks[to] + 1]
end

function move!(part::Partition, ap, to)
    bit = 1 << (ap - 1)
    part.masks[part.assign[ap]] &= ~bit
    part.masks[to] |= bit
    part.assign[ap] = to
    return part
end

function canonical!(assign::Vector{Int})
    labels = zeros(Int, maximum(assign))
    next = 0
    for (ap, ch) in enumerate(assign)
        labels[ch] == 0 && (labels[ch] = next += 1)
        assign[ap] = labels[ch]
    end
    return assign
end

other_channel(rng, old, k) =(new = rand(rng, 1:(k - 1)); new + (new >= old))

function polish!(scores, part::Partition)
    assign = part.assign
    improved = true
    while improved
        improved = false
        for a in eachindex(assign), q in eachindex(part.masks)
            if q != assign[a] && move_delta(scores, part, a, q) > 1e-12
                move!(part, a, q)
                improved = true
            end
        end
        for a in eachindex(assign), b in (a + 1):length(assign)
            p, q = assign[a], assign[b]
            if p != q && swap_delta(scores, part, a, b) > 1e-12
                move!(part, a, q)
                move!(part, b, p)
                improved = true
            end
        end
    end
    return part
end

interference_caused(W::World) =
    [sum(W.C_AP_mW[c, j] for c in eachindex(W.client_ap) if W.client_ap[c] != j; init=0.0)
     for j in 1:W.n]

function greedy_fill!(part::Partition, scores, order)
    masks = part.masks
    fill!(masks, 0)
    for ap in order
        bit = 1 << (ap - 1)
        ch = argmax(c -> scores[(masks[c] | bit) + 1] - scores[masks[c] + 1], eachindex(masks))
        masks[ch] |= bit
        part.assign[ap] = ch
    end
    return part
end

function greedy_rate(W::World; k::Integer=W.cfg.k_channels, refine::Bool=true)
    validate_k(W, k)
    scores = subset_scores(W)
    part = Partition(zeros(Int, W.n), zeros(Int, k))
    greedy_fill!(part, scores, sortperm(interference_caused(W); rev=true))
    refine && polish!(scores, part)
    return canonical!(part.assign)
end

function initial_temperature(scores, part::Partition, rng)
    total, downhill = 0.0, 0
    for _ in 1:200
        ap = rand(rng, eachindex(part.assign))
        delta = move_delta(scores, part, ap, other_channel(rng, part.assign[ap], length(part.masks)))
        if delta < 0
            total -= delta
            downhill += 1
        end
    end
    return downhill == 0 ? 1.0 : total / (downhill * log(2))
end

function simulated_annealing(W::World; k::Integer=W.cfg.k_channels, seed::Union{Nothing,Integer}=nothing,
                             cfg::AnnealConfig=AnnealConfig())
    validate_k(W, k)
    cfg.iters >= 0 || throw(ArgumentError("iteration count must be nonnegative"))
    isnothing(cfg.T0) || isfinite(cfg.T0) && cfg.T0 > 0 ||
        throw(ArgumentError("initial temperature must be positive and finite"))
    isnothing(cfg.cooling) || isfinite(cfg.cooling) && 0 < cfg.cooling <= 1 ||
        throw(ArgumentError("cooling must be within (0, 1]"))
    decay = isnothing(cfg.cooling) ? exp(log(0.01) / max(cfg.iters, 1)) : cfg.cooling
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    scores = subset_scores(W)
    part = Partition(rand(rng, 1:k, W.n), k)
    score = masks_score(scores, part)
    history = Vector{Float64}(undef, cfg.iters)

    if k == 1
        fill!(history, score)
        return part.assign, score, history
    end

    best = copy(part.assign)
    best_score = score
    temperature = isnothing(cfg.T0) ? initial_temperature(scores, part, rng) : cfg.T0
    for step in eachindex(history)
        ap = rand(rng, 1:W.n)
        to = other_channel(rng, part.assign[ap], k)
        delta = move_delta(scores, part, ap, to)
        if delta >= 0 || delta > -20temperature && rand(rng) < exp(delta / temperature)
            move!(part, ap, to)
            score += delta
            if score > best_score
                best_score = score
                copyto!(best, part.assign)
            end
        end
        history[step] = score
        temperature *= decay
    end
    cfg.polish && polish!(scores, Partition(best, k))
    return canonical!(best), total_score(W, best), history
end

function optimize_assignment(W::World; k::Integer=W.cfg.k_channels, seed::Union{Nothing,Integer}=nothing,
                             voice::Union{Nothing,VoiceSimConfig}=nothing)
    validate_k(W, k)
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    scores = subset_scores(W)
    if !isnothing(voice)
        validate_voice_config(voice)
        scores .+= voice_subset_scores(W, voice)
    end
    greedy_order = sortperm(interference_caused(W); rev=true)
    best, best_score = zeros(Int, W.n), -Inf
    part = Partition(zeros(Int, W.n), zeros(Int, k))
    trial = Partition(zeros(Int, W.n), zeros(Int, k))

    for restart in 1:RESTARTS
        greedy_fill!(part, scores, restart == 1 ? greedy_order : randperm(rng, W.n))
        polish!(scores, part)
        score = masks_score(scores, part)
        for _ in 1:KICKS
            copyto!(trial, part)
            for _ in 1:rand(rng, min(2, W.n):min(4, W.n))
                move!(trial, rand(rng, 1:W.n), rand(rng, 1:k))
            end
            polish!(scores, trial)
            trial_score = masks_score(scores, trial)
            if trial_score >= score
                copyto!(part, trial)
                score = trial_score
            end
        end
        if score > best_score
            copyto!(best, part.assign)
            best_score = score
        end
    end
    return canonical!(best), best_score
end

@inline function best_split(scores, previous, mask)
    low = mask & -mask
    rest = mask ⊻ low
    value, block, sub = -Inf, low, rest
    while true
        candidate = scores[(sub | low) + 1] + previous[((rest ⊻ sub) >> 1) + 1]
        candidate > value && ((value, block) = (candidate, sub | low))
        sub == 0 && break
        sub = (sub - 1) & rest
    end
    return value, block
end

function brute_force_fast(W::World; k::Integer=W.cfg.k_channels)
    validate_k(W, k)
    if k >= W.n
        assign = collect(1:W.n)
        return assign, total_score(W, assign)
    end
    scores = subset_scores(W)
    k == 1 && return ones(Int, W.n), scores[end]
    full = (1 << W.n) - 1
    best = scores[1:2:end]
    picks = Vector{Vector{Int32}}(undef, k - 1)
    for layer in 2:(k - 1)
        previous = best
        next = zeros(length(previous))
        pick = zeros(Int32, length(previous))
        Threads.@threads :greedy for chunk in collect(Iterators.partition(2:2:full, 4096))
            for mask in chunk
                next[(mask >> 1) + 1], pick[(mask >> 1) + 1] = best_split(scores, previous, mask)
            end
        end
        best, picks[layer] = next, pick
    end
    value, block = best_split(scores, best, full)
    assign = zeros(Int, W.n)
    mask = full
    for layer in k:-1:1
        for ap in 1:W.n
            block & (1 << (ap - 1)) != 0 && (assign[ap] = layer)
        end
        mask ⊻= block
        mask == 0 && break
        block = layer == 2 ? mask : Int(picks[layer - 1][(mask >> 1) + 1])
    end
    return canonical!(assign), value
end

function brute_force_naive(W::World)
    total = 1
    for _ in 1:W.n
        total > div(1_000_000, W.cfg.k_channels) &&
            throw(ArgumentError("naive search is limited to one million assignments"))
        total *= W.cfg.k_channels
    end
    assign = ones(Int, W.n)
    best = similar(assign)
    best_score = -Inf
    k = W.cfg.k_channels

    for code in 0:(total - 1)
        value = code
        for ap in eachindex(assign)
            assign[ap] = (value % k) + 1
            value ÷= k
        end
        score = score_unchecked(W, assign)
        if score > best_score
            best_score = score
            copyto!(best, assign)
        end
    end
    return best, best_score
end

end
