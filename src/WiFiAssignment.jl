module WiFiAssignment

using Random
using Base: @kwdef

export SimConfig, World, make_world, client_rate, client_rates, total_score,
       naive_coloring, greedy_rate, simulated_annealing, optimize_assignment, brute_force_fast,
       brute_force_naive, VoiceSimConfig, VoiceResult, simulate_voice,
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

# 802.11 OFDM PHY backoff: CWmin 15 (16 slots) doubling over 6 stages to CWmax 1023.
const CW_MIN = 16
const BACKOFF_STAGES = 6
# A hidden AP only destroys a frame it reaches within this margin of the wanted signal (ns-2 CPThresh).
const CAPTURE_DB = 10.0

# Bianchi's DCF fixed point: per-attempt collision probability among `n` saturated contenders
# (fractional when contenders are only partly busy). f(p) = p - 1 + (1 - τ(p))^(n-1) is increasing,
# so a Newton step that leaves the bracket falls back to bisection.
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

# Probability a frame overlaps one hidden transmitter that is busy a fraction `u` of the time with
# frames as long as ours and exponential idle gaps: busy at the start, or starting during the frame.
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

# Airtime each AP needs for its clients' demand once hidden co-channel APs interfere in
# proportion to their own airtime. The map is monotone, so iterating from the
# interference-free activity reaches the least fixed point. Channel 0 marks an absent AP.
function activity(W::World, assign::AbstractVector{<:Integer})
    all(>=(1.0), W.ap_activity) && return W.ap_activity
    act = copy(W.ap_activity)
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
end

function voice_ap_capacity(W::World, assign::AbstractVector{<:Integer})
    capacity = zeros(W.n)
    act = activity(W, assign)
    @inbounds for c in eachindex(W.client_ap)
        capacity[W.client_ap[c]] += rate_unchecked(W, c, assign, act)
    end
    return capacity
end

# Which APs carrier-sense each other, and the share of AP i's clients that AP j reaches within the
# capture margin (so a hidden j corrupts their frames). Neither depends on the channel assignment.
function mac_tables(W::World)
    capture = 10.0^(CAPTURE_DB / 10.0)
    clients(i) = ((i - 1) * W.clients_per_ap + 1):(i * W.clients_per_ap)
    sensed = [i != j && W.AP_AP[i, j] >= W.cfg.cca_dbm for i in 1:W.n, j in 1:W.n]
    share = [count(c -> W.C_AP_mW[c, i] < capture * W.C_AP_mW[c, j], clients(i)) / W.clients_per_ap
             for i in 1:W.n, j in 1:W.n]
    return sensed, share
end

# Airtime of the sensed co-channel APs (Bianchi contenders beyond `ap` itself) and the probability
# that no hidden co-channel AP overlaps `ap`'s frame.
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

function validate_policy(policy::Symbol)
    policy in (:fifo, :deadline, :adaptive, :combined) ||
        throw(ArgumentError("voice policy must be :fifo, :deadline, :adaptive, or :combined"))
end

# Steady state of `simulate_voice` for the present APs: utilization, collisions and the codec target are
# iterated to their joint fixed point (where each codec equals its target), halving the step whenever
# it stops contracting. Returns each present AP's codec rate and the fraction of its packets delivered.
function voice_steady_state(W::World, assign, act, cfg::VoiceSimConfig, policy::Symbol, mac=mac_tables(W))
    members = [ap for ap in 1:W.n if assign[ap] != 0]
    raw = zeros(W.n)
    for c in eachindex(W.client_ap)
        ap = W.client_ap[c]
        assign[ap] != 0 && (raw[ap] += rate_unchecked(W, c, assign, act) * 1000.0 * cfg.frame_ms)
    end
    flows = 2 * cfg.calls_per_ap
    overhead = 8.0 * cfg.packet_overhead_bytes
    background = policy in (:deadline, :combined) ? 0.0 : cfg.background_load
    bitrate = fill(cfg.codec_kbps, W.n)
    utilization, service, residual = zeros(W.n), zeros(W.n), zeros(W.n)
    contenders, contention = fill(NaN, W.n), zeros(W.n)
    damping, previous, converged = 1.0, Inf, false
    for _ in 1:10_000
        for ap in members
            utilization[ap] = min(1.0, cfg.background_load + flows * (bitrate[ap] * cfg.frame_ms + overhead) / raw[ap])
        end
        change = 0.0
        for ap in members
            n_contenders, clear = mac_exposure(assign, ap, utilization, mac)
            if n_contenders != contenders[ap]
                contenders[ap] = n_contenders
                contention[ap] = dcf_collision(n_contenders)
            end
            collision = 1.0 - (1.0 - contention[ap]) * clear
            service[ap] = raw[ap] / sum(collision^i for i in 0:cfg.retry_limit)
            residual[ap] = collision^(cfg.retry_limit + 1)
            if policy in (:adaptive, :combined)
                target = clamp((max(service[ap] - background * raw[ap], 0.0) / flows - overhead) / cfg.frame_ms,
                               cfg.min_codec_kbps, cfg.codec_kbps)
                change = max(change, abs(target - bitrate[ap]))
                bitrate[ap] += damping * (target - bitrate[ap])
            end
        end
        converged = change < 1e-12
        converged && break
        change >= previous && (damping /= 2)
        previous = change
    end
    converged || error("voice steady state did not converge")
    delivered = [min(1.0, service[ap] / (flows * (bitrate[ap] * cfg.frame_ms + overhead) + background * raw[ap])) *
                 (1.0 - residual[ap]) for ap in members]
    return bitrate[members], delivered
end

# Each voice flow contributes log(delivered codec rate / nominal rate), so codec quality and loss both count.
function voice_utility(W::World, assign, act, cfg::VoiceSimConfig, policy::Symbol, mac=mac_tables(W))
    bitrate, delivered = voice_steady_state(W, assign, act, cfg, policy, mac)
    return 2 * cfg.calls_per_ap *
           sum(log(max(r / cfg.codec_kbps * d, floatmin(Float64))) for (r, d) in zip(bitrate, delivered); init=0.0)
end

function voice_subset_scores(W::World, cfg::VoiceSimConfig, policy::Symbol)
    scores = zeros(1 << W.n)
    mac = mac_tables(W)
    Threads.@threads :greedy for mask in 1:(length(scores) - 1)
        assign = [(mask >> (ap - 1)) & 1 for ap in 1:W.n]
        scores[mask + 1] = voice_utility(W, assign, activity(W, assign), cfg, policy, mac)
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

function simulate_voice(W::World, assign::AbstractVector{<:Integer};
                        policy::Symbol=:combined, cfg::VoiceSimConfig=VoiceSimConfig())
    validate_assignment(W, assign)
    validate_voice_config(cfg)
    validate_policy(policy)
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
    adaptive = policy in (:adaptive, :combined)
    scored = policy in (:deadline, :combined)
    overhead_bits = 8.0 * cfg.packet_overhead_bytes
    service = zeros(W.n)
    residual_loss = zeros(W.n)

    @inbounds for step in 1:(frames + drain_frames)
        start_ms = (step - 1) * cfg.frame_ms
        end_ms = step * cfg.frame_ms
        expiry_ms = start_ms + cfg.deadline_ms
        if step <= frames
            for ap in 1:W.n
                voice_bits = packets_per_frame * (bitrate[ap] * cfg.frame_ms + overhead_bits)
                utilization[ap] = min(1.0, cfg.background_load + voice_bits / (capacity[ap] * 1000.0 * cfg.frame_ms))
            end
        end

        for ap in 1:W.n
            raw_service = capacity[ap] * 1000.0 * cfg.frame_ms
            queue = queues[ap]
            if step <= frames
                packet_bits = bitrate[ap] * cfg.frame_ms + overhead_bits
                n_contenders, clear = mac_exposure(assign, ap, utilization, mac)
                if n_contenders != contenders[ap]
                    contenders[ap] = n_contenders
                    contention[ap] = dcf_collision(n_contenders)
                end
                collision = 1.0 - (1.0 - contention[ap]) * clear
                attempts = sum(collision^i for i in 0:cfg.retry_limit)
                service[ap] = raw_service / attempts
                residual_loss[ap] = collision^(cfg.retry_limit + 1)
                background = scored ? 0.0 : cfg.background_load * raw_service

                if adaptive
                    available = max(service[ap] - background, 0.0)
                    target = clamp((available / packets_per_frame - overhead_bits) / cfg.frame_ms,
                                   cfg.min_codec_kbps, cfg.codec_kbps)
                    bitrate[ap] += cfg.adaptation_speed * (target - bitrate[ap])
                    packet_bits = bitrate[ap] * cfg.frame_ms + overhead_bits
                end

                # Under FIFO the frame's voice and background interleave, so one item carries both
                # and each voice packet accounts for its share of the mixed stream.
                stream_bits = packets_per_frame * packet_bits + background
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
    return VoiceResult(policy, total_packets, delivered, loss,
                       deadline_lost / total_packets, retry_lost / total_packets,
                       mean_delay, p95_delay, mean_codec, delivered_mbps)
end

function compare_voice_strategies(W::World, assign::AbstractVector{<:Integer};
                                  cfg::VoiceSimConfig=VoiceSimConfig())
    return [simulate_voice(W, assign; policy=policy, cfg=cfg)
            for policy in (:fifo, :deadline, :adaptive, :combined)]
end

function score_unchecked(W::World, assign::AbstractVector{<:Integer})
    act = activity(W, assign)
    score = 0.0
    for c in eachindex(W.client_ap)
        score += log(rate_unchecked(W, c, assign, act))
    end
    return score
end

# With unsaturated APs the airtime depends on the channel group, so each entry needs its own fixed point.
function contended_subset_scores(W::World)
    scores = zeros(1 << W.n)
    Threads.@threads :greedy for mask in 1:(length(scores) - 1)
        assign = [(mask >> (ap - 1)) & 1 for ap in 1:W.n]
        act = activity(W, assign)
        scores[mask + 1] = sum(log(rate_unchecked(W, c, assign, act))
                               for c in eachindex(W.client_ap) if assign[W.client_ap[c]] == 1)
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

# With every AP saturated, interference and blocked airtime add over APs, so each client needs only two
# small subset-sum tables of interference (low and high halves of the mask) and each AP two of blocked
# airtime; rows of the result are independent. Up to 16 client rates are multiplied before one log,
# which stays in Float64 range while rates lie within 1e±19.
function subset_scores(W::World)
    all(>=(1.0), W.ap_activity) || return contended_subset_scores(W)
    n = W.n
    low = n >> 1
    lo_size = 1 << low
    scores = zeros(1 << n)
    clients = [findall(==(ap), W.client_ap) for ap in 1:n]
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
            for group in Iterators.partition(clients[own], 16)
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

masks_score(scores, masks) = sum(mask -> scores[mask + 1], masks)

@inline function move_delta(scores, masks, bit, from, to)
    return scores[(masks[from] & ~bit) + 1] + scores[(masks[to] | bit) + 1] -
           scores[masks[from] + 1] - scores[masks[to] + 1]
end

@inline function swap_delta(scores, masks, bit_a, from, bit_b, to)
    return scores[((masks[from] & ~bit_a) | bit_b) + 1] +
           scores[((masks[to] & ~bit_b) | bit_a) + 1] -
           scores[masks[from] + 1] - scores[masks[to] + 1]
end

function move!(assign, masks, ap, to)
    bit = 1 << (ap - 1)
    masks[assign[ap]] &= ~bit
    masks[to] |= bit
    assign[ap] = to
end

other_channel(rng, old, k) = (new = rand(rng, 1:(k - 1)); new + (new >= old))

# Hill climb over single-AP moves and pairwise swaps until neither improves the score.
function polish!(scores, assign, masks)
    improved = true
    while improved
        improved = false
        for a in eachindex(assign), q in eachindex(masks)
            p = assign[a]
            if q != p && move_delta(scores, masks, 1 << (a - 1), p, q) > 1e-12
                move!(assign, masks, a, q)
                improved = true
            end
        end
        for a in eachindex(assign), b in (a + 1):length(assign)
            p, q = assign[a], assign[b]
            if p != q && swap_delta(scores, masks, 1 << (a - 1), p, 1 << (b - 1), q) > 1e-12
                move!(assign, masks, a, q)
                move!(assign, masks, b, p)
                improved = true
            end
        end
    end
    return assign
end

interference_caused(W::World) =
    [sum(W.C_AP_mW[c, j] for c in eachindex(W.client_ap) if W.client_ap[c] != j; init=0.0)
     for j in 1:W.n]

function greedy_fill!(assign, masks, scores, order)
    fill!(masks, 0)
    for ap in order
        bit = 1 << (ap - 1)
        ch = argmax(c -> scores[(masks[c] | bit) + 1] - scores[masks[c] + 1], eachindex(masks))
        masks[ch] |= bit
        assign[ap] = ch
    end
    return assign
end

function greedy_rate(W::World; k::Integer=W.cfg.k_channels, refine::Bool=true)
    validate_k(W, k)
    scores = subset_scores(W)
    assign = zeros(Int, W.n)
    masks = zeros(Int, k)
    greedy_fill!(assign, masks, scores, sortperm(interference_caused(W); rev=true))
    refine && polish!(scores, assign, masks)
    return assign
end

# Temperature at which an average downhill move from the starting point is accepted half the time.
function initial_temperature(scores, assign, masks, rng)
    total, downhill = 0.0, 0
    for _ in 1:200
        ap = rand(rng, eachindex(assign))
        to = other_channel(rng, assign[ap], length(masks))
        delta = move_delta(scores, masks, 1 << (ap - 1), assign[ap], to)
        if delta < 0
            total -= delta
            downhill += 1
        end
    end
    return downhill == 0 ? 1.0 : total / (downhill * log(2))
end

function simulated_annealing(W::World; k::Integer=W.cfg.k_channels,
                             iters::Integer=3000, T0::Union{Nothing,Real}=nothing,
                             cooling::Union{Nothing,Real}=nothing,
                             seed::Union{Nothing,Integer}=nothing, polish::Bool=true)
    validate_k(W, k)
    iters >= 0 || throw(ArgumentError("iteration count must be nonnegative"))
    isnothing(T0) || isfinite(T0) && T0 > 0 ||
        throw(ArgumentError("initial temperature must be positive and finite"))
    if !isnothing(cooling)
        isfinite(cooling) && 0 < cooling <= 1 || throw(ArgumentError("cooling must be within (0, 1]"))
    end
    decay = isnothing(cooling) ? exp(log(0.01) / max(iters, 1)) : Float64(cooling)
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    scores = subset_scores(W)
    assign = rand(rng, 1:k, W.n)
    masks = channel_masks(assign, k)
    score = masks_score(scores, masks)
    history = Vector{Float64}(undef, iters)

    if k == 1
        fill!(history, score)
        return assign, score, history
    end

    best = copy(assign)
    best_score = score
    temperature = isnothing(T0) ? initial_temperature(scores, assign, masks, rng) : Float64(T0)
    for step in eachindex(history)
        ap = rand(rng, 1:W.n)
        to = other_channel(rng, assign[ap], k)
        delta = move_delta(scores, masks, 1 << (ap - 1), assign[ap], to)
        if delta >= 0 || delta > -20temperature && rand(rng) < exp(delta / temperature)
            move!(assign, masks, ap, to)
            score += delta
            if score > best_score
                best_score = score
                copyto!(best, assign)
            end
        end
        history[step] = score
        temperature *= decay
    end
    polish && polish!(scores, best, channel_masks(best, k))
    return best, total_score(W, best), history
end

# With `voice`, every voice flow joins the clients in the proportional-fair objective (see `voice_utility`).
function optimize_assignment(W::World; k::Integer=W.cfg.k_channels, restarts::Integer=8,
                             kicks::Integer=200, seed::Union{Nothing,Integer}=nothing,
                             voice::Union{Nothing,VoiceSimConfig}=nothing, policy::Symbol=:combined)
    validate_k(W, k)
    restarts >= 1 && kicks >= 0 ||
        throw(ArgumentError("restarts must be positive and kicks nonnegative"))
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    scores = subset_scores(W)
    if !isnothing(voice)
        validate_voice_config(voice)
        validate_policy(policy)
        scores .+= voice_subset_scores(W, voice, policy)
    end
    greedy_order = sortperm(interference_caused(W); rev=true)
    best, best_score = zeros(Int, W.n), -Inf
    assign, masks = zeros(Int, W.n), zeros(Int, k)
    trial, trial_masks = similar(assign), similar(masks)

    for restart in 1:restarts
        greedy_fill!(assign, masks, scores, restart == 1 ? greedy_order : randperm(rng, W.n))
        polish!(scores, assign, masks)
        score = masks_score(scores, masks)
        for _ in 1:kicks
            copyto!(trial, assign)
            copyto!(trial_masks, masks)
            for _ in 1:rand(rng, min(2, W.n):min(4, W.n))
                move!(trial, trial_masks, rand(rng, 1:W.n), rand(rng, 1:k))
            end
            polish!(scores, trial, trial_masks)
            trial_score = masks_score(scores, trial_masks)
            if trial_score >= score
                copyto!(assign, trial)
                copyto!(masks, trial_masks)
                score = trial_score
            end
        end
        if score > best_score
            copyto!(best, assign)
            best_score = score
        end
    end
    return best, best_score
end

# Best split of `mask` into one more channel than `previous` covers: the channel holding the mask's
# lowest AP plus the best split of the rest. `previous` is indexed by masks without AP 1 (even masks).
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

# Exact optimum over every split of the APs into at most k channels. Layer j holds the best split of
# each mask into j channels. AP 1 always opens the first channel, so lower layers only need masks
# without it, and the top layer only the full set. With k >= n every AP gets its own channel, since
# sharing a channel only lowers rates.
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
    return assign, value
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
