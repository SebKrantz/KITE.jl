# Convergence criteria, all measured as relative changes with a `max(|old|, eps)` denominator
# so that zero entries cannot produce a division by zero.
#
# Unlike the R implementation, `:sample` uses a deterministic stride rather than `sample()`, so
# repeated runs of the same problem give identical results.

"""
    _criterion(new, old, method::Symbol) -> Float64

Relative distance between successive iterates.

* `:root_mean_square` — RMS of the relative changes. The default: a good balance of sensitivity
  and cost.
* `:aggregate` — relative change of the sums. Cheapest, but blind to offsetting movements.
* `:element_wise` — largest relative change. Strictest and most expensive.
* `:sample` — `:element_wise` over a deterministic stride of about a tenth of the elements.

Returns `0.0` when both iterates are empty or entirely non-finite.
"""
function _criterion(new_values, old_values, method::Symbol)
    length(new_values) == length(old_values) ||
        throw(DimensionMismatch("new_values has length $(length(new_values)), old_values " *
                                "$(length(old_values))."))
    n = length(new_values)
    n == 0 && return 0.0

    if method === :aggregate
        s_new = _finite_sum(new_values)
        s_old = _finite_sum(old_values)
        (s_new == 0 && s_old == 0) && return 0.0
        return abs((s_new - s_old) / max(abs(s_old), eps()))

    elseif method === :root_mean_square
        acc = 0.0
        cnt = 0
        @inbounds @simd for i in eachindex(new_values, old_values)
            a = new_values[i]
            b = old_values[i]
            if isfinite(a) && isfinite(b)
                r = (a - b) / max(abs(b), eps())
                acc += r * r
                cnt += 1
            end
        end
        cnt == 0 && return 0.0
        return sqrt(acc / cnt)

    elseif method === :element_wise
        return _max_rel(new_values, old_values, eachindex(new_values, old_values))

    elseif method === :sample
        n_sample = min(n, max(10, round(Int, 0.1 * n)))
        step = max(1, n ÷ n_sample)
        return _max_rel(new_values, old_values, firstindex(new_values):step:lastindex(new_values))
    end

    error("method must be :root_mean_square, :aggregate, :element_wise or :sample; got :$method.")
end

function _max_rel(new_values, old_values, idx)
    m = 0.0
    found = false
    @inbounds for i in idx
        a = new_values[i]
        b = old_values[i]
        if isfinite(a) && isfinite(b)
            r = abs(a - b) / max(abs(b), eps())
            m = ifelse(r > m, r, m)
            found = true
        end
    end
    return found ? m : 0.0
end

function _finite_sum(x)
    s = 0.0
    @inbounds @simd for i in eachindex(x)
        v = x[i]
        s += ifelse(isfinite(v), v, zero(v))
    end
    return s
end
