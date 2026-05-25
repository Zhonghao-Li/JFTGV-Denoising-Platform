function result = ftvd_core_adaptive(f, kernel, mu, beta0, beta_max, opts)
%FTVD_CORE_ADAPTIVE  FTVd with adaptive β-continuation scheduling.
%
%   Extends standard FTVd with an adaptive β-growth rule.  Instead of
%   always doubling β (β ← 2β), the multiplier is adjusted based on how
%   quickly the inner loop converges at the current β:
%
%     - Fast convergence (few inner iterations) → larger multiplier
%     - Slow convergence (many inner iterations) → smaller multiplier
%
%   This avoids wasting iterations when β is far from its effective limit
%   and prevents overshooting when the problem becomes stiff.
%
%   Adaptive rule:
%     n_inner = actual iterations used in this β-stage
%     β_mult = clamp( β_mult_max - (n_inner-1)/N_ref * (β_mult_max - β_mult_min),
%                     β_mult_min, β_mult_max )
%
%   where N_ref = 30 is the reference point (n_inner = N_ref → β_mult ≈ mid).
%
%   Usage:
%     result = ftvd_core_adaptive(f, kernel, mu, beta0, beta_max, opts)
%
%   opts fields (in addition to standard ftvd_core opts):
%     beta_adaptive    - true/false (default: false → fixed ξ=2.0)
%     beta_mult_fixed  - fixed multiplier when adaptive=false (default: 2.0)
%     beta_mult_min    - min multiplier for adaptive mode (default: 1.3)
%     beta_mult_max    - max multiplier for adaptive mode (default: 4.0)
%     beta_n_ref       - reference inner-iters for half-range (default: 30)
%     return_beta_hist - also return β schedule history (default: false)

if nargin < 2 || isempty(kernel), kernel = 1; end
if nargin < 3 || isempty(mu), mu = 350; end
if nargin < 4 || isempty(beta0), beta0 = 1; end
if nargin < 5 || isempty(beta_max), beta_max = 128; end
if nargin < 6, opts = struct(); end

% Standard opts
if ~isfield(opts, 'tol'),          opts.tol = 1e-4; end
if ~isfield(opts, 'max_iter'),     opts.max_iter = 1000; end
if ~isfield(opts, 'inner_iter'),   opts.inner_iter = 100; end
if ~isfield(opts, 'verbose'),      opts.verbose = false; end
if ~isfield(opts, 'gt'),           opts.gt = []; end

% Adaptive β opts
if ~isfield(opts, 'beta_adaptive'),    opts.beta_adaptive = false; end
if ~isfield(opts, 'beta_mult_fixed'),  opts.beta_mult_fixed = 2.0; end
if ~isfield(opts, 'beta_mult_min'),    opts.beta_mult_min = 1.3; end
if ~isfield(opts, 'beta_mult_max'),    opts.beta_mult_max = 4.0; end
if ~isfield(opts, 'beta_n_ref'),       opts.beta_n_ref = 30; end
if ~isfield(opts, 'return_beta_hist'), opts.return_beta_hist = false; end

f = im2double(f);
F = precompute_fft_operators(f, kernel, 0);

u = f;
[dux, duy] = grad_forward(u);
w = cat(3, dux, duy);

max_hist = min(opts.max_iter, 10000);
hist_rel   = zeros(max_hist, 1);
hist_beta  = zeros(max_hist, 1);
hist_mult  = zeros(max_hist, 1);   % record multipliers for analysis

beta = beta0;
k_total = 0;
stage_count = 0;

beta_mult = opts.beta_mult_fixed;  % initial (will be updated if adaptive)

while beta <= beta_max && k_total < opts.max_iter
    mu_over_beta = mu / beta;
    stage_count = stage_count + 1;

    % ---- Inner loop ----
    inner_iters_used = 0;
    for k_inner = 1:opts.inner_iter
        if k_total >= opts.max_iter, break; end

        u_old = u;

        [dux, duy] = grad_forward(u);
        w = isotropic_shrink_2d(dux, duy, 1 / beta);
        u = F.update(w, mu_over_beta);

        rel = norm(u(:) - u_old(:)) / max(norm(u_old(:)), eps);
        k_total = k_total + 1;
        inner_iters_used = inner_iters_used + 1;

        if k_total <= max_hist
            hist_rel(k_total) = rel;
            hist_beta(k_total) = beta;
        end

        if opts.verbose && mod(k_total, 10) == 0
            fprintf('  [FTVd-adapt] iter=%4d  β=%8.2f  mult=%.2f  rel=%.2e\n', ...
                k_total, beta, beta_mult, rel);
        end

        if rel < opts.tol, break; end
    end

    % ---- Adaptive β multiplier ----
    if opts.beta_adaptive
        beta_mult = compute_adaptive_mult(inner_iters_used, ...
            opts.beta_mult_min, opts.beta_mult_max, opts.beta_n_ref);

        if opts.verbose
            fprintf('  → β-stage %2d:  n_inner=%3d  β=%7.2f → β_mult=%.2f → β=%.2f\n', ...
                stage_count, inner_iters_used, beta, beta_mult, min(beta * beta_mult, beta_max));
        end
    else
        beta_mult = opts.beta_mult_fixed;
    end

    hist_mult(stage_count) = beta_mult;

    % Safety: don't jump past beta_max by more than one step
    next_beta = beta * beta_mult;
    if next_beta > beta_max && beta < beta_max
        beta_mult = beta_max / beta;
        next_beta = beta_max;
    end

    beta = next_beta;
end

% Trim history
hist_rel  = hist_rel(1:k_total);
hist_beta = hist_beta(1:k_total);
hist_mult = hist_mult(1:stage_count);

result.u = u;
result.iters = k_total;
result.beta_stages = stage_count;
result.history.rel_change = hist_rel;
result.history.beta = hist_beta;

if opts.return_beta_hist
    result.history.beta_multipliers = hist_mult;
end
result.beta_adaptive = opts.beta_adaptive;

if ~isempty(opts.gt)
    result.psnr = compute_psnr(u, opts.gt);
end
end

function beta_mult = compute_adaptive_mult(n_inner, mult_min, mult_max, n_ref)
%COMPUTE_ADAPTIVE_MULT  Map inner iterations to β growth multiplier.
%
%   Uses linear interpolation:
%     β_mult = mult_max - (n_inner - 1) / n_ref * (mult_max - mult_min)
%
%   clamped to [mult_min, mult_max].
%
%   Examples:
%     n_inner = 1  → β_mult ≈ mult_max  (trivial subproblem, accelerate)
%     n_inner = n_ref → β_mult ≈ (min+max)/2  (moderate)
%     n_inner = 2*n_ref → β_mult = mult_min  (stiff, slow down)

beta_mult = mult_max - (n_inner - 1) / n_ref * (mult_max - mult_min);
beta_mult = max(mult_min, min(mult_max, beta_mult));
end

%% ---- Helper Functions ----

function [gx, gy] = grad_forward(u)
gx = circshift(u, [0, -1]) - u;
gy = circshift(u, [-1, 0]) - u;
end

function w = isotropic_shrink_2d(dux, duy, tau)
mag = sqrt(dux.^2 + duy.^2) + eps;
scale = max(0, 1 - tau ./ mag);
w = cat(3, scale .* dux, scale .* duy);
end

function v = compute_psnr(x, y)
mse = mean((x(:) - y(:)).^2);
if mse <= 0, v = Inf; else, v = 10 * log10(1 / mse); end
end
