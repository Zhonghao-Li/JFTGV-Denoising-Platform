function result = ftvd_alg1_nobeta_core(f, kernel, mu, opts)
%FTVD_ALG1_NOBETA_CORE  FTVd Algorithm 1 (fixed-penalty alternating minimization).
%
%   This implementation follows the Algorithm-1 spirit in
%   "A New Alternating Minimization Algorithm for Total Variation Image Reconstruction"
%   but does NOT use beta continuation (Algorithm-2). A fixed internal penalty is used.
%
%   Model:
%     min_u  TV(u) + (mu/2)||Ku-f||^2
%
%   Split form for fixed beta:
%     min_{u,w}  sum_i ||w_i||_2 + (beta/2)||w-Du||^2 + (mu/2)||Ku-f||^2
%
%   Inputs:
%     f      - observed image in [0,1]
%     kernel - blur kernel (1 for denoising)
%     mu     - data-fidelity parameter
%     opts   - struct fields (optional):
%              .tol, .max_iter, .inner_iter, .verbose, .gt, .iter_callback
%              .track_relerr, .stop_by_rel_change, .stop_by_relerr, .tol_relerr
%              .beta_fixed   (internal fixed penalty; default 2)

if nargin < 2 || isempty(kernel), kernel = 1; end
if nargin < 3 || isempty(mu), mu = 8; end
if nargin < 4, opts = struct(); end

if ~isfield(opts, 'tol'), opts.tol = 1e-6; end
if ~isfield(opts, 'max_iter'), opts.max_iter = 1000; end
if ~isfield(opts, 'inner_iter'), opts.inner_iter = 1000; end
if ~isfield(opts, 'verbose'), opts.verbose = false; end
if ~isfield(opts, 'gt'), opts.gt = []; end
if ~isfield(opts, 'iter_callback'), opts.iter_callback = []; end
if ~isfield(opts, 'track_relerr'), opts.track_relerr = false; end
if ~isfield(opts, 'stop_by_rel_change'), opts.stop_by_rel_change = true; end
if ~isfield(opts, 'stop_by_relerr'), opts.stop_by_relerr = false; end
if ~isfield(opts, 'tol_relerr'), opts.tol_relerr = 1e-3; end
if ~isfield(opts, 'beta_fixed'), opts.beta_fixed = 2.0; end

f = im2double(f);
mu = max(mu, 1e-12);
beta = max(opts.beta_fixed, 1e-8);

F = precompute_fft_operators(f, kernel, 0);

u = f;
[dux, duy] = grad_forward(u);
w = cat(3, dux, duy);

max_hist = min(opts.max_iter, 10000);
hist_rel = zeros(max_hist, 1);
hist_relerr = zeros(max_hist, 1);

k_total = 0;
mu_over_beta = mu / beta;
max_loop = min(opts.inner_iter, opts.max_iter);
for k = 1:max_loop
    u_old = u;

    % w-update (isotropic shrinkage)
    [dux, duy] = grad_forward(u);
    w = isotropic_shrink_2d(dux, duy, 1 / beta);

    % u-update (FFT closed form)
    u = F.update(w, mu_over_beta);

    rel = norm(u(:) - u_old(:)) / max(norm(u_old(:)), eps);
    k_total = k_total + 1;
    if k_total <= max_hist
        hist_rel(k_total) = rel;
        if opts.track_relerr && ~isempty(opts.gt)
            hist_relerr(k_total) = norm(u(:) - opts.gt(:)) / max(norm(opts.gt(:)), eps);
        end
    end

    if opts.verbose && mod(k_total, 20) == 0
        fprintf('[FTVd-Alg1] iter=%4d rel=%.2e\n', k_total, rel);
    end

    if ~isempty(opts.iter_callback)
        opts.iter_callback(k_total, rel, beta);
    end

    cond_rel = (~opts.stop_by_rel_change) || (rel < opts.tol);
    cond_relerr = true;
    if opts.stop_by_relerr
        if opts.track_relerr && ~isempty(opts.gt)
            cond_relerr = (hist_relerr(k_total) < opts.tol_relerr);
        else
            cond_relerr = false;
        end
    end
    if cond_rel && cond_relerr
        break;
    end
end

hist_rel = hist_rel(1:k_total);
hist_relerr = hist_relerr(1:k_total);

result.u = min(max(u, 0), 1);
result.iters = k_total;
result.history.rel_change = hist_rel;
result.history.beta = beta * ones(k_total, 1);
if opts.track_relerr && ~isempty(opts.gt)
    result.history.rel_err_gt = hist_relerr;
end
if ~isempty(opts.gt)
    result.psnr = compute_psnr(result.u, opts.gt);
end
end

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
if mse <= 0
    v = Inf;
else
    v = 10 * log10(1 / mse);
end
end
