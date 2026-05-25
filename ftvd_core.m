function result = ftvd_core(f, kernel, mu, beta0, beta_max, opts)
%FTVD_CORE  Fast Total Variation Deconvolution (baseline).
%
%   Implements Algorithm 1 & 2 from:
%   Wang, Yang, Yin, Zhang, "A New Alternating Minimization Algorithm for
%   Total Variation Image Reconstruction", SIAM J. Imaging Sci., 2008.
%
%   Solves: min_u { sum_i ||D_i u||_2 + (mu/2)||Ku - f||^2 }
%
%   via alternating minimization with beta-continuation:
%
%   For fixed beta:
%     1. w_i = shrink_2D(D_i u, 1/beta)      (isotropic 2D shrinkage)
%     2. u = FFT_solve(w, mu/beta)           (eq. 2.4)
%
%   Usage:
%     result = ftvd_core(f, kernel, mu, beta0, beta_max, opts)
%
%   Input:
%     f        - observed image [m x n], double in [0,1]
%     kernel   - blur kernel (1 for denoising only)
%     mu       - data-fidelity regularization parameter
%     beta0    - initial penalty parameter
%     beta_max - final penalty parameter
%     opts     - options struct with fields (all optional):
%         tol         - stopping tolerance (default: 1e-4)
%         max_iter    - max total iterations (default: 1000)
%         inner_iter  - max inner iterations per beta (default: 100)
%         verbose     - print progress (default: false)
%         gt          - ground truth for PSNR (default: [])
%
%   Output:
%     result - struct with fields: u, psnr, history, iters

% --- Set defaults ---
if nargin < 2 || isempty(kernel), kernel = 1; end
if nargin < 3 || isempty(mu), mu = 350; end
if nargin < 4 || isempty(beta0), beta0 = 1; end
if nargin < 5 || isempty(beta_max), beta_max = 128; end
if nargin < 6, opts = struct(); end

if ~isfield(opts, 'tol'), opts.tol = 1e-4; end
if ~isfield(opts, 'max_iter'), opts.max_iter = 1000; end
if ~isfield(opts, 'inner_iter'), opts.inner_iter = 100; end
if ~isfield(opts, 'verbose'), opts.verbose = false; end
if ~isfield(opts, 'gt'), opts.gt = []; end
if ~isfield(opts, 'iter_callback'), opts.iter_callback = []; end
if ~isfield(opts, 'track_relerr'), opts.track_relerr = false; end
if ~isfield(opts, 'stop_by_rel_change'), opts.stop_by_rel_change = true; end
if ~isfield(opts, 'stop_by_relerr'), opts.stop_by_relerr = false; end
if ~isfield(opts, 'tol_relerr'), opts.tol_relerr = 1e-3; end

f = im2double(f);

% Precompute FFT operators
F = precompute_fft_operators(f, kernel, 0);

% Initialization
u = f;
[dux, duy] = grad_forward(u);
w = cat(3, dux, duy);

% History tracking
max_hist = min(opts.max_iter, 10000);
hist_rel = zeros(max_hist, 1);
hist_beta = zeros(max_hist, 1);
hist_relerr = zeros(max_hist, 1);

beta = beta0;
k_total = 0;

while beta <= beta_max && k_total < opts.max_iter
    mu_over_beta = mu / beta;

    for k_inner = 1:opts.inner_iter
        if k_total >= opts.max_iter, break; end

        u_old = u;

        % ---- Step 1: w-update (isotropic 2D shrinkage) ----
        [dux, duy] = grad_forward(u);
        w = isotropic_shrink_2d(dux, duy, 1 / beta);

        % ---- Step 2: u-update (FFT) ----
        u = F.update(w, mu_over_beta);

        % ---- Convergence check ----
        rel = norm(u(:) - u_old(:)) / max(norm(u_old(:)), eps);
        k_total = k_total + 1;

        if k_total <= max_hist
        hist_rel(k_total) = rel;
        hist_beta(k_total) = beta;
        if opts.track_relerr && ~isempty(opts.gt)
            hist_relerr(k_total) = norm(u(:) - opts.gt(:)) / max(norm(opts.gt(:)), eps);
        end
        end

        if opts.verbose && mod(k_total, 10) == 0
            fprintf('  [FTVd] iter=%4d  beta=%8.2f  rel=%.2e\n', ...
                k_total, beta, rel);
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

    beta = beta * 2;
end

% Trim history
hist_rel = hist_rel(1:k_total);
hist_beta = hist_beta(1:k_total);
hist_relerr = hist_relerr(1:k_total);

result.u = u;
result.u = min(max(result.u, 0), 1);
result.iters = k_total;
result.history.rel_change = hist_rel;
result.history.beta = hist_beta;
if opts.track_relerr && ~isempty(opts.gt)
    result.history.rel_err_gt = hist_relerr(1:k_total);
end

if ~isempty(opts.gt)
    result.psnr = compute_psnr(u, opts.gt);
end
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
if mse <= 0
    v = Inf;
else
    v = 10 * log10(1 / mse);
end
end
