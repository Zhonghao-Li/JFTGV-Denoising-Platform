function result = admm_sat_core(f, kernel, mu, alpha, beta, opts)
%ADMM_SAT_CORE  ADMM with closed-form S_tau(z) denoiser in TV split model.
%
%   Model:
%     min_u (mu/2)||Ku-f||^2 + alpha*TV(u)
%   Split:
%     w = Du, use ADMM and replace w-prox with S_tau applied channel-wise.

if nargin < 2 || isempty(kernel), kernel = 1; end
if nargin < 3 || isempty(mu), mu = 350; end
if nargin < 4 || isempty(alpha), alpha = 1.0; end
if nargin < 5 || isempty(beta), beta = 2.0; end
if nargin < 6, opts = struct(); end

if ~isfield(opts, 'max_iter'), opts.max_iter = 200; end
if ~isfield(opts, 'tol'), opts.tol = 1e-4; end
if ~isfield(opts, 'verbose'), opts.verbose = false; end
if ~isfield(opts, 'gt'), opts.gt = []; end
if ~isfield(opts, 'st_mode'), opts.st_mode = 'iso'; end
if ~isfield(opts, 'beta_mult'), opts.beta_mult = 1.0; end
if ~isfield(opts, 'beta_max'), opts.beta_max = 1024; end
if ~isfield(opts, 'iter_callback'), opts.iter_callback = []; end
if ~isfield(opts, 'track_relerr'), opts.track_relerr = false; end
if ~isfield(opts, 'stop_by_rel_change'), opts.stop_by_rel_change = true; end
if ~isfield(opts, 'stop_by_relerr'), opts.stop_by_relerr = false; end
if ~isfield(opts, 'tol_relerr'), opts.tol_relerr = 1e-3; end

f = im2double(f);
F = precompute_fft_operators(f, kernel, 0);

u = f;
[gx, gy] = grad_forward(u);
w = cat(3, gx, gy);
l = zeros(size(w));

hist = zeros(opts.max_iter, 1);
hist_relerr = zeros(opts.max_iter, 1);
b = beta;

for k = 1:opts.max_iter
    u_old = u;

    % u-step
    rhs = w - l;
    u = F.update(rhs, mu / b);

    % w-step with S_tau per channel
    [gx, gy] = grad_forward(u);
    qx = gx + l(:,:,1);
    qy = gy + l(:,:,2);
    tau = alpha / b;
    wx = sat_tv_prox(qx, tau, opts.st_mode);
    wy = sat_tv_prox(qy, tau, opts.st_mode);
    w = cat(3, wx, wy);

    % dual update
    l(:,:,1) = l(:,:,1) + gx - w(:,:,1);
    l(:,:,2) = l(:,:,2) + gy - w(:,:,2);

    rel = norm(u(:) - u_old(:)) / max(norm(u_old(:)), eps);
    hist(k) = rel;
    if opts.track_relerr && ~isempty(opts.gt)
        hist_relerr(k) = norm(u(:) - opts.gt(:)) / max(norm(opts.gt(:)), eps);
    end
    if opts.verbose && (mod(k, 10) == 0 || k == 1)
        fprintf('[ADMM-S_tau] iter=%4d beta=%.2f rel=%.2e\n', k, b, rel);
    end
    if ~isempty(opts.iter_callback)
        opts.iter_callback(k, rel, b);
    end
    cond_rel = (~opts.stop_by_rel_change) || (rel < opts.tol);
    cond_relerr = true;
    if opts.stop_by_relerr
        if opts.track_relerr && ~isempty(opts.gt)
            cond_relerr = (hist_relerr(k) < opts.tol_relerr);
        else
            cond_relerr = false;
        end
    end
    if cond_rel && cond_relerr
        hist = hist(1:k);
        break;
    end

    if opts.beta_mult > 1
        b = min(opts.beta_max, b * opts.beta_mult);
    end
end

result.u = min(max(u, 0), 1);
result.iters = numel(hist);
result.history.rel_change = hist;
if opts.track_relerr && ~isempty(opts.gt)
    result.history.rel_err_gt = hist_relerr(1:result.iters);
end
result.beta_final = b;
if ~isempty(opts.gt), result.psnr = compute_psnr(result.u, opts.gt); end
end

function [gx, gy] = grad_forward(u)
gx = circshift(u, [0, -1]) - u;
gy = circshift(u, [-1, 0]) - u;
end

function v = compute_psnr(x, y)
mse = mean((x(:) - y(:)).^2);
if mse <= 0, v = Inf; else, v = 10 * log10(1 / mse); end
end
