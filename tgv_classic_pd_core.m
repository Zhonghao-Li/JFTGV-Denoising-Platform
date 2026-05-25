function result = tgv_classic_pd_core(f, kernel, mu, alpha0, alpha1, opts)
%TGV_CLASSIC_PD_CORE  Classical primal-dual (Chambolle-Pock style) solver for TGV^2.
%
%   min_{u,v} (mu/2)||Ku-f||_2^2 + alpha1||Du-v||_{1,2} + alpha0||Ev||_{1,2}

if nargin < 2 || isempty(kernel), kernel = 1; end
if nargin < 3 || isempty(mu), mu = 350; end
if nargin < 4 || isempty(alpha0), alpha0 = 0.5; end
if nargin < 5 || isempty(alpha1), alpha1 = 1.0; end
if nargin < 6, opts = struct(); end

if ~isfield(opts, 'max_iter'), opts.max_iter = 200; end
if ~isfield(opts, 'tol'), opts.tol = 1e-4; end
if ~isfield(opts, 'verbose'), opts.verbose = false; end
if ~isfield(opts, 'gt'), opts.gt = []; end
if ~isfield(opts, 'iter_callback'), opts.iter_callback = []; end
if ~isfield(opts, 'tau'), opts.tau = 0.2; end
if ~isfield(opts, 'sigma'), opts.sigma = 0.2; end
if ~isfield(opts, 'theta'), opts.theta = 1.0; end
if ~isfield(opts, 'track_relerr'), opts.track_relerr = false; end
if ~isfield(opts, 'stop_by_rel_change'), opts.stop_by_rel_change = true; end
if ~isfield(opts, 'stop_by_relerr'), opts.stop_by_relerr = false; end
if ~isfield(opts, 'tol_relerr'), opts.tol_relerr = 1e-3; end

f = im2double(f);
[m, n] = size(f);
F = local_fft_setup(f, kernel);

u = f;
v = zeros(m, n, 2);
p = zeros(m, n, 2);   % dual for Du-v
q = zeros(m, n, 4);   % dual for Ev
ubar = u;
vbar = v;

hist = zeros(opts.max_iter, 1);
hist_relerr = zeros(opts.max_iter, 1);

tau = opts.tau; sigma = opts.sigma; theta = opts.theta;

for k = 1:opts.max_iter
    u_old = u;
    v_old = v;

    % dual ascent + projection
    p = p + sigma * (grad_op(ubar) - vbar);
    p = proj_l2_ball(p, alpha1);

    q = q + sigma * E_op(vbar);
    q = proj_l2_ball(q, alpha0);

    % primal updates
    rhs_u = mu * F.Ktf + (1 / tau) * u_old - div_op(p);
    u = solve_u_fft(rhs_u, F, mu, 1 / tau);
    u = min(max(u, 0), 1);

    v = v_old + tau * (p - Et_op(q));

    % extrapolation
    ubar = u + theta * (u - u_old);
    vbar = v + theta * (v - v_old);

    rel = norm(u(:) - u_old(:)) / max(norm(u_old(:)), eps);
    hist(k) = rel;
    if opts.track_relerr && ~isempty(opts.gt)
        hist_relerr(k) = norm(u(:) - opts.gt(:)) / max(norm(opts.gt(:)), eps);
    end

    if opts.verbose && (mod(k, 10) == 0 || k == 1)
        fprintf('[TGV-Classic-PD] iter=%4d rel=%.2e\n', k, rel);
    end
    if ~isempty(opts.iter_callback)
        opts.iter_callback(k, rel, NaN);
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
end

result.u = min(max(u, 0), 1);
result.iters = numel(hist);
result.history.rel_change = hist;
if opts.track_relerr && ~isempty(opts.gt)
    result.history.rel_err_gt = hist_relerr(1:result.iters);
end
if ~isempty(opts.gt), result.psnr = local_psnr(result.u, opts.gt); end
end

function F = local_fft_setup(f, kernel)
[m, n] = size(f);
if isscalar(kernel) && kernel == 1
    FK = ones(m, n);
else
    K_pad = zeros(m, n);
    [km, kn] = size(kernel);
    K_pad(1:km, 1:kn) = kernel;
    K_pad = circshift(K_pad, -floor([km, kn] / 2));
    FK = fft2(K_pad);
end
F.FK = FK;
F.Ktf = real(ifft2(conj(FK) .* fft2(f)));
end

function u = solve_u_fft(rhs, F, mu, c)
Frhs = fft2(rhs);
den = mu * abs(F.FK).^2 + c + eps;
u = real(ifft2(Frhs ./ den));
end

function g = grad_op(u)
g = zeros(size(u,1), size(u,2), 2);
g(:,:,1) = circshift(u, [0, -1]) - u;
g(:,:,2) = circshift(u, [-1, 0]) - u;
end

function d = div_op(p)
px = p(:,:,1); py = p(:,:,2);
d = circshift(px, [0, 1]) - px + circshift(py, [1, 0]) - py;
end

function e = E_op(v)
vx = v(:,:,1); vy = v(:,:,2);
dx_vx = circshift(vx, [0, -1]) - vx;
dy_vy = circshift(vy, [-1, 0]) - vy;
dy_vx = circshift(vx, [-1, 0]) - vx;
dx_vy = circshift(vy, [0, -1]) - vy;
sym12 = 0.5 * (dy_vx + dx_vy);
e = zeros(size(v,1), size(v,2), 4);
e(:,:,1) = dx_vx;
e(:,:,2) = sym12;
e(:,:,3) = sym12;
e(:,:,4) = dy_vy;
end

function v = Et_op(p)
p1 = p(:,:,1); p2 = p(:,:,2); p3 = p(:,:,3); p4 = p(:,:,4);
DxT_p1 = circshift(p1, [0, 1]) - p1;
DxT_p2 = circshift(p2, [0, 1]) - p2;
DxT_p3 = circshift(p3, [0, 1]) - p3;
DyT_p2 = circshift(p2, [1, 0]) - p2;
DyT_p3 = circshift(p3, [1, 0]) - p3;
DyT_p4 = circshift(p4, [1, 0]) - p4;
v = zeros(size(p,1), size(p,2), 2);
v(:,:,1) = DxT_p1 + 0.5 * (DyT_p2 + DyT_p3);
v(:,:,2) = DyT_p4 + 0.5 * (DxT_p2 + DxT_p3);
end

function y = proj_l2_ball(x, alpha)
mag = sqrt(sum(x.^2, 3)) + eps;
scale = min(1, alpha ./ mag);
y = x .* scale;
end

function v = local_psnr(x, y)
mse = mean((x(:) - y(:)).^2);
if mse <= 0, v = Inf; else, v = 10 * log10(1 / mse); end
end
