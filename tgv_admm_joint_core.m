function result = tgv_admm_joint_core(f, kernel, mu, alpha0, alpha1, beta1, beta2, opts)
%TGV_ADMM_JOINT_CORE  TGV ADMM with joint (u,v) quadratic solve.
%
% This version solves the ADMM x-block x=(u,v) jointly by PCG, matching the
% two-block ADMM convergence statement more closely than one u-step plus one
% v-step.

if nargin < 2 || isempty(kernel), kernel = 1; end
if nargin < 3 || isempty(mu), mu = 350; end
if nargin < 4 || isempty(alpha0), alpha0 = 0.5; end
if nargin < 5 || isempty(alpha1), alpha1 = 1.0; end
if nargin < 6 || isempty(beta1), beta1 = 2.0; end
if nargin < 7 || isempty(beta2), beta2 = 2.0; end
if nargin < 8, opts = struct(); end

if ~isfield(opts, 'max_iter'), opts.max_iter = 200; end
if ~isfield(opts, 'tol'), opts.tol = 1e-4; end
if ~isfield(opts, 'verbose'), opts.verbose = false; end
if ~isfield(opts, 'gt'), opts.gt = []; end
if ~isfield(opts, 'pcg_tol'), opts.pcg_tol = 1e-6; end
if ~isfield(opts, 'pcg_maxit'), opts.pcg_maxit = 200; end
if ~isfield(opts, 'iter_callback'), opts.iter_callback = []; end
if ~isfield(opts, 'track_relerr'), opts.track_relerr = false; end
if ~isfield(opts, 'stop_by_rel_change'), opts.stop_by_rel_change = true; end
if ~isfield(opts, 'stop_by_relerr'), opts.stop_by_relerr = false; end
if ~isfield(opts, 'tol_relerr'), opts.tol_relerr = 1e-3; end

f = im2double(f);
[m, n] = size(f);
N = m * n;
F = local_fft_setup(f, kernel);

u = f;
v = zeros(m, n, 2);
w1 = zeros(m, n, 2);
w2 = zeros(m, n, 4);
l1 = zeros(m, n, 2);
l2 = zeros(m, n, 4);

hist = zeros(opts.max_iter, 1);
hist_p1 = zeros(opts.max_iter, 1);
hist_p2 = zeros(opts.max_iter, 1);
hist_pcg_flag = zeros(opts.max_iter, 1);
hist_pcg_relres = zeros(opts.max_iter, 1);
hist_pcg_iter = zeros(opts.max_iter, 1);
hist_relerr = zeros(opts.max_iter, 1);

for k = 1:opts.max_iter
    u_old = u;

    a1 = w1 - l1;
    a2 = w2 - l2;
    rhs_u = mu * F.Ktf + beta1 * div_op(a1);
    rhs_v = -beta1 * a1 + beta2 * Et_op(a2);
    b = pack_uv(rhs_u, rhs_v);
    x0 = pack_uv(u, v);

    Afun = @(x) joint_A(x, m, n, F, mu, beta1, beta2);
    [x, flag, relres, pcg_iter] = pcg(Afun, b, opts.pcg_tol, opts.pcg_maxit, [], [], x0);
    if isempty(x)
        x = x0;
        flag = -1;
        relres = NaN;
        pcg_iter = 0;
    end
    [u, v] = unpack_uv(x, m, n);

    Du = grad_op(u);
    Ev = E_op(v);

    w1 = shrink_vec(Du - v + l1, alpha1 / beta1);
    w2 = shrink_vec(Ev + l2, alpha0 / beta2);

    l1 = l1 + (Du - v - w1);
    l2 = l2 + (Ev - w2);

    r1 = Du - v - w1;
    r2 = Ev - w2;
    rel = norm(u(:) - u_old(:)) / max(norm(u_old(:)), eps);
    hist(k) = rel;
    hist_p1(k) = norm(r1(:));
    hist_p2(k) = norm(r2(:));
    hist_pcg_flag(k) = flag;
    hist_pcg_relres(k) = relres;
    hist_pcg_iter(k) = pcg_iter;
    if opts.track_relerr && ~isempty(opts.gt)
        hist_relerr(k) = norm(u(:) - opts.gt(:)) / max(norm(opts.gt(:)), eps);
    end

    if opts.verbose && (mod(k, 10) == 0 || k == 1)
        fprintf('[TGV-ADMM-Joint] iter=%4d rel=%.2e pcg_flag=%d pcg_rel=%.2e\n', ...
            k, rel, flag, relres);
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
result.history.primal_r1 = hist_p1(1:result.iters);
result.history.primal_r2 = hist_p2(1:result.iters);
result.history.pcg_flag = hist_pcg_flag(1:result.iters);
result.history.pcg_relres = hist_pcg_relres(1:result.iters);
result.history.pcg_iter = hist_pcg_iter(1:result.iters);
if opts.track_relerr && ~isempty(opts.gt)
    result.history.rel_err_gt = hist_relerr(1:result.iters);
end
if ~isempty(opts.gt), result.psnr = local_psnr(result.u, opts.gt); end
end

function y = joint_A(x, m, n, F, mu, beta1, beta2)
[u, v] = unpack_uv(x, m, n);
Du_minus_v = grad_op(u) - v;
Au = mu * real(ifft2(conj(F.FK) .* (F.FK .* fft2(u)))) + beta1 * div_op(Du_minus_v);
Av = -beta1 * Du_minus_v + beta2 * Et_op(E_op(v));
y = pack_uv(Au, Av);
end

function x = pack_uv(u, v)
vx = v(:,:,1);
vy = v(:,:,2);
x = [u(:); vx(:); vy(:)];
end

function [u, v] = unpack_uv(x, m, n)
N = m * n;
u = reshape(x(1:N), m, n);
v = zeros(m, n, 2);
v(:,:,1) = reshape(x(N+1:2*N), m, n);
v(:,:,2) = reshape(x(2*N+1:3*N), m, n);
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

function y = shrink_vec(x, tau)
mag = sqrt(sum(x.^2, 3)) + eps;
scale = max(0, 1 - tau ./ mag);
y = x .* scale;
end

function v = local_psnr(x, y)
mse = mean((x(:) - y(:)).^2);
if mse <= 0, v = Inf; else, v = 10 * log10(1 / mse); end
end
