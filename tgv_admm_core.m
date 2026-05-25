function result = tgv_admm_core(f, kernel, mu, alpha0, alpha1, beta1, beta2, opts)
%TGV_ADMM_CORE  Second-order TGV restoration via scaled ADMM.
%
%   Solves:
%     min_u  (mu/2)||K u - f||_2^2 + alpha1||Du - v||_{1,2} + alpha0||E v||_{1,2}
%
%   with variables v, w1=Du-v, w2=Ev and scaled dual variables.

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
if ~isfield(opts, 'pcg_tol'), opts.pcg_tol = 1e-5; end
if ~isfield(opts, 'pcg_maxit'), opts.pcg_maxit = 80; end
if ~isfield(opts, 'iter_callback'), opts.iter_callback = []; end
if ~isfield(opts, 'mode'), opts.mode = 'exact'; end
if ~isfield(opts, 'st_mode'), opts.st_mode = 'iso'; end
if ~isfield(opts, 'pnp_w2_mode'), opts.pnp_w2_mode = 'sym3'; end
if ~isfield(opts, 'track_relerr'), opts.track_relerr = false; end
if ~isfield(opts, 'stop_by_rel_change'), opts.stop_by_rel_change = true; end
if ~isfield(opts, 'stop_by_relerr'), opts.stop_by_relerr = false; end
if ~isfield(opts, 'tol_relerr'), opts.tol_relerr = 1e-3; end

f = im2double(f);
[m, n] = size(f);

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
hist_obj = zeros(opts.max_iter, 1);
hist_relerr = zeros(opts.max_iter, 1);

for k = 1:opts.max_iter
    u_old = u;

    % u-step: (mu K'K + beta1 D'D)u = mu K'f + beta1 D' (v + w1 - l1)
    rhs_u = mu * F.Ktf + beta1 * div_op(v + w1 - l1);
    u = solve_u_fft(rhs_u, F, mu, beta1);

    % v-step: (beta1 I + beta2 E'E) v = beta1(Du - w1 + l1) + beta2 E'(w2-l2)
    Du = grad_op(u);
    rhs_v = beta1 * (Du - w1 + l1) + beta2 * Et_op(w2 - l2);
    v = solve_v_pcg(v, rhs_v, beta1, beta2, opts.pcg_tol, opts.pcg_maxit);

    % w1-step
    Du = grad_op(u);
    z1 = Du - v + l1;
    if strcmpi(opts.mode, 'pnp_sat')
        w1 = sat_tv_prox_multich(z1, alpha1 / beta1, opts.st_mode);
    else
        w1 = shrink_vec(z1, alpha1 / beta1);
    end

    % w2-step
    Ev = E_op(v);
    z2 = Ev + l2;
    if strcmpi(opts.mode, 'pnp_sat')
        z2sym = cat(3, z2(:,:,1), 0.5 * (z2(:,:,2) + z2(:,:,3)), z2(:,:,4));
        w2sym = sat_tv_prox_multich(z2sym, alpha0 / beta2, opts.st_mode);
        w2 = cat(3, w2sym(:,:,1), w2sym(:,:,2), w2sym(:,:,2), w2sym(:,:,3));
    else
        w2 = shrink_vec(z2, alpha0 / beta2);
    end

    % dual update
    l1 = l1 + (Du - v - w1);
    l2 = l2 + (Ev - w2);

    rel = norm(u(:) - u_old(:)) / max(norm(u_old(:)), eps);
    r1 = Du - v - w1;
    r2 = Ev - w2;
    hist_p1(k) = norm(r1(:));
    hist_p2(k) = norm(r2(:));
    Ku = real(ifft2(F.FK .* fft2(u)));
    tv1 = sum(sqrt(sum((Du - v).^2, 3) + eps), 'all');
    tv2 = sum(sqrt(sum((Ev).^2, 3) + eps), 'all');
    hist_obj(k) = 0.5 * mu * sum((Ku(:) - f(:)).^2) + alpha1 * tv1 + alpha0 * tv2;
    hist(k) = rel;
    if opts.track_relerr && ~isempty(opts.gt)
        hist_relerr(k) = norm(u(:) - opts.gt(:)) / max(norm(opts.gt(:)), eps);
    end

    if opts.verbose && (mod(k, 10) == 0 || k == 1)
        fprintf('[TGV-ADMM] iter=%4d rel=%.2e\n', k, rel);
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
result.history.obj = hist_obj(1:result.iters);
if opts.track_relerr && ~isempty(opts.gt)
    result.history.rel_err_gt = hist_relerr(1:result.iters);
end
if ~isempty(opts.gt)
    result.psnr = local_psnr(result.u, opts.gt);
end
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

Dx = zeros(m, n); Dx(1,1) = 1; Dx(1,2) = -1;
Dy = zeros(m, n); Dy(1,1) = 1; Dy(2,1) = -1;
FDx = fft2(circshift(Dx, [0, 1]));
FDy = fft2(circshift(Dy, [1, 0]));
F.DDt = abs(FDx).^2 + abs(FDy).^2;
end

function u = solve_u_fft(rhs, F, mu, beta1)
Frhs = fft2(rhs);
den = mu * abs(F.FK).^2 + beta1 * F.DDt + eps;
u = real(ifft2(Frhs ./ den));
end

function v = solve_v_pcg(v0, rhs, beta1, beta2, tol, maxit)
[m, n, ~] = size(rhs);
bx = rhs(:,:,1); by = rhs(:,:,2);
b = [bx(:); by(:)];

Af = @(x) apply_Av(x, m, n, beta1, beta2);
v0x = v0(:,:,1);
v0y = v0(:,:,2);
x0 = [v0x(:); v0y(:)];
[x, ~] = pcg(Af, b, tol, maxit, [], [], x0);
if isempty(x), x = x0; end
v = zeros(m, n, 2);
v(:,:,1) = reshape(x(1:m*n), m, n);
v(:,:,2) = reshape(x(m*n+1:end), m, n);
end

function y = apply_Av(x, m, n, beta1, beta2)
vx = reshape(x(1:m*n), m, n);
vy = reshape(x(m*n+1:end), m, n);
v = cat(3, vx, vy);
z = beta1 * v + beta2 * Et_op(E_op(v));
zx = z(:,:,1);
zy = z(:,:,2);
y = [zx(:); zy(:)];
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

function y = sat_tv_prox_multich(z, tau, mode)
% Coupled S_tau for multi-channel fields:
% one shared shrink scale per pixel over all channel high-pass coefficients.
[m, n, c] = size(z);
d = 2;
w_scale = 1 / sqrt(2 * d);
haar = 1 / sqrt(2);
thr = tau * sqrt(2 * d);

Mh = zeros(m, n, c); Mv = zeros(m, n, c);
Dh = zeros(m, n, c); Dv = zeros(m, n, c);
for k = 1:c
    zk = z(:,:,k);
    Mh(:,:,k) = w_scale * haar * (zk + circshift(zk, [0, -1]));
    Mv(:,:,k) = w_scale * haar * (zk + circshift(zk, [-1, 0]));
    Dh(:,:,k) = w_scale * haar * (zk - circshift(zk, [0, -1]));
    Dv(:,:,k) = w_scale * haar * (zk - circshift(zk, [-1, 0]));
end

switch lower(mode)
    case 'aniso'
        Dh = sign(Dh) .* max(abs(Dh) - thr, 0);
        Dv = sign(Dv) .* max(abs(Dv) - thr, 0);
    otherwise
        mag = sqrt(sum(Dh.^2 + Dv.^2, 3)) + eps;
        scl = max(0, 1 - thr ./ mag);
        for k = 1:c
            Dh(:,:,k) = scl .* Dh(:,:,k);
            Dv(:,:,k) = scl .* Dv(:,:,k);
        end
end

y = zeros(m, n, c);
for k = 1:c
    Mh_adj = haar * (Mh(:,:,k) + circshift(Mh(:,:,k), [0, 1]));
    Mv_adj = haar * (Mv(:,:,k) + circshift(Mv(:,:,k), [1, 0]));
    Dh_adj = haar * (Dh(:,:,k) - circshift(Dh(:,:,k), [0, 1]));
    Dv_adj = haar * (Dv(:,:,k) - circshift(Dv(:,:,k), [1, 0]));
    y(:,:,k) = w_scale * (Mh_adj + Mv_adj + Dh_adj + Dv_adj);
end
end

function v = local_psnr(x, y)
mse = mean((x(:) - y(:)).^2);
if mse <= 0
    v = Inf;
else
    v = 10 * log10(1 / mse);
end
end
