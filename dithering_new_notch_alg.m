
%  dithering_refactored.m
%  Reflected-seismic-pulse / dithering simulation (refactored).
%
%  Structure (top to bottom):
%    1. PARAMETERS  - every configurable number lives here, no globals
%    2. SETUP       - build x, noise, dither once
%    3. MAIN LOOP   - per impulse: analyse, report, plot
%    4. FUNCTIONS   - makeSums, analyzeSignal, analyzeDithered,
%                     reportSignal, plotStep (+ small helpers)
%
%  The three signal variants (unnoised / noised / dithered) are no longer
%  carried as *_unn / *_noi / *_dit suffixed variables. Each is one struct
%  produced by analyzeSignal, so a struct field like  unn.fzero_x  replaces
%  the old  fzero_min_X_Pol_unn,  and the giant multi-output signatures are
%  gone.
%
%  NOTE on preserved behaviours (see the chat notes for details):
%   (a) sumNoi adds `noise` to each pulse, so the noise term appears twice.
%   (b) With params.dithRegen = false (default, = your original), the dither
%       array is fixed, so all params.nDith realizations are identical and
%       the "averaging" returns that single value. Set dithRegen = true for
%       genuine per-realization (Monte-Carlo) dithering.
%   (c) All three variants now take derivatives of the *normalized* polynomial
%       (the original normalized the dithered poly only after differentiating,
%       putting its diff curves on a different vertical scale).
% =====================================================================

clear; clc; 

%% 1. PARAMETERS ------------------------------------------------------
params.dt           = 0.5;    % step of the moving pulse
params.fixedPos     = 5;      % position of the fixed pulse: sech(x - fixedPos)
params.fragMargin   = 1.5;    % fragment half-padding (x-units) added beyond
                              % each of the two detected peaks (replaces the
                              % old amplitude cut-off params.ampThreshold)
params.polyDeg      = 7;      % polynomial degree for all three fits
params.noiseAmp     = 0.20;   % noise amplitude (was a1)
params.dithAmp      = params.noiseAmp / 5;  % dither amplitude (was b1 = a1/5)
params.nDith        = 100;    % number of dithering realizations
params.impulses     = 1:8;      % which impulse steps to run (e.g. 1:8)

params.dithRegen    = true;  % false = original behaviour; true = real dithering
params.seed         = [];     % fixed RNG seed for reproducible runs; [] = random
params.saveResults  = false;  % true -> save results+params to a .mat file

% --- quantizer stochastic-resonance sweep (optional) ---
params.runResonanceSweep = true;               % run the SR sweep after the main loop
params.resonanceImpulse  = params.impulses(1); % which impulse to analyse
params.quantFrac         = 0.25;  % quantizer step Δ as a fraction of max(sumUnn);
                                  % LARGER = coarser quantizer = stronger resonance
params.srSigmaMax        = 1.5;   % sweep noise σ up to this multiple of Δ
params.srNumSigma        = 25;    % number of σ points in the sweep
params.srNumAvg          = 100;   % N: dithered copies averaged per position estimate
params.srNumRepeat       = 40;    % M: independent repeats per σ (for the statistics)
params.srNoiseSigma      = 0;     % your measurement-noise std, for the σ_d recommendation

%% 2. SETUP -----------------------------------------------------------
left_x = -5; right_x = 10; n_points = 2002;
x = linspace(left_x, right_x, n_points);

if ~isempty(params.seed)
    rng(params.seed);
end
% noise = params.noiseAmp * (rand(1, n_points) - 0.5);
noise1 = params.noiseAmp * (rand(1, n_points) - 0.5);
noise2 = params.noiseAmp * (rand(1, n_points) - 0.5);

% total noise std ≈ √2 · std(noise_single)
dith  = params.dithAmp  * (rand(1, n_points) - 0.5);

%% 3. MAIN LOOP -------------------------------------------------------
results = struct([]);
k = 0;
for i_nImp = params.impulses

    [sumUnn, sumNoi] = makeSums(x, i_nImp, params, noise1, noise2);

    % fragment mask: single source of truth, built from the clean sum.
    % Peak-bracket criterion (replaces the amplitude cut).
    mask = peakBracketMask(x, sumUnn, params);

    unn = analyzeSignal(x, sumUnn, mask, params.polyDeg);
    noi = analyzeSignal(x, sumNoi, mask, params.polyDeg);
    dit = analyzeDithered(x, sumNoi, dith, mask, params.polyDeg, ...
                          params.nDith, params.dithRegen, params.dithAmp);

    k = k + 1;
    results(k).i_nImp = i_nImp;
    results(k).unn    = unn;
    results(k).noi    = noi;
    results(k).dit    = dit;

    reportSignal(i_nImp, unn, noi, dit);
    plotStep(i_nImp, x, sumUnn, unn, noi, dit);
end

%% 3b. QUANTIZER STOCHASTIC-RESONANCE SWEEP (optional) ---------------
%  Demonstrates real stochastic resonance: noise added before a coarse
%  quantizer, averaged over realizations, recovers the sub-step (sub-LSB)
%  pulse position. RMSE vs noise should dip at a nonzero optimum σ*.
if params.runResonanceSweep
    iSR = params.resonanceImpulse;
    sumUnnSR = makeSums(x, iSR, params, noise1, noise2);
    maskSR   = peakBracketMask(x, sumUnnSR, params);

    sr = sweepResonance(x, sumUnnSR, maskSR, params);
    plotResonance(sr, iSR);

    [minRmse, iStar] = min(sr.rmse);
    sigmaStar = sr.sigma(iStar);
    sigma_d   = ditherForResonance(sigmaStar, params.srNoiseSigma);

    fprintf('\n--- Quantizer stochastic resonance (impulse %d) ---\n', iSR);
    fprintf(' quantizer step Δ        = %.6f\n', sr.Delta);
    fprintf(' clean (true) position   = %.6f\n', sr.x_true);
    fprintf(' RMSE at σ = 0 (no noise)= %.6f\n', sr.rmse(1));
    fprintf(' optimal noise σ*        = %.6f  (= %.3f·Δ)\n', sigmaStar, sigmaStar/sr.Delta);
    fprintf(' min RMSE at σ*          = %.6f\n', minRmse);
    fprintf(' recommended dither σ_d  = %.6f  (for measured σ_n = %.6f)\n', sigma_d, params.srNoiseSigma);
end

if params.saveResults
    save('dithering_results.mat', 'results', 'params');
end

fprintf('\n--Конец программы--\n');


% =====================================================================
%%  4. FUNCTIONS
% =====================================================================

% ---------------------------------------------------------------------
%  Build the moving + fixed pulses and their sums.
% ---------------------------------------------------------------------
function [sumUnn, sumNoi] = makeSums(x, i_nImp, params, noise1, noise2)
    imp_n        = sech(x - i_nImp * params.dt);   % moving pulse
    imp_constant = sech(x - params.fixedPos);      % fixed pulse

    sumUnn = imp_n + imp_constant;

    % Original adds `noise` to EACH pulse before summing, so the noise term
    % ends up in the sum twice (sumNoi = sumUnn + 2*noise). Preserved as-is.
    sumNoi = (imp_n + noise1) + (imp_constant + noise2);
end

% ---------------------------------------------------------------------
%  Fragment selector (replaces the amplitude cut).
%  The fragment is the span between the two dominant peaks of the clean
%  sum, padded by params.fragMargin on each side. If only one peak exists
%  (pulses merged) it returns a symmetric window around that peak, so the
%  downstream fit still has data but analyzeSignal will report no valley.
% ---------------------------------------------------------------------
function mask = peakBracketMask(x, sig, params)
    maxTF = islocalmax(sig, 'MaxNumExtrema', 2, 'SamplePoints', x);
    xPk   = sort(x(maxTF));
    if numel(xPk) >= 2
        lo = xPk(1)   - params.fragMargin;
        hi = xPk(end) + params.fragMargin;
    else
        if isempty(xPk), [~, im] = max(sig); xPk = x(im); end
        lo = xPk(1) - params.fragMargin;
        hi = xPk(1) + params.fragMargin;
    end
    mask = x >= lo & x <= hi;
end

% ---------------------------------------------------------------------
%  Analyse ONE signal: fragment, polynomial fit, derivatives, extrema,
%  and the exact derivative-zero (polynomial minimum). Returns a struct.
%  Replaces getPol + getDiffSumAndPol + get_Inflection_Points_And_Zeros
%  for a single signal.
% ---------------------------------------------------------------------
function s = analyzeSignal(x, sig, mask, polyDeg, opts)
    arguments
        x       (1,:) double
        sig     (1,:) double
        mask    (1,:) logical
        polyDeg (1,1) double
        opts.Normalize (1,1) logical = true
    end

    s.x_frag   = x(mask);
    s.sig_frag = sig(mask);

    ext = {'MaxNumExtrema', 1, 'SamplePoints', s.x_frag}; %  'ProminenceWindow', [3 4],

    % raw-fragment extrema (used for the 'мин Суммарного импульса' marker)
    sMinTF = islocalmin(s.sig_frag, ext{:});
    sMaxTF = islocalmax(s.sig_frag, ext{:});
    s.sig_x_min = s.x_frag(sMinTF);  s.sig_y_min = s.sig_frag(sMinTF);
    s.sig_x_max = s.x_frag(sMaxTF);  s.sig_y_max = s.sig_frag(sMaxTF);
    

    % polynomial fit on the fragment (mu = centring/scaling from polyfit)
    [s.p_coef, ~, s.mu] = polyfit(s.x_frag, s.sig_frag, polyDeg);
    polRaw = polyval(s.p_coef, s.x_frag, [], s.mu);
    s.polMaxRaw = max(polRaw);                 % scale kept for the continuous polFun
    if opts.Normalize
        pol = polRaw ./ s.polMaxRaw;
    else
        pol = polRaw;  s.polMaxRaw = 1;
    end
    s.pol = pol;

    % derivatives: numeric on the polynomial, plus the analytic polyder
    x_mid       = (s.x_frag(1:end-1) + s.x_frag(2:end)) / 2;
    s.diff1_pol = diff(s.pol)       ./ diff(s.x_frag);
    s.diff2_pol = diff(s.diff1_pol) ./ diff(x_mid);
    s.p_coef_d1 = polyder(s.p_coef);
    s.polyder   = polyval(s.p_coef_d1, s.x_frag, [], s.mu);

    % extrema of the polynomial: the two dominant maxima are the pulse peaks
    maxTF = islocalmax(s.pol, 'MaxNumExtrema', 2, 'SamplePoints', s.x_frag);
    s.x_max = s.x_frag(maxTF);  s.y_max = s.pol(maxTF);

    % ---- local minimum by bracketed minimisation between the two peaks ----
    % No amplitude cut. The valley is by definition the lowest point strictly
    % between the two peaks, so we minimise the continuous (normalised) poly on
    % the bracket [leftPeak, rightPeak]. fminbnd (Brent) is confined to the
    % bracket, so it can never run off to a fragment edge (Runge spikes), and a
    % missing second peak is reported as "no resolvable valley" (NaN) — that is
    % the resolution limit itself, not an error.
    polFun = @(xi) polyval(s.p_coef, xi, [], s.mu) ./ s.polMaxRaw;
    xPk = sort(s.x_max);
    if numel(xPk) >= 2
        [s.fzero_x, s.fzero_y] = fminbnd(polFun, xPk(1), xPk(end), ...
                                         optimset('TolX', 1e-6));
    else
        s.fzero_x = NaN;  s.fzero_y = NaN;   % single peak => pulses merged
    end

    % keep the old field names so report/plot/dither code is unchanged
    if isnan(s.fzero_x)
        s.x_min = [];          s.y_min = [];
    else
        s.x_min = s.fzero_x;   s.y_min = s.fzero_y;
    end
end

% ---------------------------------------------------------------------
%  Dithering: run analyzeSignal over nReal realizations and average the
%  scalar locations. Returns:
%    d.rep  - a representative single realization (curves for plotting)
%    d.mean - averaged locations (sig_frag, x_min, y_min, fzero_x, fzero_y)
%  Replaces Add_ditering.
% ---------------------------------------------------------------------
function d = analyzeDithered(x, sumBase, dith, mask, polyDeg, nReal, regen, dithAmp)
    fragAccum = [];
    fzx = []; fzy = [];
    xmin = []; ymin = [];
    repS = struct();

    for kk = 1:nReal
        if regen
            rng(kk*17);                              % new realization each loop
            d_k = dithAmp * (rand(size(x)) - 0.5);
        else
            d_k = dith;                              % original behaviour: fixed dither
        end

        s = analyzeSignal(x, sumBase + d_k, mask, polyDeg);

        fragAccum = [fragAccum; s.sig_frag];                       %#ok<AGROW>
        if ~isempty(s.x_min)
            xmin(end+1) = s.x_min;    ymin(end+1) = s.y_min;       %#ok<AGROW>
            fzx(end+1)  = s.fzero_x;  fzy(end+1)  = s.fzero_y;     %#ok<AGROW>
        end
        repS = s;   % keep the last realization as the representative one
    end

    d.rep           = repS;
    d.n             = numel(fzx);     % realizations that yielded a detected minimum
    d.mean.sig_frag = mean(fragAccum, 1);
    d.mean.x_min    = mean(xmin);
    d.mean.y_min    = mean(ymin);
    d.mean.fzero_x  = mean(fzx);

    % Spread across realizations (sample std). Meaningful once dithRegen = true;
    % with the fixed-dither default every realization is identical, so std = 0.
    d.std.x_min   = std(xmin);
    d.std.y_min   = std(ymin);
    d.std.fzero_x = std(fzx);
    d.std.fzero_y = std(fzy);

    % Y at the averaged zero, read off the representative normalised polynomial
    if ~isempty(repS) && isfield(repS, 'pol')
        d.mean.fzero_y = interp1(repS.x_frag, repS.pol, d.mean.fzero_x, 'makima');
    else
        d.mean.fzero_y = NaN;
    end
end

% ---------------------------------------------------------------------
%  Console report for one impulse (all the fprintf, moved out of compute).
% ---------------------------------------------------------------------
function reportSignal(i_nImp, unn, noi, dit)
    fprintf('\n            -- ИМПУЛЬС %d. --\n', i_nImp);

    if i_nImp <= 7
        fprintf('\n Sum без шума.   x max Sum = %.7f   y max Sum = %.7f', ...
                scalarOrNaN(unn.sig_x_max), scalarOrNaN(unn.sig_y_max));
    else
        fprintf('\n Sum без шума.   x min Sum = %.7f   y min Sum = %.7f', ...
                scalarOrNaN(unn.sig_x_min), scalarOrNaN(unn.sig_y_min));
    end

    fprintf('\n медиана x_frag %d = %.7f\n', i_nImp, median(unn.x_frag));

    % --- polynomial minima (x) ---
    fprintf('\n    Минимум Pol   x min Pol без шум %d = %.7f', i_nImp, scalarOrNaN(unn.x_min));
    fprintf('\n     fzero        x min Pol без шум %d = %.7f', i_nImp, unn.fzero_x);
    fprintf('\n    Минимум Pol   x min Pol  +  шум %d = %.7f', i_nImp, scalarOrNaN(noi.x_min));
    fprintf('\n     fzero        x min Pol  +  шум %d = %.7f', i_nImp, noi.fzero_x);
    fprintf('\n    Минимум Pol   x min Pol  +  диз %d = %.7f ± %.7f', i_nImp, dit.mean.x_min, dit.std.x_min);
    fprintf('\n     fzero        x min Pol  +  диз %d = %.7f ± %.7f  (n=%d)\n', i_nImp, dit.mean.fzero_x, dit.std.fzero_x, dit.n);

    % --- polynomial minima (y) ---
    fprintf('\n    Минимум Pol   y min Pol без шум %d = %.7f', i_nImp, scalarOrNaN(unn.y_min));
    fprintf('\n     fzero        y min Pol без шум %d = %.7f', i_nImp, unn.fzero_y);
    fprintf('\n    Минимум Pol   y min Pol  +  шум %d = %.7f', i_nImp, scalarOrNaN(noi.y_min));
    fprintf('\n     fzero        y min Pol  +  шум %d = %.7f', i_nImp, noi.fzero_y);
    fprintf('\n    Минимум Pol   y min Pol  +  диз %d = %.7f ± %.7f', i_nImp, dit.mean.y_min, dit.std.y_min);
    fprintf('\n     fzero        y min Pol  +  диз %d = %.7f ± %.7f\n', i_nImp, dit.mean.fzero_y, dit.std.fzero_y);

    % --- polynomial maxima ---
    fprintf('\n    Максимум Pol.   x max Pol без шум %d = %.7f', i_nImp, scalarOrNaN(unn.x_max));
    fprintf('\n    Максимум Pol.   y max Pol без шум %d = %.7f\n', i_nImp, scalarOrNaN(unn.y_max));
    fprintf('\n    Максимум Pol.   x max Pol + шум %d = %.7f', i_nImp, scalarOrNaN(noi.x_max));
    fprintf('\n    Максимум Pol.   y max Pol + шум %d = %.7f\n', i_nImp, scalarOrNaN(noi.y_max));
    fprintf('\n    Максимум Pol.   x max Pol + диз %d = %.7f', i_nImp, scalarOrNaN(dit.rep.x_max));
    fprintf('\n    Максимум Pol.   y max Pol + диз %d = %.7f\n', i_nImp, scalarOrNaN(dit.rep.y_max));
end

% ---------------------------------------------------------------------
%  All plotting for one impulse (moved out of the main loop).
% ---------------------------------------------------------------------
function plotStep(i_nImp, x, sumUnn, unn, noi, dit)
    cSky = '#5abbe8';  cSky2 = '#4d9bbf';
    cOrn = '#FF8C00';  cOrn2 = '#FFB347';
    cVio = '#431e61';
    xf = unn.x_frag;

    figure('Name', sprintf('Шаг %d ', i_nImp));
    title(sprintf(' Шаг %d ', i_nImp));
    hold on; grid on;

    yline(0, '-k', 'LineWidth', 2);

    if i_nImp < 7
        if ~isempty(unn.x_min),     xline(unn.x_min,     '--k', 'LineWidth', 1); end
        if ~isempty(noi.x_min),     xline(noi.x_min,     '--k', 'LineWidth', 1); end
        if ~isempty(dit.rep.x_min), xline(dit.rep.x_min, '--k', 'LineWidth', 1); end
    else
        if ~isempty(unn.x_max),     xline(unn.x_max,     '--k', 'LineWidth', 1); end
    end

    % --- summed clean pulse + the three polynomials ---
    plot(x,  sumUnn,      'Color', cSky, 'LineWidth', 1,   'LineStyle', '--', 'DisplayName', sprintf('Импульс %d без шума', i_nImp));
    plot(xf, unn.pol,     'Color', cSky, 'LineWidth', 1.5, 'DisplayName', sprintf('Полином %d без шума', i_nImp));
    plot(xf, noi.pol,     'Color', 'r',  'LineWidth', 1.5, 'DisplayName', sprintf('Полином %d + шум', i_nImp));
    plot(xf, dit.rep.pol, 'Color', cVio, 'LineWidth', 1.5, 'DisplayName', sprintf('Полином %d + дизеринг', i_nImp));

    % --- extrema markers ---
    plot(unn.sig_x_min, unn.sig_y_min, 'bo', 'MarkerSize', 10, 'DisplayName', 'мин Суммарного импульса');
    plot(unn.x_min,     unn.y_min,     'k*', 'MarkerSize', 10, 'DisplayName', 'мин полинома');

    % --- fzero (exact polynomial minimum) markers on the curve ---
    plot(unn.fzero_x,      unn.fzero_y,      'bd', 'MarkerSize', 9, 'MarkerFaceColor', cSky, 'DisplayName', sprintf('мин Pol без шум %d', i_nImp));
    plot(noi.fzero_x,      noi.fzero_y,      'rd', 'MarkerSize', 9, 'MarkerFaceColor', 'r',  'DisplayName', sprintf('мин Pol + шум %d', i_nImp));
    plot(dit.mean.fzero_x, dit.mean.fzero_y, 'kd', 'MarkerSize', 9, 'MarkerFaceColor', 'k',  'DisplayName', sprintf('мин Pol + диз %d', i_nImp));

    % horizontal ± std bar on the dithered minimum (visible only when it spreads)
    if isfield(dit, 'std') && ~isnan(dit.std.fzero_x) && dit.std.fzero_x > 0
        errorbar(dit.mean.fzero_x, dit.mean.fzero_y, dit.std.fzero_x, 'horizontal', ...
                 'Color', 'k', 'LineWidth', 1, 'CapSize', 8, 'HandleVisibility', 'off');
    end

    % --- polynomial derivatives ---
    plot(xf(1:numel(unn.diff1_pol)),     unn.diff1_pol,     'Color', cSky,  'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', sprintf('diff 1 Pol без шум %d', i_nImp));
    plot(xf(1:numel(unn.diff2_pol)),     unn.diff2_pol,     'Color', cSky2, 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', sprintf('diff 2 Pol без шум %d', i_nImp));
    plot(xf(1:numel(noi.diff1_pol)),     noi.diff1_pol,     'Color', cOrn,  'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', sprintf('diff 1 Pol + шум %d', i_nImp));
    plot(xf(1:numel(noi.diff2_pol)),     noi.diff2_pol,     'Color', cOrn2, 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', sprintf('diff 2 Pol + шум %d', i_nImp));
    plot(xf(1:numel(dit.rep.diff1_pol)), dit.rep.diff1_pol, 'Color', cVio,  'LineWidth', 1.8, 'LineStyle', ':',  'DisplayName', sprintf('diff 1 Pol + диз %d', i_nImp));
    plot(xf(1:numel(dit.rep.diff2_pol)), dit.rep.diff2_pol, 'Color', cVio,  'LineWidth', 1.8, 'LineStyle', ':',  'DisplayName', sprintf('diff 2 Pol + диз %d', i_nImp));
    plot(xf(1:numel(dit.rep.polyder)),   dit.rep.polyder,   'Color', cVio,  'LineWidth', 1.8, 'LineStyle', ':',  'DisplayName', sprintf('polyder d1 диз %d', i_nImp));

    % --- markers where the 1st derivative crosses zero (placed on the d1 curve) ---
    yd1_unn = interpOnDeriv(xf, unn.diff1_pol,     unn.fzero_x);
    yd1_noi = interpOnDeriv(xf, noi.diff1_pol,     noi.fzero_x);
    yd1_dit = interpOnDeriv(xf, dit.rep.diff1_pol, dit.mean.fzero_x);
    plot(unn.fzero_x,      yd1_unn, 'b^', 'MarkerSize', 9, 'MarkerFaceColor', cSky, 'DisplayName', sprintf('d'' Pol без шум %d', i_nImp));
    plot(noi.fzero_x,      yd1_noi, 'r^', 'MarkerSize', 9, 'MarkerFaceColor', 'r',  'DisplayName', sprintf('d'' Pol + шум %d', i_nImp));
    plot(dit.mean.fzero_x, yd1_dit, 'k^', 'MarkerSize', 9, 'MarkerFaceColor', 'k',  'DisplayName', sprintf('d'' Pol + диз %d', i_nImp));

    if ~isempty(xf)
        xlim([min(xf), max(xf)]);
    end
    ylim([-max(unn.pol), 0.3 + max(unn.pol)]);
    hold off;
    % legend show   % <- uncomment to display the legend (DisplayNames are set)
end

% ---------------------------------------------------------------------
%  Small helpers
% ---------------------------------------------------------------------
function v = scalarOrNaN(a)
    % Return the first element, or NaN if the input is empty (so fprintf and
    % markers behave when islocalmin/islocalmax found nothing).
    if isempty(a)
        v = NaN;
    else
        v = a(1);
    end
end

function y = interpOnDeriv(xf, dvals, xq)
    % y-value on a derivative curve at query x, guarding empty / NaN inputs.
    xd = xf(1:numel(dvals));
    if isempty(xq) || any(isnan(xq)) || numel(xd) < 2
        y = NaN;
    else
        y = interp1(xd, dvals, xq, 'makima');
    end
end

% ---------------------------------------------------------------------
%  Quantizer stochastic-resonance functions
% ---------------------------------------------------------------------

% Uniform mid-tread quantizer with step Delta (the LSB).
function q = quantize(v, Delta)
    q = Delta * round(v / Delta);
end

% Sweep noise level σ and measure the error in the recovered pulse position.
%  For each σ: repeat M times { add Gaussian noise to N copies of the signal,
%  quantize each, average them, detect the position with analyzeSignal }.
%  At σ = 0 the copies are identical, so only quantization bias remains; as σ
%  grows the dither linearises the quantizer (error drops), then averaging can
%  no longer suppress the noise (error rises) — the dip is the resonance.
function sr = sweepResonance(x, sig, mask, params)
    Delta = params.quantFrac * max(sig);
    sigma = linspace(0, params.srSigmaMax * Delta, params.srNumSigma);
    N = params.srNumAvg;
    M = params.srNumRepeat;
    L = numel(sig);

    % ground truth: position from the clean, un-quantized signal
    clean  = analyzeSignal(x, sig, mask, params.polyDeg);
    x_true = clean.fzero_x;

    rmse   = nan(1, params.srNumSigma);
    bias   = nan(1, params.srNumSigma);
    sd     = nan(1, params.srNumSigma);
    validN = zeros(1, params.srNumSigma);

    for is = 1:params.srNumSigma
        s_noise = sigma(is);
        xhat = nan(1, M);
        for m = 1:M
            noisy = sig + s_noise * randn(N, L);     % N dithered copies (rows)
            rec   = mean(quantize(noisy, Delta), 1); % average the quantized copies
            s     = analyzeSignal(x, rec, mask, params.polyDeg);
            xhat(m) = s.fzero_x;
        end
        good = xhat(isfinite(xhat));
        validN(is) = numel(good);
        if ~isempty(good)
            err      = good - x_true;
            rmse(is) = sqrt(mean(err.^2));
            bias(is) = mean(err);
            sd(is)   = std(good);
        end
    end

    sr.sigma  = sigma;   sr.Delta  = Delta;   sr.x_true = x_true;
    sr.rmse   = rmse;    sr.bias   = bias;    sr.sd     = sd;
    sr.validN = validN;
end

% Plot the resonance curve (RMSE vs σ/Δ) and mark the optimum.
function plotResonance(sr, iSR)
    figure('Name', sprintf('Stochastic resonance — impulse %d', iSR));
    plot(sr.sigma / sr.Delta, sr.rmse, '-o', 'Color', '#431e61', ...
         'LineWidth', 1.5, 'MarkerFaceColor', '#431e61', 'DisplayName', 'position RMSE');
    hold on; grid on;
    [~, iStar] = min(sr.rmse);
    plot(sr.sigma(iStar)/sr.Delta, sr.rmse(iStar), 'rp', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'r', 'DisplayName', '\sigma^* (resonance)');
    xlabel('noise level  \sigma / \Delta');
    ylabel('position RMSE');
    title(sprintf('Quantizer stochastic resonance (impulse %d)', iSR));
    legend('Location', 'best');
    hold off;
end

% Dither std needed to reach the resonance optimum given the measured noise.
%  σ_d = sqrt(σ*^2 - σ_n^2), clamped at 0 (no dither once σ_n ≥ σ*).
function sigma_d = ditherForResonance(sigmaStar, sigmaMeas)
    sigma_d = sqrt(max(0, sigmaStar^2 - sigmaMeas^2));
end
