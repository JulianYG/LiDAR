%  =====================================================================
%   date    : 2015/10/29
%   Author  : Yosuke Konno <ykonno@ok.ctrl.titech.ac.jp>
%   version : 1.0
%
%   Copyright 2015 Okutomi & Tanaka lab, Tokyo Institute of Technology
%   All rights reserved.
%  =====================================================================

function HR_depth = resint(LR_depth, guide, varargin)
% RESINT Upsamples depth map.
%   HR_DEPTH = RESINT(LR_DEPTH, GUIDE) returns an upsampled depth map of  
%   the same size as the GUIDE. The GUIDE must be the same aspect ratio
%   as the LR_DEPTH and the size of it must be an integer multiple of the 
%   size of the LR_DEPTH.
%
%   You can adjust RESINT by using additional parameter/value pairs
%   following syntax. For example:
%   
%       HR_DEPTH = RESINT(LR_DEPTH, GUIDE, ...
%                           PARAM1, VALUE1, PARAM2, VALUE2, ... )
%
%   Parameters:
%       
%       'wsize' - A scalar or two-element vector [height, width] specifying  
%                 a size of local window used in the tentative upsampling.
%                 
%       'eps'   - A scalar, which is larger than zero, specifying the 
%                 coefficient of the regularization term of the objective
%                 function used in the tentative upsampling.  
%
%       'delta' - A scalar, which is larger than zero, specifying the 
%                 threshold value in the weighted averaging in the 
%                 tentative upsampling.
%
%   For detailed information, see the folloing paper.
%  
%   Y. Konno, Y, Monno, D. Kiku, M. Tanaka and M. Okutomi,
%   "Intensity Guided Depth Upsampling by Residual Interpolation",
%   In JSME/RMD International Conference on Advanced Mechatronics, 2015


[LR_depth, guide, scale, GFparams] = parseInputs(LR_depth, guide, varargin{:});

depth_sp = sparseMat(LR_depth, scale);
mask = sparseMat(ones(size(LR_depth)), scale);
HR_depth_tentative = GuidedFilter(guide, depth_sp, mask, GFparams);

residual = LR_depth - HR_depth_tentative(1:scale:end, 1:scale:end);
HR_residual = interpolation(residual, scale);
        
HR_depth = HR_depth_tentative + HR_residual;
HR_depth(HR_depth > 1) = 1;
HR_depth(HR_depth < 0) = 0;
%---------------------------------------------------------------------

%=====================================================================
function [depth, guide, scale, GFparams] = parseInputs(varargin)
% Parse the input arguments, returning the resulting set of parameters.

narginchk(2, Inf);

GFparams = parseGFParams(varargin);
[depth, guide, scale] = parsePreMethodArgs(varargin);
%---------------------------------------------------------------------

%=====================================================================
function [depth, guide, scale] = parsePreMethodArgs(args)
% Parse all the input arguments before the GFoption argument.

idx = find(cellfun('isclass', args, 'char'));

if ~isempty(idx)
    args = args(1:idx(1)-1);
end

if numel(args) < 2
    error('ErrorTests:convertTest',...
          'Function RESIDUALINTERPOLATION needs at least following 2 arguments\n  Low resolution depth map\n  Guide image\n');
end

depth = args{1};
guide = args{2};

validateattributes(depth, {'numeric'}, {'ndims',2}, mfilename, 'Depth', 1);
validateattributes(guide, {'numeric'}, {'nonsparse'}, mfilename, 'Guide', 2);

size_guide = size(guide);
ratio = size_guide(1:2)./size(depth);
valid = abs(ratio(1)-ratio(2))<eps && ...
        prod(abs(ratio-floor(ratio))<eps);

if ~valid
    error('ErrorTests:convertTest',...
          'Size missmatch occured!\nThe size of guide image has to be nth power of 2 of the size of the input depth map.');
else
    scale = ratio(1);
end

if size(guide, 3) ~= 1
    guide = rgb2gray(guide);
end
%---------------------------------------------------------------------

%=====================================================================
function GFparams = parseGFParams(args)

% set default parameter 
GFparams.h = 2;
GFparams.v = 2;
GFparams.eps = 1e-03;
GFparams.delta = 1e-04;

% set numeric parameters
valid_params = {'wsize', 'eps', 'delta'};
param_check_fcns = {@processWsizeParam, @processEpsParam, @processWeightParam};

idx = cell(1,length(valid_params));
for k = 1:length(valid_params)
    idx{k} = find(strncmpi(valid_params{k}, args, numel(valid_params{k})));
end

for k = 1:length(idx)
    if idx{k} == length(args)
        error('Invalid syntax');
    elseif ~isempty(idx{k})
        check_fcn = param_check_fcns{k};
        GFparams = check_fcn(args{idx{k}+1}, GFparams);
    end
end
%---------------------------------------------------------------------

%=====================================================================
function GFparams = processWeightParam(arg, params_in)

valid = (isnumeric(arg) && arg > 0);

if ~valid
    error('Invalid value : delta');
end

GFparams = params_in;
GFparams.delta = arg;
%---------------------------------------------------------------------

%=====================================================================
function GFparams = processWsizeParam(arg, params_in)

valid = ((numel(arg) == 1) || (numel(arg) == 2)) && ...
        prod(abs(arg-floor(arg))<eps) && all(arg > 0);
if ~valid
    error('Invalid value : wsize');
end

GFparams = params_in;
if numel(arg) == 1
    GFparams.h = arg;
    GFparams.v = arg;
else
    GFparams.h = arg(1);
    GFparams.v = arg(2);
end
%---------------------------------------------------------------------

%=====================================================================   
function GFparams = processEpsParam(arg, params_in)

valid = isnumeric(arg) && arg > 0;

if ~valid
    error('Invalid value : eps');
end

GFparams = params_in;
GFparams.eps = arg;
%---------------------------------------------------------------------

%=====================================================================
function output = sparseMat(input, scale)

output = zeros(scale * size(input));
output(1:scale:end, 1:scale:end ) = input;
%---------------------------------------------------------------------

%=====================================================================
function q = GuidedFilter(I, p, M, GFparams)
h = GFparams.h;
v = GFparams.v;
eps = GFparams.eps;
delta = GFparams.delta;
bf = @boxfilter;

% The number of the sammpled pixels in each local patch
N = bf(M, h, v);
N(N == 0) = 1;

mean_I = bf(I.*M, h, v) ./ N;
mean_p = bf(p.*M, h, v) ./ N;
mean_Ip = bf(I.*p.*M, h, v) ./ N;

cov_Ip = mean_Ip - mean_I .* mean_p;
mean_II = bf(I.*I.*M, h, v) ./ N;
var_I = mean_II - mean_I .* mean_I;

% linear coefficients
a = cov_Ip ./ (var_I + eps);
b = mean_p - a .* mean_I;

% calculate the mean square error : (aI+b-p)^2
diff = bf(I.*I.*M, h, v).*a.^2 + N.*b.^2 + bf(p.*p, h ,v) + ...
       2*a.*b.*bf(I.*M, h, v) - 2*b.*bf(p, h, v) - 2*a.*bf(I.*p.*M, h, v);
diff = diff ./ N;
diff(diff < delta) = delta;
diff = 1 ./ diff;
wdiff = bf(diff, h, v);
mean_a = bf(a.*diff, h, v) ./ wdiff; 
mean_b = bf(b.*diff, h, v) ./ wdiff;

q = mean_a .* I + mean_b;
%---------------------------------------------------------------------

%=====================================================================
function imDst = boxfilter(imSrc, h, v)

[H, W] = size(imSrc);
imDst = zeros( size(imSrc) );

if v ~= 0
    %cumulative sum over Y axis
    imCum = cumsum(imSrc, 1);
    %difference over Y axis
    imDst(1:v+1, :) = imCum(1+v:2*v+1, :);
    imDst(v+2:H-v, :) = imCum(2*v+2:H, :) - imCum(1:H-2*v-1, :);
    imDst(H-v+1:H, :) = repmat(imCum(H, :), [v, 1]) - imCum(H-2*v:H-v-1, :);
end

if h ~= 0
    if v ~= 0
        %cumulative sum over X axis
        imCum = cumsum(imDst, 2);
    else
        %cumulative sum over X axis
        imCum = cumsum(imSrc, 2);
    end
    %difference over Y axis
    imDst(:, 1:h+1) = imCum(:, 1+h:2*h+1);
    imDst(:, h+2:W-h) = imCum(:, 2*h+2:W) - imCum(:, 1:W-2*h-1);
    imDst(:, W-h+1:W) = repmat(imCum(:, W), [1, h]) - imCum(:, W-2*h:W-h-1);
end
%---------------------------------------------------------------------

%=====================================================================
function output = interpolation(A, scale)

s = size(A);
A = [A, A(1:s(1), s(2));
     A(s(1), 1:s(2)), A(s(1),s(2))];

s = size(A)*scale;
[X_LR, Y_LR] = meshgrid(1:scale:s(2), 1:scale:s(1));
[X_HR, Y_HR] = meshgrid(1:s(2), 1:s(1));

output = interp2(X_LR, Y_LR, A, X_HR, Y_HR, 'cubic');
output = output(1:end-scale, 1:end-scale);