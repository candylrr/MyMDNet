function vl_setupnn()
% VL_SETUPNN  Setup the MatConvNet toolbox
%    VL_SETUPNN() function adds the MatConvNet toolbox to MATLAB path.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

root = vl_rootnn() ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', 'xtest')) ;
%this function is to Installing and compiling the library MatConvNet,since
%imake it and save the things we want(some projects that end with mex) in 
%/workspace/matconvnet-1.0-beta10/matlab/mex so i just copy the mex to this
%project and use it directly

