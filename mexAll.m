% % for windows
% % minFunc
% addpath(genpath('D:\work\2_LEARNING\CRFChain\crfChain_Partial_Regularized'));
% fprintf('Compiling minFunc files...\n');
% %mex -f 'C:\Users\Anku\Documents\Wen\code\gnumex2.06\mexopts.bat' minFunc/lbfgsC.c
% mex minFunc/lbfgsC.c
% % 
% % % KPM
% fprintf('Compiling KPM files...\n');
% mex -IKPM KPM/repmatC.c
% % 
% % % crfChain
% fprintf('Compiling crfChain files...\n');
% mex crfChain/mex/crfChain_makePotentialsC.c
% mex crfChain/mex/crfChain_inferC.c
% mex crfChain/mex/crfChain_lossC2.c

% % for mac
pwd
% addpath(genpath('/Users/zfshao/Documents/MATLAB/crfChain0130'));
addpath(pwd);
% minFunc
fprintf('Compiling minFunc files...\n');
mex -Dchar16_t=uint16_T minFunc/lbfgsC.c

% KPM
fprintf('Compiling KPM files...\n');
mex -Dchar16_t=uint16_T -IKPM KPM/repmatC.c

% crfChain
fprintf('Compiling crfChain files...\n');

mex -Dchar16_t=uint16_T crfChain/mex/crfChain_makePotentialsC.c
mex -Dchar16_t=uint16_T crfChain/mex/crfChain_inferC.c
mex -Dchar16_t=uint16_T crfChain/mex/crfChain_lossC2.c