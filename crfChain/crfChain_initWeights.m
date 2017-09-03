function [w,lamda,v_start,v_end,v] = crfChain_initWeights(nFeatures,nStates,type)

if strcmp(type,'zeros')
    initFunc = @zeros;
else
    if strcmp(type,'ones')
        initFunc = @ones;
    else
        initFunc = @randn;
    end
end
v_start = initFunc(nStates,1); % potential for tags to start sentences
v = initFunc(nStates,nStates); % potentials for transitions between tags
v_end = initFunc(nStates,1); % potential for tags to end sentences
w = initFunc(sum(nFeatures)*nStates,1); % potential of tag given individual features
lamda = initFunc(1,1);