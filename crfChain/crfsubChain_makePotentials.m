function [nodePot,edgePot] = crfsubChain_makePotentials(X_s,startFlag,endFlag,...
    y_start,y_end,subv_start,subv_end,w,v,nFeatures,featureStart,sentences,s)
% Make Potentials for Sentence s
nFeaturesTotal = featureStart(end)-1;
nNodes = sentences(s,2)-sentences(s,1)+1;
nStates = length(subv_start);

% Make node potentials
nodePot = zeros(nNodes,nStates);
for n = 1:nNodes
    features = X_s(sentences(s,1)+n-1,:); % features for word w in sentence s
    for state = 1:nStates
        pot = 0;
        for f = 1:nFeatures
            if features(f) ~= 0 % we ignore features that are 0
                % % be careful of the index of w
                featureParam = featureStart(f);
                pot = pot+features(f)*w(featureParam+nFeaturesTotal*(state-1)); 
            end
        end
        nodePot(n,state) = pot;
    end
end

if strcmp(startFlag,'begLoc')  %subchain is on the beggining location of the whole sentence
    nodePot(1,:) = nodePot(1,:) + subv_start'; % Modification for beginning of sentence
    if subv_start == -1*ones(size(subv_start))
        disp('wrong with subv_start in line 27 crfsubChain_makePotentials.m');
    end
end

if strcmp(endFlag,'endLoc')
    nodePot(end,:) = nodePot(end,:) + subv_end'; % Modification for end of sentence
    if subv_end == -1*ones(size(subv_start))
        disp('wrong with subv_end in line 34 crfsubChain_makePotentials.m');
    end
end

if strcmp(startFlag,'midLoc')
    nodePot(1,:) = nodePot(1,:) + v(y_start,:);
end

if strcmp(endFlag,'midLoc')
    nodePot(end,:) = nodePot(end,:) + v(:,y_end)';
end

nodePot = exp(nodePot);

% Transitions are not dependent on features, so are position independent
edgePot = exp(v); 