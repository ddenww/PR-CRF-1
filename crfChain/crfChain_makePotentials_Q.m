function [nodePot,edgePot] = crfChain_makePotentials_Q(X,w,lamda,v_start,v_end,v,nFeatures,featureStart,sentences,s)

nFeaturesTotal = featureStart(end)-1;
nNodes = sentences(s,2)-sentences(s,1)+1;
nStates = length(v_start);

% Make node potentials
nodePot = zeros(nNodes,nStates);
for n = 1:nNodes
    features = X(sentences(s,1)+n-1,:); % features for word w in sentence s
    
    for state = 1:nStates
        pot = 0;
        for f = 1:nFeatures
            if features(f) ~= 0 % we ignore features that are 0
                featureParam = featureStart(f);
                pot = pot+features(f)*w(featureParam+nFeaturesTotal*(state-1)); 
            end
        end   
        if state == 1              % 对正类进行约束
            nodePot(n,state) = pot - state*lamda;
        else
            nodePot(n,state) = pot;
        end        
    end
end
nodePot(1,:) = nodePot(1,:) + v_start'; % Modification for beginning of sentence
nodePot(end,:) = nodePot(end,:) + v_end'; % Modification for end of sentence
nodePot = exp(nodePot);

edgePot = exp(v); 