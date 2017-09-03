function [nodeBel,edgeBel,partialNodeBel,partialEdgeBel,logZ,logUZ] = crfChain_inferPartiallabel(nodePot,edgePot,y_s,X_s,w,v_start,v_end,v,nFeatures,featureStart)

% disp(' crfChain_inferPartiallabel begin')
[nNodes,nStates] = size(nodePot);

% Forward Pass
alpha = zeros(nNodes,nStates);
alpha(1,:) = nodePot(1,:);
Z(1) = sum(alpha(1,:));
alpha(1,:) = alpha(1,:)/Z(1);
for n = 2:nNodes % Forward Pass
    tmp = repmatC(alpha(n-1,:)',1,nStates).*edgePot;
    alpha(n,:) = nodePot(n,:).*sum(tmp);    
    % Normalize
    Z(n) = sum(alpha(n,:));
    alpha(n,:) = alpha(n,:)/Z(n);
end
subchain = segpartialchain(y_s);
[nSubchain,~] = size(subchain);

partialNodeBel = zeros(size(nodePot));
partialEdgeBel = zeros(nStates,nStates,nNodes-1);
UZ = zeros(0,1);
for s = 1:nSubchain
    if subchain(s,1) == 1
        n = subchain(s,2);
        if n == 1
            UZ(n) = nodePot(n,y_s(n));
        else
            % for labeled subchain located in the middle, 
            % specifically deal with the start of the subchain
            UZ(n) = sum(edgePot(:,y_s(n)).*nodePot(n,y_s(n)));
        end
        n = subchain(s,2)+1;
        while n<=subchain(s,3)
            UZ(n) = edgePot(y_s(n-1),y_s(n))*nodePot(n,y_s(n));
            n = n + 1; 
        end
        % % for labeled data in the midLoc
        partialNodeBel(subchain(s,2):subchain(s,3),:) = -1;
        partialEdgeBel(:,:,subchain(s,2):subchain(s,3)) = -1;
    else    
        if subchain(s,2) ==  1
            startFlag = 'begLoc';
            subv_start = v_start;
            y_start = -1; %default value,will be unused
        else
            startFlag = 'midLoc';
            subv_start = -1*ones(nStates,1);%default value,will be unused
            y_start = y_s(subchain(s,2)-1);            
        end
                
        if subchain(s,3) ==  nNodes
            endFlag = 'endLoc';
            subv_end = v_end;
            y_end = -1; %default value,will be unused            
        else
            endFlag = 'midLoc';
            subv_end = -1*ones(nStates,1);  %default value,will be unused
            y_end = y_s(subchain(s,3)+1);            
        end        
        
        subsentence = [subchain(s,2) subchain(s,3)];
        
        [subnodePot,subedgePot] = crfsubChain_makePotentials(X_s,startFlag,endFlag,...
            y_start,y_end,subv_start,subv_end,w,v,nFeatures,featureStart,subsentence,1);        
        
        [tempZ,tempNodeBel,tempEdgeBel] = crfsubChain_infer(subnodePot,subedgePot);
        UZ(subchain(s,2):subchain(s,3)) = tempZ(1:end);
        partialNodeBel(subchain(s,2):subchain(s,3),:) = tempNodeBel(1:end,:);
        partialEdgeBel(:,:,subchain(s,2):subchain(s,3)-1) = tempEdgeBel(:,:,1:end);        
    end     
end

% Backward Pass
beta = zeros(nNodes,nStates);
beta(nNodes,:) = 1;
for n = nNodes-1:-1:1 % Backward Pass
    tmp = repmatC(nodePot(n+1,:),nStates,1).*edgePot;
    tmp2 = repmatC(beta(n+1,:),nStates,1);
    beta(n,:) = sum(tmp.*tmp2,2)';    
    % Normalize
    beta(n,:) = beta(n,:)/sum(beta(n,:));
end

% Compute Node Beliefs 
nodeBel = zeros(size(nodePot));
for n = 1:nNodes
    tmp = alpha(n,:).*beta(n,:);
    nodeBel(n,:) = tmp/sum(tmp);
end

% Compute Edge Beliefs
edgeBel = zeros(nStates,nStates,nNodes-1);
for n = 1:nNodes-1
    tmp = zeros(nStates);
    for i = 1:nStates
        for j = 1:nStates
            tmp(i,j) = alpha(n,i)*nodePot(n+1,j)*beta(n+1,j)*edgePot(i,j);
        end
    end
    edgeBel(:,:,n) = tmp./sum(tmp(:));
end

% Compute logZ
logZ = sum(log(Z));
%%%B
logUZ = sum(log(UZ)); 
%%%E
