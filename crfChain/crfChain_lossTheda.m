function [nll,g] = crfChain_lossTheda(wv,lamda,X,y,nStates,nFeatures,featureStart,sentences,alpha1)

nSentences = size(sentences,1);
nFeaturesTotal = featureStart(end)-1;

w = reshape(wv(1:nFeaturesTotal*nStates),nFeaturesTotal,nStates);
v_start = wv(nFeaturesTotal*nStates+1:nFeaturesTotal*nStates+nStates);
v_end = wv(nFeaturesTotal*nStates+nStates+1:nFeaturesTotal*nStates+2*nStates);
v = reshape(wv(nFeaturesTotal*nStates+2*nStates+1:end),nStates,nStates);

f = 0;
gw = zeros(featureStart(end)-1,nStates);
gv_start = zeros(nStates,1);
gv_end = zeros(nStates,1);
gv = zeros(nStates);

for s = 1:nSentences
    gw_partiallabel = zeros(featureStart(end)-1,nStates);
    gv_partiallabel_start = zeros(nStates,1);
    gv_partiallabel_end = zeros(nStates,1);
    gv_partiallabel = zeros(nStates);
    nNodes = sentences(s,2)-sentences(s,1)+1;    
    y_s = y(sentences(s,1):sentences(s,2));            
    
    [nodePot,edgePot]=crfChain_makePotentials(X,w,v_start,v_end,v,nFeatures,featureStart,sentences,s);
    X_s = X(sentences(s,1):sentences(s,2),:);    
    
    [nodeBel,edgeBel,partialNodeBel,partialEdgeBel,logZ,logUZ] = ...
        crfChain_inferPartiallabel(nodePot,edgePot,y_s,X_s,w,v_start,v_end,v,nFeatures,featureStart);

    % % Cal the potential and conditional probablity under Q distribution
    [QnodePot,QedgePot] = crfChain_makePotentials_Q(X,w,lamda,v_start,v_end,v,nFeatures,featureStart,sentences,s);
    [QnodeBel,QedgeBel,~] = crfChain_infer(QnodePot,QedgePot);            
    % Subract the log-normalizing constant
    f = f + logUZ - logZ;    
    
    % Calculate the extra term q(y or h|x)*log(p(y or h|x))
    KLterm = 0;    
    % P_nodePot is the nodePot without the consideration of lamda*g
    P_nodePot = zeros(nNodes,nStates);
    
    for n = 1:nNodes
        features = X_s(n,:);
        for state = 1:nStates
            pot = 0;
            for feat = 1:nFeatures
                if features(feat)~=0
                    featureParam = featureStart(feat);
                    pot = pot+features(feat)*w(featureParam+nFeaturesTotal*(state-1));
                end
            end
            P_nodePot(n,state) = pot;            
        end        
                
        for state = 1:nStates
            KLterm = KLterm + P_nodePot(n,state)*QnodeBel(n,state);
        end
    end
    
    P_edgePot = v;    
    for n = 1:nNodes-1
        for state1 = 1:nStates
            for state2 = 1:nStates                ;
                KLterm = KLterm + P_edgePot(state1,state2)*QedgeBel(state1,state2,n);
            end
        end
    end    
    
    KLterm = KLterm - logZ;
    f = f - alpha1*KLterm;
      
    % Update gradient of node features of logZ
    for n = 1:nNodes
        features = X_s(n,:); % features for word w in sentence s
        % % Rev Beg
        for feat = 1:nFeatures
            if features(feat) ~= 0 % we ignore features that are 0                
                featureParam = featureStart(feat);
                for state = 1:nStates                    
                    E = features(feat) * nodeBel(n,state); % feature under expected dist'n
                    gw(featureParam,state) = gw(featureParam,state) + E;                    
                end
            end
        end        
        % % Rev End
    end
    
    % Update gradient of transitions
    %%%Calculate expected gradient of logZ
    for n = 1:nNodes-1
        for state1 = 1:nStates
            for state2 = 1:nStates                
                E = edgeBel(state1,state2,n);
                gv(state1,state2) = gv(state1,state2) + E;
            end
        end
    end
        
    subchain  = segpartialchain(y_s);  
    [nSubchain,~] = size(subchain);
    
    % Update gradient of node features of f
    for sidx = 1:nSubchain        
        if subchain(sidx,1) == 1  % for the labelled subchain            
            % Update gradiet of nodes
            % n is the relative position of data in the subchain
            % X is the input feature of the whole chain
            for n = subchain(sidx,2):subchain(sidx,3)
                % % be careful of the position of current data: sentences(s,1)+ n-1
                features = X_s(n,:);
                for feat = 1:nFeatures
                    % % Rev Beg
                    if features(feat) ~= 0 % we ignore features that are 0
                        featureParam = featureStart(feat);
                        for state = 1:nStates
                            O = (state == y_s(n))* features(feat);
                            gw_partiallabel(featureParam,state) = gw_partiallabel(featureParam,state) + O;
                        end
                    end
                    % % Rev End
                end
            end
            % Update gradiet of transitions
            for n = subchain(sidx,2):subchain(sidx,3)-1
                for state1 = 1:nStates
                    for state2 = 1:nStates
                        O = ((state1 == y_s(n)) && (state2 == y_s(n+1)));    
                        gv_partiallabel(state1,state2) = gv_partiallabel(state1,state2) + O;
                    end
                end
            end                          
        else  %% for the unlabeled subchain                      
            % update gradient of node features (for unlabeled data)            
            for n = subchain(sidx,2):subchain(sidx,3)                
                features = X_s(n,:);
                % % Rev Beg
                for feat = 1:nFeatures
                    if features(feat) ~= 0 % we ignore features that are 0
                        featureParam = featureStart(feat);
                        for state = 1:nStates
                            E = features(feat) * partialNodeBel(n,state);
                            % change the fij
                            gw_partiallabel(featureParam,state) = gw_partiallabel(featureParam,state) + E;
                        end
                    end
                end
                % % Rev End
            end
            % update gradient of transition features (for unlabeled data)
            for n = subchain(sidx,2):subchain(sidx,3)-1
                for state1 = 1:nStates
                    for state2 = 1:nStates
                        E_partiallabel = partialEdgeBel(state1,state2,n);
                        gv_partiallabel(state1,state2) = gv_partiallabel(state1,state2) + E_partiallabel;
                    end
                end
            end
        end
    end       
    
    %%% -(logUZ - logZ)
    gw = gw - gw_partiallabel;
    gv = gv - gv_partiallabel;

% % 关键部分是transition gradient 的更新，因为涉及到相邻位置标注情况

    % Update gradient of BoS and EoS transitions
    for state = 1:nStates
        % % for the gradient of logZ at BoS and EoS
        E = nodeBel(1,state);
        gv_start(state) = gv_start(state) + E;
        
        E = nodeBel(end,state);
        gv_end(state) = gv_end(state)+ E;
        
        % % for the gradient of logUZ at BoS and EoS

        if subchain(1,1) == -1  %% if current state position is unlabelled%             
            E_partiallabel = nodeBel(1,state);
            gv_partiallabel_start(state) = gv_partiallabel_start(state) + E_partiallabel;
        else   %% if current state position is labelled
            O = (state == y_s(1));
            gv_partiallabel_start(state) = gv_partiallabel_start(state) + O;
        end                        
        
        if subchain(end,1) == -1
            E_partiallabel = nodeBel(end,state);
            gv_partiallabel_end(state) = gv_partiallabel_end(state) + E_partiallabel;            
        else
            O = (state == y_s(end));
            gv_partiallabel_end(state) = gv_partiallabel_end(state) + O;
        end        
    end    
    
    gv_start = gv_start - gv_partiallabel_start;
    gv_end = gv_end - gv_partiallabel_end;   
    
    
    % % calculate Eq,Ep   
    
    % Update gradient of node features of logZ_Q
    for n = 1:nNodes
        features = X_s(n,:); % features for word w in sentence s
        % % Rev Beg
        for feat = 1:nFeatures
            if features(feat) ~= 0 % we ignore features that are 0                
                featureParam = featureStart(feat);
                for state = 1:nStates                    
                    E = features(feat) * QnodeBel(n,state); % feature under expected dist'n
                    gw(featureParam,state) = gw(featureParam,state) - alpha1*E;                    
                end
            end
        end        
        % % Rev End
    end
    
    % Update gradient of transitions
    %%%Calculate expected gradient of logZ_Q
    for n = 1:nNodes-1
        for state1 = 1:nStates
            for state2 = 1:nStates                
                E = QedgeBel(state1,state2,n);
                gv(state1,state2) = gv(state1,state2) - alpha1*E;
            end
        end
    end
    
    E = QnodeBel(1,state);
    gv_start(state) = gv_start(state) - alpha1*E;
    
    E = QnodeBel(end,state);
    gv_end(state) = gv_end(state) - alpha1*E;
    
    % Update gradient of node features of logZ
    for n = 1:nNodes
        features = X_s(n,:); % features for word w in sentence s
        % % Rev Beg
        for feat = 1:nFeatures
            if features(feat) ~= 0 % we ignore features that are 0                
                featureParam = featureStart(feat);
                for state = 1:nStates                    
                    E = features(feat) * nodeBel(n,state); % feature under expected dist'n
                    gw(featureParam,state) = gw(featureParam,state) + alpha1*E;                    
                end
            end
        end        
        % % Rev End
    end
    
    % Update gradient of transitions
    %%%Calculate expected gradient of alpha1*logZ_P
    for n = 1:nNodes-1
        for state1 = 1:nStates
            for state2 = 1:nStates                
                E = edgeBel(state1,state2,n);
                gv(state1,state2) = gv(state1,state2) + alpha1*E;
            end
        end
    end
    
    E = QnodeBel(1,state);
    gv_start(state) = gv_start(state) + alpha1*E;
    
    E = QnodeBel(end,state);
    gv_end(state) = gv_end(state)+ alpha1*E;    
end

% Make final results
drawnow;
nll = -f;
g = [gw(:);gv_start;gv_end;gv(:)];
