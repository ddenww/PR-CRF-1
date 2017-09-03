function [nll,glamda] = crfChain_loss_lamda(lamda,X,w,v_start,v_end,v,nStates,nFeatures,featureStart,sentences,ephsilon,b)

nSentences = size(sentences,1);
f = 0;
glamda = 0;

for s = 1:nSentences
    nNodes = sentences(s,2)-sentences(s,1)+1;        
    [nodePot,edgePot] = crfChain_makePotentials_Q(X,w,lamda,v_start,v_end,v,nFeatures,featureStart,sentences,s);
    [nodeBel,~,logZ] = crfChain_infer(nodePot,edgePot);            

    % Subract the log-normalizing constant
    f = f + logZ; 
        
    % Update gradient of node features of logZ
    for n = 1:nNodes               
        for state = 1:nStates                        
            % % 适合对正类进行约束
            E = (state == 1) * nodeBel(n,state); % feature: g(state=1) =1; g(state=0)=0;
            glamda = glamda + E;
        end
    end    
end
glamda = -glamda + (ephsilon*nSentences) +(b*nSentences);

% Make final results
drawnow;
nll = f + ephsilon*lamda*nSentences + lamda*nSentences*b;
