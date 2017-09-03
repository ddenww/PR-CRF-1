function predY = crfChain_predY(w,lamda,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences,type)

wv = [w(:);v_start(:);v_end(:);v(:)];

nSentences = size(sentences,1);

predY = zeros(size(y));

for s = 1:nSentences
    y_s = y(sentences(s,1):sentences(s,2));
    [nodePot,edgePot]=crfChain_makePotentials_Q(X,w,lamda,v_start,v_end,v,nFeatures,featureStart,sentences,s);    

    if strcmp(type,'infer')
        [nodeBel,~,~] = crfChain_infer(nodePot,edgePot);        
        [~, yhat] = max(nodeBel,[],2);        
    else
        yhat = crfChain_decode(nodePot,edgePot);
    end
    nNodes = length(y_s);    
    for n = 1:nNodes
        predY(sentences(s,1) + n-1) = yhat(n);        
    end
end
