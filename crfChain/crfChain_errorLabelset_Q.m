function [err,tpr,fpr,acc,precision,recall,F1] = crfChain_errorLabelset_Q(w,lamda,v_start,v_end,v,X,y,nStates,nFeatures,featureStart,sentences,type)

wv = [w(:);v_start(:);v_end(:);v(:)];

nSentences = size(sentences,1);

err = 0;
Z = 0;
pcnt = 0;
tp = 0;
ncnt = 0;
fp = 0;
fn = 0;
tn = 0;
for s = 1:nSentences
    y_s = y(sentences(s,1):sentences(s,2));    
    [nodePot,edgePot]=crfChain_makePotentials_Q(X,w,lamda,v_start,v_end,v,nFeatures,featureStart,sentences,s);    
    if strcmp(type,'infer')        
        [nodeBel,edgeBel,logZ] = crfChain_infer(nodePot,edgePot);        
        [junk yhat] = max(nodeBel,[],2);
    else
        yhat = crfChain_decode(nodePot,edgePot);
    end
    nNodes = length(y_s);    
    for n = 1:nNodes
        if y_s(n) ~= -1
            Z = Z+1;
            if (yhat(n)~=y_s(n)) 
                err = err + 1;
            end
        end            
        if y_s(n) == 1
            pcnt = pcnt + 1;
            if (yhat(n)==y_s(n)) 
                tp = tp + 1;
            else
                fn = fn + 1;
            end
        end
        if y_s(n) == 2
            ncnt = ncnt + 1;
            if (yhat(n)~=y_s(n))
                fp = fp + 1;
            else
                tn = tn + 1;
            end
        end
    end
end
if Z == 0
    Z = eps;
end
if pcnt == 0
    pcnt = eps;
end
if ncnt == 0
    ncnt = eps;
end
err=err/Z;

acc = (tp+tn)/(pcnt+ncnt);

if ((tp+fp) ~= 0)
    precision = tp/(tp+fp);
else
    precision = tp/eps;
end

% % recall is true positive rate
if ((tp+fn) ~= 0)
    recall = tp/(tp+fn);
else
    recall = tp/eps;
end

if ((precision+recall) ~= 0)
    F1 = 2*precision*recall/(precision+recall);
else
    F1 = 2*precision*recall/eps;
end

tpr = tp/pcnt;
fpr = fp/ncnt;