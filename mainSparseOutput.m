addpath(genpath('.\'));
addpath('.\analyzeData');

useMex = 0;
datafile = '.\data\TweetBrowsing';
load(datafile)
Threshold = 0.0001;
trcellnum=size(trXPartialcrf,2);
itercnt = zeros(1,trcellnum);

resfilename = 'result.txt';
resfilename = strcat(date,resfilename);

path = '.\result\'; 
resfilename = strcat(path,resfilename);
fid=fopen(resfilename,'a+');
currPath = pwd;
fprintf(fid,'************************************');
fprintf(fid,'\r\n CurrPath: %s \r\n',currPath);
fprintf(fid,'\r\n %s  Partially labeled Data\r\n ',datestr(now));


[trcellinfo,trtotinfo] = analyzeDataInfo(trXPartialcrf,trYPartialcrf);
[tscellinfo,tstotinfo] = analyzeDataInfo(tsXPartialcrf,tsYPartialcrf);


E = 0.1;
A = [0.01,0.1,1,10];

wt_b = 5;
paracnt = 0;

totparacnt = length(A)*length(wt_b)*length(E);
err = zeros (trcellnum,totparacnt);
prec = zeros (trcellnum,totparacnt);
rec = zeros(trcellnum,totparacnt);
f1 =zeros(trcellnum,totparacnt);
lamdarec = zeros(trcellnum,totparacnt);

fprintf(fid,'#####################The begining of new experiments###################### \r\n');

for ecnt = 1:length(E)
    fprintf(fid,'***************************************************** \r\n');
    for bcnt = 1:length(wt_b)
        for acnt = 1:length(A)
            ephsilon = E(ecnt);        
            alpha1 = A(acnt);
            fprintf(fid,'\r\n the parameterCNT: ecnt = %d, bcnt = %d,  acnt= %d \r\n ', ecnt , bcnt, acnt);
            fprintf(fid,'\r\n the parameters: ephsilon = %f,  alpha1 = %f, wt_b = %f \r\n,  ', ephsilon , alpha1,wt_b(bcnt));
            
            predY = cell(1,trcellnum);
            for i = 1:trcellnum
                disp('processing cellnum:');
                disp(i);
                trX = trwplusCellX{1,i};
                trY = trwplusCellY{1,i};
%                 trX = trXPartialcrf{1,i};
%                 trY = trYPartialcrf{1,i}
              
                trsentences = crfChain_initSentences(trY);  
                
                activeRatio = trcellinfo(i,3)/trcellinfo(i,5);
                %setting the parameter
                b = max(1,trcellinfo(i,1)/trcellinfo(i,5) - wt_b(bcnt)*activeRatio);
                
                nWords = size(trX,1);
                nFeatures = size(trX,2);
                featureStart = cumsum([1 ones(1,nFeatures)]);
                nStates = 2;
                
                [w,lamda,v_start,v_end,v] = crfChain_initWeights(nFeatures,nStates,'ones');
                wv = [w(:);v_start(:);v_end(:);v(:)];
                delta = 1000;
                while delta > Threshold

                    lamda0 = lamda;
                    wv0 = wv;   
               
                    options=optimset('Gradobj','on');
                    lamda = fmincon(@(lamda)crfChain_loss_lamda(lamda,trX,w,v_start,v_end,v,nStates,nFeatures,featureStart,trsentences,ephsilon,b),lamda0,[],[],[],[],0,inf,[],options);
                    disp('line41, optimize wv');

                    [wv] = minFunc(@crfChain_lossTheda,wv,[],lamda,trX,trY,nStates,nFeatures,featureStart,trsentences,alpha1);
                    
                    delta = sum((wv0-wv).*(wv0-wv));
                    delta = delta + (lamda0-lamda)*(lamda0-lamda);
                    
                    itercnt(i) = itercnt(i)+1;
                end
                lamdarec(i,paracnt+1) = lamda;
                
                % % Compute errors with learned parameters
                fprintf('Errors based on most likely sequence with learned parameters:\n');
                [w,v_start,v_end,v] = crfChain_splitWeights_Orig(wv,featureStart,nStates);
                
                trainErr1(i) = crfChain_error(w,v_start,v_end,v,trX,trY,nStates,nFeatures,featureStart,trsentences(:,:),'decode',useMex);

                [trainErr_labelSet1(i,1),trainErr_labelSet1(i,2),trainErr_labelSet1(i,3),trainErr_labelSet1(i,4),trainErr_labelSet1(i,5),trainErr_labelSet1(i,6),trainErr_labelSet1(i,7)]...
                    = crfChain_errorLabelset_Q(w,lamda,v_start,v_end,v,trX,trY,nStates,nFeatures,featureStart,trsentences(:,:),'decode');               
                

                trainErr2(i) = crfChain_error(w,v_start,v_end,v,trX,trY,nStates,nFeatures,featureStart,trsentences(:,:),'infer',useMex);

                [trainErr_labelSet2(i,1),trainErr_labelSet2(i,2),trainErr_labelSet2(i,3),trainErr_labelSet2(i,4),trainErr_labelSet2(i,5),trainErr_labelSet2(i,6),trainErr_labelSet2(i,7)]...
                    = crfChain_errorLabelset_Q(w,lamda,v_start,v_end,v,trX,trY,nStates,nFeatures,featureStart,trsentences(:,:),'infer');
                
                tr_predY{1,i} = crfChain_predY(w,lamda,v_start,v_end,v,trX,trY,nStates,nFeatures,featureStart,trsentences(:,:),'infer');

                tsX = tswplusCellX{1,i};
                tsY = tswplusCellY{1,i};
                tssentences = crfChain_initSentences(tsY);
               
                testErr1(i) = crfChain_error(w,v_start,v_end,v,tsX,tsY,nStates,nFeatures,featureStart,tssentences(:,:),'decode',useMex);
                [testErr_labelSet1(i,1), testErr_labelSet1(i,2),testErr_labelSet1(i,3),testErr_labelSet1(i,4), testErr_labelSet1(i,5),testErr_labelSet1(i,6),testErr_labelSet1(i,7)] ...
                    = crfChain_errorLabelset_Q(w,lamda,v_start,v_end,v,tsX,tsY,nStates,nFeatures,featureStart,tssentences(:,:),'decode');
                
                testErr2(i) = crfChain_error(w,v_start,v_end,v,tsX,tsY,nStates,nFeatures,featureStart,tssentences(:,:),'infer',useMex);
                [testErr_labelSet2(i,1),testErr_labelSet2(i,2),testErr_labelSet2(i,3),testErr_labelSet2(i,4),testErr_labelSet2(i,5),testErr_labelSet2(i,6),testErr_labelSet2(i,7)]...
                    = crfChain_errorLabelset_Q(w,lamda,v_start,v_end,v,tsX,tsY,nStates,nFeatures,featureStart,tssentences(:,:),'infer');
                
                ts_predY{1,i} = crfChain_predY(w,lamda,v_start,v_end,v,tsX,tsY,nStates,nFeatures,featureStart,tssentences(:,:),'infer');
            end
            
            fprintf(fid,'err \t tp \t fp \t acc\t prec \t recall \t F1\r\n');
            paracnt = paracnt + 1;
            
            for i=1:trcellnum
                err(i,paracnt) = testErr_labelSet2(i,1);
                prec(i,paracnt) = testErr_labelSet2(i,5);
                rec(i,paracnt) = testErr_labelSet2(i,6);
                f1(i,paracnt) = testErr_labelSet2(i,7);
            end
            
            % %% Processing The results
            L = length(testErr_labelSet2(1,:));
            fprintf(fid,'featDim: %d \r\n',size(trX,2));
            fprintf(fid,'Training \r\n');
            for i = 1:L
                fprintf(fid,'%.4f \t ',sum(trainErr_labelSet1(:,i))/trcellnum);
            end
            fprintf(fid,'\r\n');
            
            for i = 1:L
                fprintf(fid,'%.4f \t ',sum(trainErr_labelSet2(:,i))/trcellnum);
            end
            fprintf(fid,'\r\n');
            
            tscellnum = trcellnum;
            fprintf(fid,'Testing \r\n');
            for i = 1:L
                fprintf(fid,'%.4f \t ',sum(testErr_labelSet1(:,i))/tscellnum);
            end
            fprintf(fid,'\r\n');
            
            for i = 1:L
                fprintf(fid,'%.4f \t ',sum(testErr_labelSet2(:,i))/tscellnum);
            end
            fprintf(fid,'\r\n');
            
            fprintf(fid,'std of testErr_labelSet2 \r\n');
            for i = 1:L
                fprintf(fid,'%.4f \t ',std(testErr_labelSet2(:,i)));
            end
            fprintf(fid,'\r\n');
        end
    end
end
fprintf(fid,'##################### The ending of experiments ###################### \r\n');
fclose(fid);


