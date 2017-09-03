function [cellinfo,totinfo] = analyzeDataInfo(cellX,cellY)
% % analyze the property of dataset
% cellX = trwplusCellX;
% cellY = trwplusCellY;

cellnum = size(cellX,2);
% % currn: dim1- nodenum; dim2-posNodenum; dim3-negNodenum
cellinfo = zeros(cellnum,5);
totinfo = zeros(1,5);
for c=1:cellnum    
    splX = cellX{1,c};
    splY = cellY{1,c};
    splnum = size(splX,1);
    cellinfo(c,1) = splnum;    
    % % unlabeled nodes Num
    cellinfo(c,2) = sum(-1*ones(splnum,1)==splY(:));
    % % positive nodes Num
    cellinfo(c,3) = sum(ones(splnum,1)==splY(:));
    % % negative nodes Num
    cellinfo(c,4) = sum(2*ones(splnum,1) == splY(:));
    % % Sentence Num
    cellinfo(c,5) = sum(zeros(splnum,1) == splY(:));
    cellinfo(c,1) = splnum-cellinfo(c,5);    
end

totinfo = sum(cellinfo(:,:));