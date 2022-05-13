function f = init_embed_k_means(graphName,row)
    f = 1;
    disp(['init embedding beginning ']);
    addpath(genpath('..//NNutil'));
    load(['..//data//',graphName]); % Gwl_ud

% ================================= init =======================================
    totalNodeNum = size(Gwl_ud,1);
    matrix = getSubMatrix(Gwl_ud, [(row-1)*1000+1:row*1000]);
    G=graph(matrix);
    edgeNum=numedges(G);
    nodeNum = size(matrix,1);
    rand('state',0)
    disp(['The cluster of test is 2 for virtual node embedding.']);
    iterNum = 1;
    %[idx,minInbalance,minIndex,partitionNum] = Initialization(G,matrix,nodeNum,edgeNum,iterNum);     % 鍒濆鍖栧垎缁?

% ============================start of merge by kmeans=====================================
    %% k-means clustering on embedding features with virtual nodes
    G2=graph(matrix);
    edgeNum2=numedges(G2);
    %% this is the configuration of stacked autoencoder %%
    num_nodes2 = size(matrix,1);                    %number of vertex
    nnsize2 = [num_nodes2 512 256 128 64];        %layer-wised setting
    len = length(nnsize2); % number of layers

    rand('state',0)
    sae2 = saesetup(nnsize2);

    for i = 1: len - 1
        sae2.ae{i}.activation_function       = 'tanh';  %tanh, tanh_opt ,sigm
        sae2.ae{i}.output                    = 'tanh';
        sae2.ae{i}.dropoutFraction           = 0;          %  Dropout fraction, only used for fine tuning
        sae2.ae{i}.momentum                  = 0;          %  Momentum
        sae2.ae{i}.scaling_learningRate      = 0.95;          %  Scaling factor for the learning rate (each epoch)
        sae2.ae{i}.nonSparsityPenalty        = 0;          %  0 indicates Non sparsity penalty
        sae2.ae{i}.sparsityTarget            = 0.01;       %  Sparsity target
        sae2.ae{i}.inputZeroMaskedFraction   = 0.0;        %  Used for Denoising AutoEncoders

        if i==1
            sae2.ae{i}.learningRate              = 0.025;
            sae2.ae{i}.weightPenaltyL2           = 0.05;       %  L2 regularization
        else
            sae2.ae{i}.learningRate              = 0.015;
            sae2.ae{i}.weightPenaltyL2           = 0.25;       %  L2 regularization
        end
    end
    %% hyperparameter settings
    beta=25; %ratio of penalty on reconstruction errors of observed connections over that of unobserved connections

    r=floor(length(find(G2.Edges{:,2}==1))/length(find(G2.Edges{:,2}==-1)));
    % #positive edges/ #negative edges
    % r is the ratio of penalty for reconstruction errors of negative links over that of positive links
    % r is also the ratio of weight of pairwise constraints for negatively connected nodes over that for positively connected nodes

    alfa1=16; %weight of pairwise constraints for 1-st layer of SAE
    alfa2=1.5; %weight of pairwise constraints for deeper layers of SAE
    rep2 = DNESBP_CD(sae2, nnsize2,matrix, beta,r, alfa1,alfa2);
    idx = kmeans(rep2{end},2,'Replicates',20,'MaxIter',1000);
% ============================end of merge by kmeans=====================================
% ============================start of init_embed=====================================
    matrixVN = getMatrixVN(matrix, nodeNum, idx);       % 澧炲姞铏氭嫙鑺傜偣
    G=graph(matrixVN);
    nodeNumVN = size(matrixVN,1);
    nnsize = [nodeNumVN 512 256 128 64];        %layer-wised setting
    len = length(nnsize);                       % number of layers
    rand('state',0)

    sae = saesetup(nnsize);
    for i = 1: len - 1
        sae.ae{i}.activation_function       = 'tanh';  %tanh, tanh_opt ,sigm
        sae.ae{i}.output                    = 'tanh';
        sae.ae{i}.dropoutFraction           = 0;          %  Dropout fraction, only used for fine tuning
        sae.ae{i}.momentum                  = 0;          %  Momentum
        sae.ae{i}.scaling_learningRate      = 0.95;          %  Scaling factor for the learning rate (each epoch)
        sae.ae{i}.nonSparsityPenalty        = 0;          %  0 indicates Non sparsity penalty
        sae.ae{i}.sparsityTarget            = 0.01;       %  Sparsity target
        sae.ae{i}.inputZeroMaskedFraction   = 0.0;        %  Used for Denoising AutoEncoders
        if i==1
            sae.ae{i}.learningRate              = 0.025;
            sae.ae{i}.weightPenaltyL2           = 0.05;       %  L2 regularization
        else
            sae.ae{i}.learningRate              = 0.015;
            sae.ae{i}.weightPenaltyL2           = 0.25;       %  L2 regularization
        end
    end

    %% hyperparameter settings
    beta=25; %ratio of penalty on reconstruction errors of observed connections over that of unobserved connections

    r=floor(length(find(G.Edges{:,2}==1))/length(find(G.Edges{:,2}==-1)));
    % #positive edges/ #negative edges
    % r is the ratio of penalty for reconstruction errors of negative links over that of positive links
    % r is also the ratio of weight of pairwise constraints for negatively connected nodes over that for positively connected nodes

    alfa1=16; %weight of pairwise constraints for 1-st layer of SAE
    alfa2=1.5; %weight of pairwise constraints for deeper layers of SAE

    %% node vector representation learned by DNE-SBP
    rep = DNESBP_CD(sae, nnsize,matrixVN, beta,r, alfa1,alfa2);
    state = rep{1,4};
    save('..//data//init_embed.mat','nodeNum','matrix','idx','state','totalNodeNum');
    disp(['The init embedding of nodes finish']);
end
% ============================end of init_embed=====================================

%% 鏍规嵁鍒嗙粍淇℃伅锛屽皢鏅?氱煩闃垫墿灞曚负甯﹁櫄鎷熻妭鐐圭殑鐭╅樀
function matrixVN = getMatrixVN(matrix, nodeNum, idx)
    col = zeros(nodeNum, 2) * 0;
    row = zeros(2, nodeNum + 2) * 0;
    for i=1:nodeNum
        if(idx(i) == 0)
            col(i,1) = 1;
            row(1,i) = 1;
        else if(idx(i) == 1)
            col(i,2) = 1;
            row(2,i) = 1;
            end
        end
    end
    matrixVN = [matrix,col];
    matrixVN = [matrixVN;row];
    matrixVN = double(matrixVN);
end

%% 按照传入的序号，分割矩阵 如index = [1 3 4],则返回matrix第1 3 4 行和列的九个数构成的矩阵
function subMatrix = getSubMatrix(matrix, index)
    subMatrix = matrix(index, :);
    subMatrix = subMatrix(:, index);
end

