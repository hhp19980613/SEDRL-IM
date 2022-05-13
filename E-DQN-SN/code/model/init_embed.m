function f = init_embed(graphName, nodeNum, embedSize)

    f = 1;
    addpath(genpath('matlab//NNutil//'));
    addpath(genpath('matlab//'));
    adj = textread(['..//data//',graphName,'.txt'], '%d');
    adjIndex = 1;
    edgeNum = length(adj)/3;
    matrix = ones(nodeNum, nodeNum);

    for i = 1:edgeNum
        if adj(adjIndex+2) == 1
            matrix(adj(adjIndex),adj(adjIndex+1)) = 1;
        else if adj(adjIndex+2) == -1
            matrix(adj(adjIndex),adj(adjIndex+1)) = -1;
            end
        end
        adjIndex = adjIndex + 3;
    end
    nodeNum = size(matrix,1);
    embedSize = double(embedSize);
% ================================= 璇诲彇鏁版嵁 =======================================
    G_temp = graph(matrix);

    % [idx,minInbalance,minIndex,partitionNum] = Initialization(G_temp,matrix,nodeNum,edgeNum_temp,iterNum);
    G=graph(matrix);

    nnsize = [nodeNum 512 256 128 embedSize];        %layer-wised setting
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
%    rep2 = DNESBP_CD(sae, nnsize,matrix_VN, beta,r, alfa1,alfa2);
    rep2 = DNESBP_CD(sae, nnsize,matrix, beta,r, alfa1,alfa2);
    embedInfo = rep2{1,4};
    save(['..//data//embedding//',graphName,'.txt'],'embedInfo',"-ascii");


end
% ============================end of init_embed()=====================================

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

