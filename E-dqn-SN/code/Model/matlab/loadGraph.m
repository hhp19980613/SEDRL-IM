function totalNodeNum = init_embed(graphName)
    totalNodeNum = 1;
    addpath(genpath('..//NNutil'));
    load(['..//data//',graphName]); % Gwl_ud row

% ================================= 璇诲彇鏁版嵁 =======================================
    totalNodeNum = size(Gwl_ud,1);

end
% ============================end of loadGraph()=====================================

