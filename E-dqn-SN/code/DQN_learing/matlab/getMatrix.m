function f = init_embed(graphName)
    f = 1;

    addpath(genpath('..//NNutil'));
    load(['..//data//',graphName]); % Gwl_ud row

    matrix = Gwl_ud;
    path = ['..//data//matrix//',graphName];
    save(path,'matrix');
end
