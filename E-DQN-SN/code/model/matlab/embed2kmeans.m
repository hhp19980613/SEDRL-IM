function f = embed2kmeans()

load('..//data//embedding.mat');
load('..//data//epinions_UD.mat');
f = 1;
addpath(genpath('..//NNutil'));
matrix = Gwl_ud;
G=graph(double(Gwl_ud));
edgeNum=numedges(G);
%% k-means clustering on embedding features


idx = kmeans(embed,2,'Replicates',20,'MaxIter',20000);


%% compute errors
error=0; %sum of number of positive links between different clusters and negative links within the same cluster
for  i=1:edgeNum
    [s,t]=findedge(G,i);
    if((matrix(s,t)<0)&&(idx(s)==idx(t)))
        %negative links within a cluster
        error=error+1;
    else if((matrix(s,t)>0)&&(idx(s)~=idx(t)))
            % positive links between clusters
            error=error+1;
        end
    end
end

error=error/edgeNum;
f = error

disp(['The error_rate of k-means is ' num2str(error) '.']);
